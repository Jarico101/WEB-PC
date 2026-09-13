"""Scheduled, simulation-only challenger training for the PC28 combo policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LOG_DIR, MODEL_DIR
from .data_adapter import clean_combo_history
from .evaluate_combo import evaluate_combo_model
from .markets_combo import CONFIRMED_EXTREME_RANGES, CONFIRMED_ZERO_POLICY
from .train_combo import train_dqn_combo, train_ppo_combo


@dataclass(frozen=True)
class ChallengerConfig:
    cadence_settlements: int = 100
    train_rows: int = 8_000
    validation_rows: int = 1_000
    test_rows: int = 1_000
    window_size: int = 200
    windows: int = 5
    total_timesteps: int = 50_000
    initial_balance: float = 10_000.0
    seed: int = 42
    algorithm: str = "ppo"

    def __post_init__(self) -> None:
        if self.cadence_settlements <= 0:
            raise ValueError("cadence_settlements must be positive")
        if self.train_rows <= 0 or self.validation_rows <= 0 or self.test_rows <= 0:
            raise ValueError("train_rows, validation_rows and test_rows must be positive")
        if self.window_size * self.windows != self.validation_rows:
            raise ValueError("validation_rows must equal window_size * windows")
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.algorithm not in {"ppo", "dqn"}:
            raise ValueError("algorithm must be ppo or dqn")


def challenger_due(settled_rounds: int, *, config: ChallengerConfig = ChallengerConfig()) -> bool:
    """Return true only at an exact recurring settled-round milestone."""

    return settled_rounds > 0 and settled_rounds % config.cadence_settlements == 0


def train_challenger(
    history_source: Any,
    *,
    config: ChallengerConfig = ChallengerConfig(),
    model_dir: str | Path = MODEL_DIR,
    log_dir: str | Path = LOG_DIR,
) -> dict[str, Any]:
    """Train and gate a candidate; never replace or write a Champion pointer."""

    rows = clean_combo_history(history_source, zero_policy=CONFIRMED_ZERO_POLICY, min_rows=1)
    required = config.train_rows + config.validation_rows + config.test_rows
    if len(rows) < required:
        raise ValueError(f"need at least {required} chronological rows for challenger training")

    full_window = rows[-required:]
    evaluation_context = full_window[: config.train_rows + config.validation_rows]
    training_rows = evaluation_context[: config.train_rows]
    training_end_issue = training_rows[-1]["issue"]
    model_root = Path(model_dir)
    model_root.mkdir(parents=True, exist_ok=True)
    model_path = model_root / f"{config.algorithm}_combo_challenger_{training_end_issue}.zip"
    trainer = train_ppo_combo if config.algorithm == "ppo" else train_dqn_combo
    trainer(
        training_rows,
        model_path=model_path,
        total_timesteps=config.total_timesteps,
        initial_balance=config.initial_balance,
        seed=config.seed,
        zero_policy=CONFIRMED_ZERO_POLICY,
        extreme_ranges=CONFIRMED_EXTREME_RANGES,
    )
    return gate_challenger(
        evaluation_context,
        model_path=model_path,
        config=config,
        log_dir=log_dir,
    )


def gate_challenger(
    evaluation_context: Any,
    *,
    model_path: str | Path,
    config: ChallengerConfig = ChallengerConfig(),
    log_dir: str | Path = LOG_DIR,
) -> dict[str, Any]:
    """Evaluate an already trained candidate without training or promoting it."""

    rows = clean_combo_history(evaluation_context, zero_policy=CONFIRMED_ZERO_POLICY, min_rows=1)
    required = config.train_rows + config.validation_rows
    if len(rows) < required:
        raise ValueError(f"need at least {required} chronological rows for challenger evaluation")
    context = rows[-required:]
    metadata_path = Path(model_path).with_name(f"{Path(model_path).stem}.metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    training_end_issue = str(metadata.get("training_data_end_issue", ""))
    evaluation = evaluate_combo_model(
        context,
        model_path=model_path,
        algorithm=config.algorithm,
        window_size=config.window_size,
        windows=config.windows,
        initial_balance=config.initial_balance,
        zero_policy=CONFIRMED_ZERO_POLICY,
        extreme_ranges=CONFIRMED_EXTREME_RANGES,
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "challenger",
        "promotion_eligible": evaluation["verdict"]["status"] == "PASS",
        "automatic_promotion": False,
        "model_path": str(model_path),
        "training_data_end_issue": training_end_issue,
        "validation_start_issue": context[config.train_rows]["issue"],
        "config": asdict(config),
        "evaluation": evaluation,
    }
    output = Path(log_dir) / f"{config.algorithm}_combo_challenger_{training_end_issue}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    result["report_path"] = str(output)
    return result
