"""Validation-only automatic experiment runner for PC28 combo PPO and DQN models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from stable_baselines3.common.callbacks import CheckpointCallback

from .champion_combo import evaluate_candidate_on_validation, select_validation_champion, validation_score
from .config import LOG_DIR, MODEL_DIR
from .markets_combo import CONFIRMED_EXTREME_RANGES, CONFIRMED_ZERO_POLICY
from .train_combo import train_dqn_combo, train_ppo_combo
from .training_protocol_combo import validate_combo_split_manifest


class AutoTuneError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentDirection:
    """An ordered parameter direction; exhaustion moves automatically to the next one."""

    name: str
    candidates: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if not self.name or not self.candidates:
            raise AutoTuneError("each experiment direction requires a name and at least one candidate")


@dataclass(frozen=True)
class AutoTuneConfig:
    total_timesteps: int = 50_000
    initial_balance: float = 10_000.0
    seed: int = 42
    checkpoint_interval_steps: int = 10_000
    no_improvement_patience: int = 2
    max_trials_per_direction: int = 3
    algorithms: tuple[str, ...] = ("ppo", "dqn")
    directions: Mapping[str, tuple[ExperimentDirection, ...]] = field(default_factory=lambda: default_experiment_directions())

    def __post_init__(self) -> None:
        if self.total_timesteps <= 0 or self.checkpoint_interval_steps <= 0:
            raise AutoTuneError("total_timesteps and checkpoint_interval_steps must be positive")
        if self.no_improvement_patience <= 0 or self.max_trials_per_direction <= 0:
            raise AutoTuneError("patience and max_trials_per_direction must be positive")
        if not self.algorithms or any(algorithm not in {"ppo", "dqn"} for algorithm in self.algorithms):
            raise AutoTuneError("algorithms must contain only ppo and/or dqn")
        if any(algorithm not in self.directions or not self.directions[algorithm] for algorithm in self.algorithms):
            raise AutoTuneError("each selected algorithm requires at least one experiment direction")


Trainer = Callable[..., Path]
ValidationEvaluator = Callable[..., dict[str, Any]]


def default_experiment_directions() -> dict[str, tuple[ExperimentDirection, ...]]:
    """Conservative, bounded search directions for unattended simulation experiments."""

    return {
        "ppo": (
            ExperimentDirection("baseline", ({},)),
            ExperimentDirection(
                "learning_rate_entropy",
                (
                    {"learning_rate": 1e-4, "ent_coef": 0.005, "n_steps": 128, "batch_size": 64},
                    {"learning_rate": 5e-4, "ent_coef": 0.01, "n_steps": 128, "batch_size": 64},
                ),
            ),
            ExperimentDirection(
                "horizon_discount",
                (
                    {"learning_rate": 3e-4, "n_steps": 256, "batch_size": 64, "gamma": 0.995, "gae_lambda": 0.97},
                    {"learning_rate": 2e-4, "n_steps": 256, "batch_size": 128, "gamma": 0.99, "clip_range": 0.15},
                ),
            ),
        ),
        "dqn": (
            ExperimentDirection("baseline", ({},)),
            ExperimentDirection(
                "learning_rate_exploration",
                (
                    {"learning_rate": 5e-5, "exploration_fraction": 0.2, "exploration_final_eps": 0.02},
                    {"learning_rate": 2e-4, "exploration_fraction": 0.15, "exploration_final_eps": 0.05},
                ),
            ),
            ExperimentDirection(
                "memory_discount",
                (
                    {"buffer_size": 100_000, "batch_size": 64, "gamma": 0.995, "target_update_interval": 5_000},
                    {"buffer_size": 50_000, "batch_size": 128, "tau": 0.5, "gradient_steps": 2},
                ),
            ),
        ),
    }


def run_combo_auto_tuning(
    *,
    manifest_path: str | Path,
    config: AutoTuneConfig = AutoTuneConfig(),
    model_dir: str | Path = MODEL_DIR / "combo_auto_tune",
    log_dir: str | Path = LOG_DIR / "combo_auto_tune",
    run_id: str | None = None,
    trainers: Mapping[str, Trainer] | None = None,
    validation_evaluator: ValidationEvaluator | None = None,
) -> dict[str, Any]:
    """Run bounded Train/Validation experiments without opening the sealed Test file.

    Test is deliberately absent from this function's validation and data-loading
    calls. Test confirmation remains the separate Champion operation.
    """

    manifest = validate_combo_split_manifest(manifest_path, include_test=False)
    if manifest.get("test_sealed") is not True:
        raise AutoTuneError("automatic tuning requires a manifest with a sealed Test split")
    train_rows = _read_split(manifest, manifest_path, "train")
    validation_rows = _read_split(manifest, manifest_path, "validation")
    fingerprint = _validation_fingerprint(manifest)
    started_at = datetime.now(timezone.utc)
    identifier = run_id or started_at.strftime("%Y%m%dT%H%M%SZ")
    model_root = Path(model_dir) / identifier
    log_root = Path(log_dir)
    trial_root = log_root / identifier
    model_root.mkdir(parents=True, exist_ok=True)
    trial_root.mkdir(parents=True, exist_ok=True)

    trainer_by_algorithm = dict(trainers or {"ppo": train_ppo_combo, "dqn": train_dqn_combo})
    evaluator = validation_evaluator or evaluate_candidate_on_validation
    reports = _previous_candidate_reports(log_root / "candidate_champion.json", fingerprint)
    trials: list[dict[str, Any]] = []
    best_score: tuple[float, float, float, float] | None = _best_score(reports)

    for algorithm in config.algorithms:
        if algorithm not in trainer_by_algorithm:
            raise AutoTuneError(f"no trainer configured for {algorithm}")
        for direction in config.directions[algorithm]:
            stale_trials = 0
            direction_trials: list[dict[str, Any]] = []
            for candidate_index, raw_parameters in enumerate(direction.candidates[: config.max_trials_per_direction], start=1):
                trial_id = f"{algorithm}-{direction.name}-{candidate_index:02d}"
                parameters = dict(raw_parameters)
                model_path = model_root / f"{trial_id}.zip"
                checkpoint_dir = model_root / f"{trial_id}.checkpoints"
                checkpoint = CheckpointCallback(
                    save_freq=config.checkpoint_interval_steps,
                    save_path=str(checkpoint_dir),
                    name_prefix=trial_id,
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                )
                record = {
                    "schema_version": "pc28-combo-auto-tune-v1",
                    "run_id": identifier,
                    "trial_id": trial_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "algorithm": algorithm,
                    "direction": direction.name,
                    "candidate_index": candidate_index,
                    "hyperparameters": parameters,
                    "seed": config.seed + len(trials),
                    "total_timesteps": config.total_timesteps,
                    "checkpoint_interval_steps": config.checkpoint_interval_steps,
                    "training_split": fingerprint["train"],
                    "validation_split": fingerprint["validation"],
                    "validation_rows": len(validation_rows),
                    "test_sealed": True,
                    "test_accessed": False,
                    "model_path": str(model_path),
                }
                _write_json(trial_root / f"{trial_id}.json", {**record, "status": "running"})
                try:
                    output = trainer_by_algorithm[algorithm](
                        train_rows,
                        model_path=model_path,
                        total_timesteps=config.total_timesteps,
                        initial_balance=config.initial_balance,
                        seed=record["seed"],
                        zero_policy=CONFIRMED_ZERO_POLICY,
                        extreme_ranges=CONFIRMED_EXTREME_RANGES,
                        hyperparameters=parameters,
                        checkpoint_callback=checkpoint,
                        experiment_metadata={
                            "run_id": identifier,
                            "trial_id": trial_id,
                            "direction": direction.name,
                            "validation_fingerprint": fingerprint,
                        },
                    )
                    report = evaluator(output, algorithm=algorithm, manifest_path=manifest_path)
                    if report.get("phase") != "validation":
                        raise AutoTuneError("automatic tuning accepts Validation reports only")
                    verdict = str(report.get("evaluation", {}).get("verdict", {}).get("status", "FAIL"))
                    score = validation_score(report.get("evaluation", {}))
                    eligible = verdict == "PASS"
                    improved = eligible and (best_score is None or score > best_score)
                    if improved:
                        best_score = score
                        stale_trials = 0
                    else:
                        stale_trials += 1
                    if eligible:
                        reports.append(report)
                    result = {
                        **record,
                        "status": "completed",
                        "model_path": str(output),
                        "validation_status": verdict,
                        "validation_score": list(score),
                        "improved_candidate": improved,
                        "validation_report": report,
                        "checkpoint_paths": _checkpoint_paths(checkpoint_dir, Path(output)),
                    }
                except Exception as exc:
                    stale_trials += 1
                    result = {**record, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "checkpoint_paths": _checkpoint_paths(checkpoint_dir, model_path)}
                _write_json(trial_root / f"{trial_id}.json", result)
                trials.append(result)
                direction_trials.append(result)
                if reports:
                    interim_selection = select_validation_champion(reports)
                    _write_json(
                        log_root / "candidate_champion.json",
                        _candidate_champion_state(fingerprint, interim_selection, reports),
                    )
                if stale_trials >= config.no_improvement_patience:
                    break
            trials[-1]["direction_stop_reason"] = (
                "no_improvement_patience_reached"
                if stale_trials >= config.no_improvement_patience
                else "candidate_list_exhausted"
            )
            _write_json(trial_root / f"{direction_trials[-1]['trial_id']}.json", trials[-1])

    selection = select_validation_champion(reports)
    state = _candidate_champion_state(fingerprint, selection, reports)
    _write_json(log_root / "candidate_champion.json", state)
    result = {
        "schema_version": "pc28-combo-auto-tune-run-v1",
        "run_id": identifier,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config": _config_metadata(config),
        "validation_fingerprint": fingerprint,
        "test_accessed": False,
        "trials": trials,
        "candidate_champion": selection,
        "candidate_champion_path": str(log_root / "candidate_champion.json"),
    }
    _write_json(trial_root / "run.json", result)
    return result


def _read_split(manifest: dict[str, Any], manifest_path: str | Path, name: str) -> list[dict[str, Any]]:
    raw = Path(manifest["splits"][name]["path"])
    path = raw if raw.is_absolute() else Path(manifest_path).parent / raw.name
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validation_fingerprint(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": manifest["protocol"],
        "train": {key: manifest["splits"]["train"][key] for key in ("rows", "first_issue", "last_issue", "sha256")},
        "validation": {key: manifest["splits"]["validation"][key] for key in ("rows", "first_issue", "last_issue", "sha256")},
    }


def _previous_candidate_reports(path: Path, fingerprint: dict[str, Any]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    previous = json.loads(path.read_text(encoding="utf-8"))
    report = previous.get("selected_report")
    return [report] if previous.get("validation_fingerprint") == fingerprint and isinstance(report, dict) else []


def _best_score(reports: Iterable[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    scores = [validation_score(report.get("evaluation", {})) for report in reports]
    return max(scores) if scores else None


def _selected_report(selection: dict[str, Any], reports: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    selected = selection.get("selected") or {}
    for report in reports:
        if report.get("model_path") == selected.get("model_path"):
            return report
    return None


def _candidate_champion_state(
    fingerprint: dict[str, Any],
    selection: dict[str, Any],
    reports: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "pc28-combo-candidate-champion-v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "validation_fingerprint": fingerprint,
        "status": selection["status"],
        "selection": selection,
        "selected_report": _selected_report(selection, reports),
        "automatic_runtime_activation": False,
        "test_accessed": False,
    }


def _checkpoint_paths(checkpoint_dir: Path, model_path: Path) -> list[str]:
    paths = sorted(checkpoint_dir.glob("*.zip")) if checkpoint_dir.exists() else []
    if model_path.exists():
        paths.append(model_path)
    return [str(path) for path in dict.fromkeys(paths)]


def _config_metadata(config: AutoTuneConfig) -> dict[str, Any]:
    return {
        **asdict(config),
        "directions": {
            algorithm: [{"name": direction.name, "candidates": [dict(candidate) for candidate in direction.candidates]} for direction in directions]
            for algorithm, directions in config.directions.items()
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
