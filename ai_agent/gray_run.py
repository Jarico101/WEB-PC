from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .executor import execute_bet
from .file_policy import resolve_history_path, resolve_log_path, resolve_model_path
from .infer import predict_decision


Predictor = Callable[[Path], dict[str, Any]]
Executor = Callable[..., dict[str, Any]]
Sleeper = Callable[[float], None]


@dataclass(frozen=True)
class GrayRunSettings:
    history_path: Path
    model_path: Path
    balance: float = 1000.0
    cycles: int = 1
    interval_seconds: float = 30.0
    dry_run: bool = True
    log_path: Path = Path("ai_agent/logs/gray_run.jsonl")
    executor_api_url: str = "dry-run://local"

    def __post_init__(self) -> None:
        object.__setattr__(self, "history_path", resolve_history_path(self.history_path))
        object.__setattr__(self, "model_path", resolve_model_path(self.model_path))
        object.__setattr__(self, "log_path", resolve_log_path(self.log_path))


def run_gray_cycle(
    settings: GrayRunSettings,
    *,
    predictor: Predictor | None = None,
    executor: Executor = execute_bet,
    sleeper: Sleeper = time.sleep,
) -> list[dict[str, Any]]:
    predict = predictor or (
        lambda history_path: predict_decision(
            history_path,
            model_path=str(settings.model_path),
            balance=settings.balance,
        )
    )
    settings.log_path.parent.mkdir(parents=True, exist_ok=True)

    events: list[dict[str, Any]] = []
    for cycle in range(settings.cycles):
        decision = predict(settings.history_path)
        execution = _execute_decision(settings, decision, executor)
        event = {"cycle": cycle + 1, "decision": decision, "execution": execution}
        events.append(event)
        with settings.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        if cycle < settings.cycles - 1 and settings.interval_seconds > 0:
            sleeper(settings.interval_seconds)

    return events


def _execute_decision(settings: GrayRunSettings, decision: dict[str, Any], executor: Executor) -> dict[str, Any]:
    direction = int(decision.get("direction", 0))
    stake = float(decision.get("stake", 0.0))
    if direction == 0 or stake <= 0:
        return {"ok": True, "dry_run": settings.dry_run, "skipped": True, "reason": "watch"}

    return executor(
        api_url=settings.executor_api_url,
        issue=str(decision.get("issue", "")),
        direction=direction,
        stake=stake,
        dry_run=settings.dry_run,
    )
