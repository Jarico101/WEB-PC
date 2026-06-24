from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import LOG_DIR
from .file_policy import resolve_log_path


@dataclass(frozen=True)
class EvaluationThresholds:
    max_drawdown: float = 0.35
    min_bets: int = 30
    min_bet_rate: float = 0.02
    min_walk_forward_pass_rate: float = 0.60


@dataclass(frozen=True)
class EvaluationVerdict:
    status: str
    reasons: list[str]
    thresholds: dict[str, Any]


def classify_evaluation(
    report: dict[str, Any],
    *,
    thresholds: EvaluationThresholds = EvaluationThresholds(),
) -> EvaluationVerdict:
    train = _section(report, "train")
    valid = _section(report, "valid")
    test = _section(report, "test")
    walk_forward = report.get("walk_forward") or []
    reasons: list[str] = []

    if test is None:
        return EvaluationVerdict(
            status="FAILED",
            reasons=["missing_test_result"],
            thresholds=asdict(thresholds),
        )

    train_good = _section_passes(train, thresholds) if train else False
    valid_good = _section_passes(valid, thresholds) if valid else False
    test_beats = _beats_all_baselines(test)
    test_drawdown_ok = _drawdown_ok(test, thresholds)
    test_bets_ok = _bets_ok(test, thresholds)
    test_good = test_beats and test_drawdown_ok and test_bets_ok
    walk = _walk_forward_summary(walk_forward, thresholds)

    if train_good and ((valid is not None and not valid_good) or not test_good):
        if not test_beats:
            reasons.append("train_good_but_test_does_not_beat_baselines")
        if not test_drawdown_ok:
            reasons.append("train_good_but_test_drawdown_too_high")
        if not test_bets_ok:
            reasons.append("train_good_but_test_bets_too_few")
        if valid is not None and not valid_good:
            reasons.append("train_good_but_valid_weak")
        return EvaluationVerdict("OVERFIT", reasons or ["train_good_valid_or_test_bad"], asdict(thresholds))

    if not test_beats:
        return EvaluationVerdict("FAILED", ["test_does_not_beat_all_baselines"], asdict(thresholds))
    if not test_drawdown_ok:
        return EvaluationVerdict("FAILED", ["test_drawdown_too_high"], asdict(thresholds))

    if test_good and walk["has_segments"] and walk["pass_rate"] >= thresholds.min_walk_forward_pass_rate:
        return EvaluationVerdict("PASS", [f"walk_forward_pass_rate={walk['pass_rate']:.3f}"], asdict(thresholds))

    if test_good:
        reasons.append("test_beats_baselines_but_walk_forward_insufficient")
    if not test_bets_ok:
        reasons.append("test_bets_too_few")
    if walk["has_segments"]:
        reasons.append(f"walk_forward_pass_rate={walk['pass_rate']:.3f}")
    else:
        reasons.append("missing_walk_forward")
    return EvaluationVerdict("WEAK", reasons, asdict(thresholds))


def _section(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    section = report.get(name)
    return section if isinstance(section, dict) else None


def _section_passes(section: dict[str, Any] | None, thresholds: EvaluationThresholds) -> bool:
    return bool(section and _beats_all_baselines(section) and _drawdown_ok(section, thresholds) and _bets_ok(section, thresholds))


def _beats_all_baselines(section: dict[str, Any]) -> bool:
    agent = section.get("agent") if isinstance(section.get("agent"), dict) else section
    baselines = section.get("baselines", {})
    if not isinstance(baselines, dict) or not baselines:
        return False
    agent_profit = _metric(agent, "net_profit")
    return all(agent_profit > _metric(baseline, "net_profit") for baseline in baselines.values() if isinstance(baseline, dict))


def _drawdown_ok(section: dict[str, Any], thresholds: EvaluationThresholds) -> bool:
    agent = section.get("agent") if isinstance(section.get("agent"), dict) else section
    return _metric(agent, "max_drawdown") <= thresholds.max_drawdown


def _bets_ok(section: dict[str, Any], thresholds: EvaluationThresholds) -> bool:
    agent = section.get("agent") if isinstance(section.get("agent"), dict) else section
    steps = max(1, int(_metric(agent, "steps")))
    bets = int(_metric(agent, "bets"))
    return bets >= thresholds.min_bets and (bets / steps) >= thresholds.min_bet_rate


def _walk_forward_summary(segments: Any, thresholds: EvaluationThresholds) -> dict[str, Any]:
    if not isinstance(segments, list) or not segments:
        return {"has_segments": False, "passed": 0, "total": 0, "pass_rate": 0.0}
    passed = sum(1 for segment in segments if isinstance(segment, dict) and _section_passes(segment, thresholds))
    return {"has_segments": True, "passed": passed, "total": len(segments), "pass_rate": passed / len(segments)}


def _metric(metrics: dict[str, Any], name: str) -> float:
    try:
        return float(metrics.get(name, 0.0))
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an agent evaluation report as PASS/WEAK/FAILED/OVERFIT.")
    parser.add_argument("report", help="JSON evaluation report under ai_agent/logs")
    parser.add_argument("--max-drawdown", type=float, default=EvaluationThresholds.max_drawdown)
    parser.add_argument("--min-bets", type=int, default=EvaluationThresholds.min_bets)
    parser.add_argument("--min-bet-rate", type=float, default=EvaluationThresholds.min_bet_rate)
    parser.add_argument("--min-walk-forward-pass-rate", type=float, default=EvaluationThresholds.min_walk_forward_pass_rate)
    args = parser.parse_args()

    report_path = resolve_log_path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verdict = classify_evaluation(
        report,
        thresholds=EvaluationThresholds(
            max_drawdown=args.max_drawdown,
            min_bets=args.min_bets,
            min_bet_rate=args.min_bet_rate,
            min_walk_forward_pass_rate=args.min_walk_forward_pass_rate,
        ),
    )
    output_path = LOG_DIR / "latest_verdict.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(verdict), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(asdict(verdict), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
