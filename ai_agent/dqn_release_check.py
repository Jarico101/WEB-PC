from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import LOG_DIR
from .evaluate_current import _format_table
from .file_policy import resolve_log_path


@dataclass(frozen=True)
class DqnReleaseThresholds:
    min_profit_window_rate: float = 0.60
    max_drawdown: float = 0.50
    min_final_balance: float = 500.0
    max_bets: int = 45
    min_cashout_bets: int = 15
    min_complete_bets: int = 30
    max_top_profit_share: float = 0.75
    ignore_model_violations: bool = False


def classify_dqn_release(
    report: dict[str, Any],
    *,
    thresholds: DqnReleaseThresholds = DqnReleaseThresholds(),
) -> dict[str, Any]:
    episodes = [episode for episode in report.get("episodes", []) if isinstance(episode, dict)]
    total = len(episodes)
    total_pnl = sum(_float(episode, "pnl") for episode in episodes)
    profit_windows = sum(1 for episode in episodes if _float(episode, "pnl") > 0.0)
    profit_window_rate = profit_windows / max(total, 1)
    max_drawdown = max((_float(episode, "max_drawdown") for episode in episodes), default=0.0)
    min_final_balance = min((_float(episode, "final_balance") for episode in episodes), default=0.0)
    max_bets = max((_int(episode, "bets") for episode in episodes), default=0)
    positive_pnls = sorted((_float(episode, "pnl") for episode in episodes if _float(episode, "pnl") > 0.0), reverse=True)
    top_profit_share = (positive_pnls[0] / sum(positive_pnls)) if positive_pnls else 1.0
    violations = [_episode_violation(index, episode, thresholds) for index, episode in enumerate(episodes, start=1)]
    violations = [violation for violation in violations if violation["reasons"]]

    checks = [
        {
            "name": "profit_window_rate",
            "threshold": f">= {thresholds.min_profit_window_rate:.0%}",
            "value": f"{profit_window_rate:.2%}",
            "passed": profit_window_rate >= thresholds.min_profit_window_rate,
        },
        {
            "name": "total_pnl",
            "threshold": "> 0",
            "value": round(total_pnl, 6),
            "passed": total_pnl > 0.0,
        },
        {
            "name": "max_drawdown",
            "threshold": f"<= {thresholds.max_drawdown:.0%}",
            "value": f"{max_drawdown:.2%}",
            "passed": max_drawdown <= thresholds.max_drawdown,
        },
        {
            "name": "min_final_balance",
            "threshold": f">= {thresholds.min_final_balance:.2f}",
            "value": round(min_final_balance, 6),
            "passed": min_final_balance >= thresholds.min_final_balance,
        },
        {
            "name": "max_bets_per_window",
            "threshold": f"<= {thresholds.max_bets}",
            "value": max_bets,
            "passed": max_bets <= thresholds.max_bets,
        },
        {
            "name": "top_profit_share",
            "threshold": f"<= {thresholds.max_top_profit_share:.0%}",
            "value": f"{top_profit_share:.2%}",
            "passed": top_profit_share <= thresholds.max_top_profit_share,
        },
        {
            "name": "release_violations",
            "threshold": "0",
            "value": len(violations),
            "passed": not violations,
        },
    ]
    status = "PASS" if all(check["passed"] for check in checks) else "FAIL"
    return {
        "status": status,
        "thresholds": asdict(thresholds),
        "summary": {
            "episodes": total,
            "profit_windows": profit_windows,
            "profit_window_rate": round(profit_window_rate, 6),
            "total_pnl": round(total_pnl, 6),
            "max_drawdown": round(max_drawdown, 6),
            "min_final_balance": round(min_final_balance, 6),
            "max_bets": max_bets,
            "top_profit_share": round(top_profit_share, 6),
            "violations": len(violations),
        },
        "checks": checks,
        "violations": violations,
    }


def format_release_report(result: dict[str, Any]) -> str:
    checks = [
        [
            check["name"],
            str(check["threshold"]),
            str(check["value"]),
            "PASS" if check["passed"] else "FAIL",
        ]
        for check in result["checks"]
    ]
    lines = [f"# DQN Release Check: {result['status']}", ""]
    lines.extend(_format_table(["check", "threshold", "value", "result"], checks, right_align={1, 2}))
    if result["violations"]:
        rows = [
            [
                str(item["window"]),
                str(item["start_step"]),
                item["result"],
                str(item["bets"]),
                f"{item['pnl']:.2f}",
                f"{item['final_balance']:.2f}",
                f"{item['max_drawdown'] * 100:.2f}%",
                "; ".join(item["reasons"]),
            ]
            for item in result["violations"]
        ]
        lines.extend(["", "## Violations", ""])
        lines.extend(
            _format_table(
                ["window", "start", "result", "bets", "pnl", "final_balance", "drawdown", "reasons"],
                rows,
                right_align={0, 1, 3, 4, 5, 6},
            )
        )
    return "\n".join(lines) + "\n"


def _episode_violation(index: int, episode: dict[str, Any], thresholds: DqnReleaseThresholds) -> dict[str, Any]:
    result = str(episode.get("result", ""))
    reason = str(episode.get("reason", ""))
    bets = _int(episode, "bets")
    reasons: list[str] = []

    if result == "violation" and not thresholds.ignore_model_violations:
        reasons.append(reason or "model_violation")
    if reason in {"risk_cashout", "risk_cashout_sprint_failed"} and bets < thresholds.min_cashout_bets:
        reasons.append(f"cashout_bets_below_{thresholds.min_cashout_bets}")
    if result in {"complete", "fail"} and bets < thresholds.min_complete_bets:
        reasons.append(f"complete_bets_below_{thresholds.min_complete_bets}")
    if bets > thresholds.max_bets:
        reasons.append(f"bets_above_{thresholds.max_bets}")

    return {
        "window": index,
        "start_step": _int(episode, "start_step"),
        "result": result,
        "reason": reason,
        "bets": bets,
        "pnl": round(_float(episode, "pnl"), 6),
        "final_balance": round(_float(episode, "final_balance"), 6),
        "max_drawdown": round(_float(episode, "max_drawdown"), 6),
        "reasons": reasons,
    }


def _float(item: dict[str, Any], key: str) -> float:
    try:
        return float(item.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _int(item: dict[str, Any], key: str) -> int:
    try:
        return int(item.get(key, 0))
    except (TypeError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply DQN eval-only release thresholds to an evaluation JSON report.")
    parser.add_argument("report", help="DQN evaluation JSON report under ai_agent/logs")
    parser.add_argument("--output", default=str(LOG_DIR / "dqn_release_check.json"))
    parser.add_argument("--table-output", default=str(LOG_DIR / "dqn_release_check.md"))
    parser.add_argument("--min-profit-window-rate", type=float, default=DqnReleaseThresholds.min_profit_window_rate)
    parser.add_argument("--max-drawdown", type=float, default=DqnReleaseThresholds.max_drawdown)
    parser.add_argument("--min-final-balance", type=float, default=DqnReleaseThresholds.min_final_balance)
    parser.add_argument("--max-bets", type=int, default=DqnReleaseThresholds.max_bets)
    parser.add_argument("--min-cashout-bets", type=int, default=DqnReleaseThresholds.min_cashout_bets)
    parser.add_argument("--min-complete-bets", type=int, default=DqnReleaseThresholds.min_complete_bets)
    parser.add_argument("--max-top-profit-share", type=float, default=DqnReleaseThresholds.max_top_profit_share)
    parser.add_argument("--ignore-model-violations", action="store_true")
    args = parser.parse_args()

    report_path = resolve_log_path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = classify_dqn_release(
        report,
        thresholds=DqnReleaseThresholds(
            min_profit_window_rate=args.min_profit_window_rate,
            max_drawdown=args.max_drawdown,
            min_final_balance=args.min_final_balance,
            max_bets=args.max_bets,
            min_cashout_bets=args.min_cashout_bets,
            min_complete_bets=args.min_complete_bets,
            max_top_profit_share=args.max_top_profit_share,
            ignore_model_violations=args.ignore_model_violations,
        ),
    )
    output_path = resolve_log_path(args.output)
    table_path = resolve_log_path(args.table_output)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    table = format_release_report(result)
    table_path.write_text(table, encoding="utf-8")
    print(table)
    print(f"saved release check: {output_path}")
    print(f"saved release table: {table_path}")


if __name__ == "__main__":
    main()
