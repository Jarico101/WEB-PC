from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from .backtest import BASELINE_NAMES, run_backtest, run_baselines
from .config import DEFAULT_INITIAL_BALANCE, LOG_DIR
from .data_adapter import load_history
from .evaluation import classify_evaluation
from .file_policy import resolve_history_path, resolve_log_path, resolve_model_path
from .model_policy import resolve_ppo_model_path


SPLIT_PATHS = {
    "train": Path("data/train.jsonl"),
    "valid": Path("data/valid.jsonl"),
    "test": Path("data/test.jsonl"),
}


def evaluate_model(
    *,
    model_path: str | Path,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    output_path: str | Path = LOG_DIR / "latest_evaluation.json",
) -> dict[str, Any]:
    model = PPO.load(resolve_ppo_model_path(model_path))
    report: dict[str, Any] = {}
    for split_name, split_path in SPLIT_PATHS.items():
        report[split_name] = _evaluate_section(
            split_path,
            model=model,
            initial_balance=initial_balance,
        )

    walk_forward = []
    for index in range(1, 6):
        path = Path(f"data/walk_forward_{index}.jsonl")
        if resolve_history_path(path).exists():
            walk_forward.append(
                _evaluate_section(
                    path,
                    model=model,
                    initial_balance=initial_balance,
                )
            )
    report["walk_forward"] = walk_forward
    report["verdict"] = asdict(classify_evaluation(report))

    output = resolve_log_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _evaluate_section(
    path: str | Path,
    *,
    model: PPO,
    initial_balance: float,
) -> dict[str, Any]:
    history = load_history(resolve_history_path(path))
    baselines = run_baselines(history, initial_balance=initial_balance)

    def policy(obs, info):
        action, _ = model.predict(obs, deterministic=True)
        return [int(value) for value in action]

    agent = run_backtest(history, policy=policy, initial_balance=initial_balance)
    return {
        "source": str(resolve_history_path(path)),
        "agent": _compact_metrics(agent),
        "baselines": {name: _compact_metrics(metrics) for name, metrics in baselines.items()},
    }


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "steps": metrics["steps"],
        "bets": metrics["bets"],
        "bet_rate": metrics.get("bet_rate", round(metrics["bets"] / max(metrics["steps"], 1), 6)),
        "final_balance": metrics["final_balance"],
        "net_profit": metrics["net_profit"],
        "max_drawdown": metrics["max_drawdown"],
    }


def format_chinese_tables(report: dict[str, Any]) -> str:
    lines: list[str] = []
    verdict = report.get("verdict", {})
    lines.append(f"总判定：{verdict.get('status', 'UNKNOWN')}")
    reasons = verdict.get("reasons") or []
    if reasons:
        lines.append(f"原因：{', '.join(str(reason) for reason in reasons)}")
    lines.append("")

    for split_name in ("train", "valid", "test"):
        section = report.get(split_name)
        if isinstance(section, dict):
            lines.extend(_section_table(split_name, section))
            lines.append("")

    walk_forward = report.get("walk_forward") or []
    if walk_forward:
        rows = []
        for index, section in enumerate(walk_forward, start=1):
            status = classify_evaluation({"test": section}).status
            agent = section["agent"]
            rows.append(_metric_cells(f"WF{index}", "PPO", agent, status))
        lines.extend(
            _format_table(
                ["Walk-Forward", "策略", "下注数", "下注率", "最终余额", "盈亏", "最大回撤", "判定"],
                rows,
                right_align={2, 3, 4, 5, 6},
            )
        )
        lines.append("")

    return "\n".join(lines).rstrip()


def _section_table(split_name: str, section: dict[str, Any]) -> list[str]:
    label = {"train": "训练", "valid": "验证", "test": "考试"}.get(split_name, split_name)
    status = classify_evaluation({"test": section}).status
    rows = [_strategy_cells("PPO", section["agent"], status)]
    for name in BASELINE_NAMES:
        baseline = section["baselines"][name]
        rows.append(_strategy_cells(name, baseline, "BASELINE"))
    return [
        f"{label}集",
        *_format_table(
            ["策略", "下注数", "下注率", "最终余额", "盈亏", "最大回撤", "判定"],
            rows,
            right_align={1, 2, 3, 4, 5},
        ),
    ]


def _strategy_cells(name: str, metrics: dict[str, Any], status: str) -> list[str]:
    return [
        name,
        str(metrics["bets"]),
        f"{float(metrics.get('bet_rate', 0.0)) * 100:.2f}%",
        f"{float(metrics['final_balance']):.2f}",
        f"{float(metrics['net_profit']):.2f}",
        f"{float(metrics['max_drawdown']) * 100:.2f}%",
        status,
    ]


def _metric_cells(prefix: str, name: str, metrics: dict[str, Any], status: str) -> list[str]:
    return [
        prefix,
        name,
        str(metrics["bets"]),
        f"{float(metrics.get('bet_rate', 0.0)) * 100:.2f}%",
        f"{float(metrics['final_balance']):.2f}",
        f"{float(metrics['net_profit']):.2f}",
        f"{float(metrics['max_drawdown']) * 100:.2f}%",
        status,
    ]


def _format_table(headers: list[str], rows: list[list[str]], *, right_align: set[int]) -> list[str]:
    table = [headers, *rows]
    widths = [max(_display_width(row[index]) for row in table) for index in range(len(headers))]
    lines = [_format_table_row(headers, widths, right_align)]
    separators = [
        (("-" * max(widths[index] - 1, 1)) + ":") if index in right_align else ("-" * widths[index])
        for index in range(len(headers))
    ]
    lines.append("| " + " | ".join(separators) + " |")
    lines.extend(_format_table_row(row, widths, right_align) for row in rows)
    return lines


def _format_table_row(cells: list[str], widths: list[int], right_align: set[int]) -> str:
    padded = []
    for index, cell in enumerate(cells):
        padding = " " * (widths[index] - _display_width(cell))
        padded.append(f"{padding}{cell}" if index in right_align else f"{cell}{padding}")
    return "| " + " | ".join(padded) + " |"


def _display_width(value: str) -> int:
    width = 0
    for char in str(value):
        width += 2 if ord(char) > 127 else 1
    return width


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the current PPO model against new splits.")
    parser.add_argument("--model", default="ai_agent/models/ppo_pc28_2025_12.zip")
    parser.add_argument("--initial-balance", type=float, default=DEFAULT_INITIAL_BALANCE)
    parser.add_argument("--output", default=str(LOG_DIR / "latest_evaluation.json"))
    parser.add_argument("--table-output", default=str(LOG_DIR / "latest_evaluation_zh.md"))
    args = parser.parse_args()

    report = evaluate_model(
        model_path=args.model,
        initial_balance=args.initial_balance,
        output_path=args.output,
    )
    table = format_chinese_tables(report)
    table_path = resolve_log_path(args.table_output)
    table_path.write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
