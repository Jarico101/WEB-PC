from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from .backtest import run_backtest
from .config import DEFAULT_INITIAL_BALANCE, LOG_DIR
from .data_adapter import load_history
from .evaluate_current import _format_table
from .file_policy import resolve_history_path, resolve_log_path, resolve_model_path
from .model_policy import resolve_ppo_model_path


DEFAULT_REPLAY_PATHS = {
    "train": Path("data/train.jsonl"),
    "valid": Path("data/valid.jsonl"),
    "test": Path("data/test.jsonl"),
}


def replay_model(
    *,
    model_path: str | Path,
    histories: dict[str, str | Path] | None = None,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
) -> dict[str, Any]:
    model = PPO.load(resolve_ppo_model_path(model_path))
    paths = histories or DEFAULT_REPLAY_PATHS
    report: dict[str, Any] = {
        "model": str(resolve_model_path(model_path)),
        "initial_balance": initial_balance,
        "deterministic": True,
        "sections": {},
    }

    for name, path in paths.items():
        history_path = resolve_history_path(path)
        history = load_history(history_path)

        def policy(obs, info):
            action, _ = model.predict(obs, deterministic=True)
            return [int(value) for value in action]

        metrics = run_backtest(history, policy=policy, initial_balance=initial_balance)
        metrics["source"] = str(history_path)
        report["sections"][name] = metrics

    return report


def save_replay_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output = resolve_log_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output


def format_replay_table(report: dict[str, Any]) -> str:
    rows = []
    for name, metrics in report.get("sections", {}).items():
        rows.append(
            [
                str(name),
                str(metrics["steps"]),
                str(metrics["bets"]),
                f"{float(metrics['bet_rate']) * 100:.2f}%",
                f"{float(metrics['win_rate']) * 100:.2f}%",
                f"{float(metrics['final_balance']):.2f}",
                f"{float(metrics['max_drawdown']) * 100:.2f}%",
                str(metrics["longest_losing_streak"]),
                str(metrics["bankruptcies"]),
            ]
        )
    lines = [
        f"模型：{report.get('model')}",
        "模式：model.predict(..., deterministic=True)，未调用 model.learn()",
        "",
        *_format_table(
            ["数据", "期数", "下注数", "下注率", "胜率", "最终余额", "最大回撤", "最长连亏", "爆仓"],
            rows,
            right_align={1, 2, 3, 4, 5, 6, 7, 8},
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an existing PPO model without training.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--initial-balance", type=float, default=DEFAULT_INITIAL_BALANCE)
    parser.add_argument("--output", default=str(LOG_DIR / "latest_replay.json"))
    args = parser.parse_args()

    report = replay_model(model_path=args.model, initial_balance=args.initial_balance)
    output = save_replay_report(report, args.output)
    print(format_replay_table(report))
    print(f"\n资金曲线：{output}")


if __name__ == "__main__":
    main()
