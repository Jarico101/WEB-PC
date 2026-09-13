"""CLI for time-ordered, metadata-gated PC28 combo model evaluation."""

from __future__ import annotations

import argparse
import json

from .config import LOG_DIR
from .data_adapter import load_history
from .evaluation_combo import evaluate_combo_model
from .file_policy import resolve_log_path
from .markets_combo import CONFIRMED_EXTREME_RANGES, ZeroPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a PC28 combo model on post-training walk-forward windows.")
    parser.add_argument("history", help="CSV/JSON/JSONL evaluation history")
    parser.add_argument("--model", required=True)
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], required=True)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--zero-policy", choices=[item.value for item in ZeroPolicy], default=ZeroPolicy.EXCLUDE.value)
    parser.add_argument("--output", default=str(LOG_DIR / "combo_evaluation.json"))
    args = parser.parse_args()
    report = evaluate_combo_model(
        load_history(args.history),
        model_path=args.model,
        algorithm=args.algorithm,
        window_size=args.window_size,
        windows=args.windows,
        initial_balance=args.initial_balance,
        zero_policy=ZeroPolicy(args.zero_policy),
        extreme_ranges=CONFIRMED_EXTREME_RANGES,
    )
    output_path = resolve_log_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"saved evaluation: {output_path}")


if __name__ == "__main__":
    main()
