from __future__ import annotations

import argparse
import json
from typing import Any

from stable_baselines3 import PPO

from .casino_env import DEFAULT_MIN_STAKE, CasinoEnv, DIRECTION_LABELS
from .config import DEFAULT_INITIAL_BALANCE, DEFAULT_MAX_STAKE, MODEL_PATH
from .data_adapter import clean_history
from .model_policy import resolve_ppo_model_path


def action_to_decision(
    action: list[int] | tuple[int, ...],
    *,
    balance: float,
    max_stake: float = DEFAULT_MAX_STAKE,
    min_stake: float = DEFAULT_MIN_STAKE,
) -> dict[str, Any]:
    if len(action) == 2:
        intended_direction = int(action[0])
        should_bet = intended_direction != 0
        amount_bin = int(action[1])
    elif len(action) == 3:
        should_bet = bool(int(action[0]))
        intended_direction = int(action[1])
        amount_bin = int(action[2])
    else:
        raise ValueError("action must contain 2 or 3 integers")

    direction = intended_direction if should_bet else 0
    amount_ratio = 0.0 if direction == 0 else (amount_bin + 1) / 1000.0
    stake = (
        0.0
        if direction == 0 or float(balance) < float(min_stake)
        else min(float(balance), max(float(min_stake), float(balance) * amount_ratio), float(max_stake))
    )
    return {
        "should_bet": bool(should_bet),
        "intended_direction": intended_direction,
        "intended_direction_label": DIRECTION_LABELS.get(intended_direction, "unknown"),
        "direction": direction,
        "direction_label": DIRECTION_LABELS.get(direction, "unknown"),
        "amount_bin": amount_bin,
        "amount_ratio": round(amount_ratio, 6),
        "min_stake": round(float(min_stake), 6),
        "max_stake": round(float(max_stake), 6),
        "stake": round(stake, 6),
    }


def predict_decision(
    history_source,
    *,
    model_path: str = str(MODEL_PATH),
    balance: float = DEFAULT_INITIAL_BALANCE,
    deterministic: bool = True,
) -> dict[str, Any]:
    history = clean_history(history_source, initial_capital=balance, min_rows=1)
    env = CasinoEnv(history, initial_balance=balance)
    obs, _ = env.reset(options={"start_step": len(history) - 1})
    model = PPO.load(resolve_ppo_model_path(model_path))
    action, _ = model.predict(obs, deterministic=deterministic)
    decision = action_to_decision(list(action), balance=balance)
    decision["issue"] = history[-1].get("issue")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one PC28 PPO action.")
    parser.add_argument("history", help="CSV/JSON/JSONL history or API payload")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--balance", type=float, default=DEFAULT_INITIAL_BALANCE)
    args = parser.parse_args()
    print(json.dumps(predict_decision(args.history, model_path=args.model, balance=args.balance), ensure_ascii=False))


if __name__ == "__main__":
    main()
