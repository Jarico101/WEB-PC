from __future__ import annotations

import argparse
import json
import random
from typing import Any, Callable

from .config import DEFAULT_INITIAL_BALANCE


Policy = Callable[[Any, dict[str, Any]], list[int] | tuple[int, ...]]

BASELINE_NAMES = (
    "watch_all",
    "always_non_straight_min",
    "always_non_straight_only_min",
    "always_straight_min",
    "random_min",
)


def run_backtest(
    history: list[dict[str, Any]],
    *,
    policy: Policy,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    use_risk_guard: bool = False,
) -> dict[str, Any]:
    from .casino_env import CasinoEnv
    from .risk_guard import RuntimeState, evaluate_risk

    env = CasinoEnv(history, initial_balance=initial_balance)
    obs, reset_info = env.reset()
    info = dict(reset_info)
    equity_curve = [initial_balance]
    rewards: list[float] = []
    bets = 0
    wins = 0
    losses = 0
    longest_losing_streak = 0
    guard_denies = 0
    bankruptcies = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = list(policy(obs, info))
        should_bet, direction, amount_bin = _normalize_action(action)
        action = [direction if should_bet else 0, amount_bin]
        if use_risk_guard:
            guard = evaluate_risk(
                direction=direction if should_bet else 0,
                amount_ratio=(amount_bin + 1) / 1000.0,
                state=RuntimeState(
                    issue=str(env.history[env.current_step].get("issue")),
                    balance=env.balance,
                    initial_balance=env.initial_balance,
                    losing_streak=env.losing_streak,
                    pressure_score=float(env.history[env.current_step]["features"].get("pressure_score", 0.0)),
                ),
            )
            if not guard.allowed:
                guard_denies += 1
                action = [0, amount_bin]
                should_bet = False
            elif guard.capped_amount_ratio < guard.amount_ratio:
                amount_bin = max(0, min(99, int(round(guard.capped_amount_ratio * 1000.0)) - 1))
                action = [direction if should_bet else 0, amount_bin]

        if should_bet and direction != 0:
            bets += 1
        obs, reward, terminated, truncated, info = env.step(action)
        if float(info.get("stake", 0.0)) > 0.0:
            profit_loss = float(info.get("profit_loss", 0.0))
            if profit_loss > 0.0:
                wins += 1
            elif profit_loss < 0.0:
                losses += 1
            longest_losing_streak = max(longest_losing_streak, int(info.get("losing_streak", 0)))
        if terminated and float(info.get("balance", 0.0)) <= 0.0:
            bankruptcies += 1
        rewards.append(float(reward))
        equity_curve.append(float(info["balance"]))

    return {
        "steps": len(rewards),
        "bets": bets,
        "bet_rate": round(bets / max(len(rewards), 1), 6),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / max(bets, 1), 6),
        "longest_losing_streak": longest_losing_streak,
        "bankruptcies": bankruptcies,
        "guard_denies": guard_denies,
        "final_balance": round(equity_curve[-1], 6),
        "net_profit": round(equity_curve[-1] - initial_balance, 6),
        "max_drawdown": round(_max_drawdown(equity_curve), 6),
        "total_reward": round(sum(rewards), 6),
        "equity_curve": [round(value, 6) for value in equity_curve],
    }


def baseline_policy(name: str, *, seed: int = 42) -> Policy:
    from .casino_env import DIRECTION_LABELS

    if name == "watch_all":
        return lambda obs, info: [0, 0]
    if name == "always_non_straight_min":
        return lambda obs, info: [2, 0]
    if name == "always_non_straight_only_min":
        direction = next(key for key, label in DIRECTION_LABELS.items() if label == "non_straight_only")
        return lambda obs, info: [direction, 0]
    if name == "always_straight_min":
        return lambda obs, info: [1, 0]
    if name == "random_min":
        rng = random.Random(seed)
        return lambda obs, info: [rng.choice([0, 1, 2]), 0]
    raise ValueError(f"unknown baseline: {name}")


def _normalize_action(action: list[int] | tuple[int, ...]) -> tuple[bool, int, int]:
    if len(action) == 2:
        direction = int(action[0])
        amount_bin = int(action[1])
        return direction != 0, direction, amount_bin
    if len(action) == 3:
        return bool(int(action[0])), int(action[1]), int(action[2])
    raise ValueError("action must contain 2 or 3 integers")


def run_baselines(
    history: list[dict[str, Any]],
    *,
    initial_balance: float = DEFAULT_INITIAL_BALANCE,
    use_risk_guard: bool = False,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    return {
        name: run_backtest(
            history,
            policy=baseline_policy(name, seed=seed),
            initial_balance=initial_balance,
            use_risk_guard=use_risk_guard,
        )
        for name in BASELINE_NAMES
    }


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)
    return max_dd


def main() -> None:
    from .data_adapter import load_history

    parser = argparse.ArgumentParser(description="Run a PC28 PPO backtest smoke policy.")
    parser.add_argument("history", help="CSV/JSON/JSONL history file")
    parser.add_argument("--initial-balance", type=float, default=DEFAULT_INITIAL_BALANCE)
    parser.add_argument("--with-risk-guard", action="store_true")
    parser.add_argument("--baselines", action="store_true", help="Run watch/reverse/straight/random baseline policies.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    history = load_history(args.history)
    if args.baselines:
        result = run_baselines(
            history,
            initial_balance=args.initial_balance,
            use_risk_guard=args.with_risk_guard,
            seed=args.seed,
        )
    else:
        result = run_backtest(
            history,
            policy=baseline_policy("always_non_straight_min", seed=args.seed),
            initial_balance=args.initial_balance,
            use_risk_guard=args.with_risk_guard,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
