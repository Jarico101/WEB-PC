from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settlement:
    reward: float
    profit_loss: float
    next_losing_streak: int
    reason: str


def settle_action(
    *,
    direction: int,
    stake: float,
    outcome_is_straight: bool,
    pressure_score: float,
    losing_streak: int,
    amount_ratio: float,
    current_omission: float,
    omission_avg_ratio: float,
    straight_profit_multiplier: float,
    non_straight_profit_multiplier: float,
    non_straight_loss_multiplier: float = 20.0,
) -> Settlement:
    if direction == 0 or stake <= 0.0:
        return Settlement(
            reward=0.0,
            profit_loss=0.0,
            next_losing_streak=losing_streak,
            reason="watch",
        )

    won = (direction == 1 and outcome_is_straight) or (direction == 2 and not outcome_is_straight)
    if won:
        if direction == 1:
            profit_loss = stake * straight_profit_multiplier
        else:
            # 反向：未出顺，净赚 = 名义下注 * 盈利倍率（例：100 -> +100）
            profit_loss = stake * non_straight_profit_multiplier
        return Settlement(
            reward=profit_loss,
            profit_loss=profit_loss,
            next_losing_streak=0,
            reason="win",
        )

    next_losing_streak = losing_streak + 1
    if direction == 2:
        # 反向：出顺，扣掉冻结资金（例：下注 100 -> 净亏 2000）
        profit_loss = -stake * non_straight_loss_multiplier
    else:
        profit_loss = -stake

    return Settlement(
        reward=profit_loss,
        profit_loss=profit_loss,
        next_losing_streak=next_losing_streak,
        reason="loss",
    )


def apply_bankruptcy_penalty(reward: float, *, bankrupt: bool, penalty: float) -> float:
    return reward - penalty if bankrupt else reward


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))
