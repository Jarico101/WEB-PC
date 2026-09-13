from __future__ import annotations

from dataclasses import dataclass

from .schemas_combo import ComboSettlement


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


@dataclass(frozen=True)
class ComboRewardConfig:
    """Reward shaping for the virtual PC28 combo environment only."""

    drawdown_penalty: float = 0.25
    losing_streak_penalty: float = 0.005
    calibration_penalty: float = 0.05
    reward_scale: float = 100.0
    shadow_reward_weight: float = 0.15
    repeat_base_combo_penalty: float = 0.0
    # Experimental PPO can be taught to map prior observable state to a
    # different legal combo.  This is a behaviour objective, not a claim that
    # those statistics predict the next outcome.
    state_combo_alignment_bonus: float = 0.0

    def __post_init__(self) -> None:
        if self.repeat_base_combo_penalty < 0 or self.state_combo_alignment_bonus < 0:
            raise ValueError("combo reward penalties and bonuses must be non-negative")


@dataclass(frozen=True)
class ComboReward:
    reward: float
    realized_pnl_component: float
    shadow_pnl_component: float
    drawdown_penalty: float
    losing_streak_penalty: float
    calibration_penalty: float
    calibration_error: float


def calculate_combo_reward(
    settlement: ComboSettlement,
    *,
    initial_balance: float,
    drawdown: float,
    losing_streak: int,
    config: ComboRewardConfig = ComboRewardConfig(),
) -> ComboReward:
    """Score realized or shadow combo decisions without changing settlement state.

    The payout is normalized by initial virtual capital.  In shadow mode the
    same counterfactual PnL is retained with reduced weight: it can inform the
    observation decision while the settlement layer still leaves balance intact.
    """

    if initial_balance <= 0:
        raise ValueError("initial_balance must be positive")
    calibration_error = _combo_calibration_error(settlement)
    pnl_component = settlement.net_profit / initial_balance * config.reward_scale
    realized_pnl_component = 0.0 if settlement.is_shadow else pnl_component
    shadow_pnl_component = pnl_component * config.shadow_reward_weight if settlement.is_shadow else 0.0
    drawdown_cost = max(0.0, drawdown) * config.drawdown_penalty
    streak_cost = max(0, losing_streak) * config.losing_streak_penalty
    calibration_cost = calibration_error * config.calibration_penalty
    reward = realized_pnl_component + shadow_pnl_component - drawdown_cost - streak_cost - calibration_cost
    return ComboReward(
        reward=reward,
        realized_pnl_component=realized_pnl_component,
        shadow_pnl_component=shadow_pnl_component,
        drawdown_penalty=drawdown_cost,
        losing_streak_penalty=streak_cost,
        calibration_penalty=calibration_cost,
        calibration_error=calibration_error,
    )


def _combo_calibration_error(settlement: ComboSettlement) -> float:
    if not settlement.items:
        return 0.0
    squared_errors = [
        (max(0.0, min(1.0, item.selection.probability)) - float(item.hit)) ** 2
        for item in settlement.items
    ]
    return sum(squared_errors) / len(squared_errors)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, float(value)))
