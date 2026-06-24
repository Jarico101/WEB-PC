from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data_adapter import FEATURE_COLUMNS, clean_history
from .config import DEFAULT_INITIAL_BALANCE, DEFAULT_MAX_STAKE
from .reward import Settlement, apply_bankruptcy_penalty, settle_action


DEFAULT_MIN_STAKE = 10.0
DEFAULT_MAX_PAYOUT = 1_000_000.0
NON_STRAIGHT_ONLY_PROFIT_MULTIPLIER = 1.0
NON_STRAIGHT_ONLY_LOSS_MULTIPLIER = 17.0


@dataclass(frozen=True)
class BetSpec:
    label: str
    kind: str
    target: Any
    gross_odds: float
    max_payout: float = DEFAULT_MAX_PAYOUT


SUM_EXACT_GROSS_ODDS = {
    0: 888.0,
    1: 300.0,
    2: 150.0,
    3: 90.0,
    4: 60.0,
    5: 45.0,
    6: 33.0,
    7: 27.0,
    8: 22.0,
    9: 18.0,
    10: 15.0,
    11: 14.0,
    12: 13.0,
    13: 11.0,
    14: 11.0,
    15: 13.0,
    16: 14.0,
    17: 15.0,
    18: 18.0,
    19: 22.0,
    20: 27.0,
    21: 33.0,
    22: 45.0,
    23: 60.0,
    24: 90.0,
    25: 150.0,
    26: 300.0,
    27: 888.0,
}


def _build_bet_specs() -> dict[int, BetSpec]:
    specs: dict[int, BetSpec] = {
        0: BetSpec("watch", "watch", None, 0.0),
        1: BetSpec("straight", "legacy_straight", None, 18.0),
        2: BetSpec("non_straight", "legacy_non_straight", None, 2.0),
    }
    next_id = 3
    for total in range(28):
        specs[next_id] = BetSpec(f"sum_{total}", "sum_exact", total, SUM_EXACT_GROSS_ODDS[total], 20_000.0)
        next_id += 1
    for target in ("big", "small", "odd", "even"):
        specs[next_id] = BetSpec(target, "sum_simple", target, 1.98, 100_000.0)
        next_id += 1
    for target, odds in (
        ("small_odd", 3.69),
        ("big_even", 3.69),
        ("big_odd", 4.29),
        ("small_even", 4.29),
    ):
        specs[next_id] = BetSpec(target, "sum_combo", target, odds, 50_000.0)
        next_id += 1
    for label, odds, cap in (
        ("pair", 3.5, 20_000.0),
        ("leopard", 60.0, 20_000.0),
        ("extreme_small", 16.0, DEFAULT_MAX_PAYOUT),
        ("extreme_big", 16.0, DEFAULT_MAX_PAYOUT),
    ):
        specs[next_id] = BetSpec(label, "pattern", label, odds, cap)
        next_id += 1
    for position, prefix in enumerate(("A", "B", "C")):
        for target in ("big", "small", "odd", "even"):
            specs[next_id] = BetSpec(f"{prefix}_{target}", "digit_simple", (position, target), 1.9, 100_000.0)
            next_id += 1
        for target, odds in (
            ("small_odd", 4.7),
            ("big_even", 4.7),
            ("small_even", 3.1),
            ("big_odd", 3.1),
        ):
            specs[next_id] = BetSpec(f"{prefix}_{target}", "digit_combo", (position, target), odds, 50_000.0)
            next_id += 1
        for digit in range(10):
            specs[next_id] = BetSpec(f"{prefix}_{digit}", "digit_exact", (position, digit), 9.5, 20_000.0)
            next_id += 1
    specs[next_id] = BetSpec("non_straight_only", "non_straight_only", None, 2.0)
    return specs


BET_SPECS = _build_bet_specs()
DIRECTION_LABELS = {direction: spec.label for direction, spec in BET_SPECS.items()}
ACTION_SPACE_SIZE = len(BET_SPECS)


SUM_OUTCOME_COUNTS = {
    total: sum(1 for a in range(10) for b in range(10) for c in range(10) if a + b + c == total)
    for total in range(28)
}


def _sum_probability(total: int) -> float:
    return SUM_OUTCOME_COUNTS.get(int(total), 0) / 1000.0


@dataclass(frozen=True)
class TrainingRiskConfig:
    straight_min_rate: float = 0.0605
    non_straight_max_rate: float = 0.047
    max_drawdown: float = 0.35
    cooldown_steps_after_loss: int = 3
    bet_cost: float = 0.05
    gray_zone_penalty_multiplier: float = 0.01
    threshold_penalty_multiplier: float = 0.08
    cooldown_penalty_multiplier: float = 0.1
    drawdown_penalty_multiplier: float = 1.0
    non_straight_exposure_limit: float = 0.2
    non_straight_exposure_penalty_multiplier: float = 0.5
    negative_edge_penalty_multiplier: float = 0.25
    high_stake_ratio: float = 0.05
    high_stake_penalty_multiplier: float = 0.15
    frequent_bet_penalty_start: int = 3
    frequent_bet_penalty_multiplier: float = 0.05
    max_bet_rate: float = 0.35
    bet_rate_penalty_min_steps: int = 20
    bet_rate_penalty_multiplier: float = 3.0
    min_bet_rate: float = 0.02
    low_bet_rate_penalty_min_steps: int = 20
    low_bet_rate_penalty_multiplier: float = 0.001
    watch_penalty_start: int = 5
    excessive_watch_penalty: float = 0.25
    enabled: bool = True


TrainingGateConfig = TrainingRiskConfig


@dataclass(frozen=True)
class GameObjectiveConfig:
    enabled: bool = False
    max_steps: int = 200
    pass_balance_multiplier: float = 1.3
    fail_balance_multiplier: float = 0.7
    target_bet_rate: float = 0.2
    pass_reward: float = 300.0
    fail_penalty: float = 300.0
    bet_limit_penalty: float = 300.0
    completion_reward_multiplier: float = 1.0
    bet_rate_bonus: float = 50.0

    @property
    def max_bets(self) -> int:
        return max(0, int(self.max_steps * self.target_bet_rate))


class CasinoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        history: list[dict[str, Any]],
        *,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        max_stake: float = DEFAULT_MAX_STAKE,
        min_stake: float = DEFAULT_MIN_STAKE,
        max_payout: float = DEFAULT_MAX_PAYOUT,
        straight_profit_multiplier: float = 17.0,
        non_straight_profit_multiplier: float = 1.0,
        non_straight_loss_multiplier: float = 20.0,
        bankruptcy_penalty: float = 100000.0,
        reward_mode: str = "pnl",
        training_gate: TrainingRiskConfig | None = None,
        game_objective: GameObjectiveConfig | None = None,
    ) -> None:
        super().__init__()
        if reward_mode not in {"pnl", "normalized"}:
            raise ValueError("reward_mode must be one of: pnl, normalized")
        self.history = clean_history(history, initial_capital=initial_balance, min_rows=1)
        self.initial_balance = float(initial_balance)
        self.max_stake = float(max_stake)
        self.min_stake = float(min_stake)
        self.max_payout = float(max_payout)
        self.straight_profit_multiplier = float(straight_profit_multiplier)
        self.non_straight_profit_multiplier = float(non_straight_profit_multiplier)
        self.non_straight_loss_multiplier = float(non_straight_loss_multiplier)
        self.bankruptcy_penalty = float(bankruptcy_penalty)
        self.reward_mode = reward_mode
        self.training_risk = training_gate or TrainingRiskConfig()
        self.training_gate = self.training_risk
        self.game_objective = game_objective or GameObjectiveConfig()

        self.action_space = spaces.MultiDiscrete([ACTION_SPACE_SIZE, 100])
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, -100.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([300.0, 100.0, 1.0, 1.0, 100.0, 100.0, 1.0, 100.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.current_step = 0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.losing_streak = 0
        self.cooldown_remaining = 0
        self.watch_streak = 0
        self.bet_streak = 0
        self.episode_bets = 0
        self.episode_steps = 0
        self.last_info: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.current_step = int(options.get("start_step", 0)) if options else 0
        self.current_step = max(0, min(self.current_step, len(self.history) - 1))
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.losing_streak = 0
        self.cooldown_remaining = 0
        self.watch_streak = 0
        self.bet_streak = 0
        self.episode_bets = 0
        self.episode_steps = 0
        self.last_info = {"balance": self.balance, "step": self.current_step}
        return self._get_obs(), dict(self.last_info)

    def step(self, action):
        should_bet, intended_direction, amount_bin = self._parse_action(action)
        if isinstance(intended_direction, tuple):
            directions = intended_direction if should_bet else ()
            return self._step_package(directions, amount_bin)
        row = self.history[self.current_step]
        direction = intended_direction if should_bet else 0
        label = DIRECTION_LABELS[direction]
        amount_ratio = self._amount_ratio(amount_bin)
        stake = self._stake_for(direction, amount_ratio)
        selected_odds = self._selected_odds(direction, row, stake)
        bet_cost = self.training_risk.bet_cost if direction != 0 and stake > 0.0 else 0.0
        risk_penalty, risk_reasons = self._risk_penalty(direction, stake, row)
        watch_penalty, watch_reasons = self._watch_penalty(direction)
        settlement = self._settle_bet(direction, stake, row, amount_ratio)
        profit_loss = settlement.profit_loss
        self.balance += profit_loss
        self.peak_balance = max(self.peak_balance, self.balance)
        self.losing_streak = settlement.next_losing_streak
        raw_reward = settlement.reward - bet_cost - risk_penalty - watch_penalty
        reward = self._training_reward(
            settlement_reward=settlement.reward,
            bet_cost=bet_cost,
            risk_penalty=risk_penalty,
            watch_penalty=watch_penalty,
        )
        reward_reason = settlement.reason
        if bet_cost:
            reward_reason = f"{reward_reason}_bet_cost"
        if risk_penalty:
            reward_reason = f"{reward_reason}_risk_penalty"
        if watch_penalty:
            reward_reason = f"{reward_reason}_watch_penalty"

        terminated = self.balance <= 0.0
        if terminated:
            self.balance = max(0.0, self.balance)
            raw_reward = apply_bankruptcy_penalty(raw_reward, bankrupt=True, penalty=self.bankruptcy_penalty)
            reward = self._bankruptcy_reward(reward)
            reward_reason = f"{reward_reason}_bankrupt"

        cooldown_before_decrement = self.cooldown_remaining
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        if direction != 0 and profit_loss < 0.0 and self.training_risk.cooldown_steps_after_loss > 0:
            self.cooldown_remaining = max(self.cooldown_remaining, self.training_risk.cooldown_steps_after_loss)
        if direction == 0:
            self.watch_streak += 1
            self.bet_streak = 0
        else:
            self.watch_streak = 0
            self.bet_streak += 1
            if stake > 0.0:
                self.episode_bets += 1

        self.episode_steps += 1
        self.current_step += 1
        bet_rate = self.episode_bets / max(self.episode_steps, 1)
        truncated = self.current_step >= len(self.history)
        objective_reward, objective_reason, objective_done = self._game_objective_reward(
            terminated=terminated,
            truncated=truncated,
            bet_rate=bet_rate,
        )
        if objective_reward:
            raw_reward += objective_reward
            reward += objective_reward
            reward_reason = f"{reward_reason}_{objective_reason}"
        if objective_done == "pass":
            terminated = True
            truncated = False
        elif objective_done in {"fail", "max_steps"}:
            truncated = True

        obs = np.zeros(8, dtype=np.float32) if (terminated or truncated) else self._get_obs()
        self.last_info = {
            "issue": row.get("issue"),
            "step": self.current_step - 1,
            "intended_direction": DIRECTION_LABELS[intended_direction],
            "should_bet": bool(should_bet),
            "direction": label,
            "bet_kind": selected_odds["kind"],
            "selected_gross_odds": selected_odds["gross_odds"],
            "selected_net_profit_multiplier": selected_odds["net_profit_multiplier"],
            "selected_loss_multiplier": selected_odds["loss_multiplier"],
            "amount_ratio": round(amount_ratio, 6),
            "min_stake": round(self.min_stake, 6),
            "max_stake": round(self.max_stake, 6),
            "max_payout": round(self.max_payout, 6),
            "stake": round(stake, 6),
            "profit_loss": round(profit_loss, 6),
            "bet_cost": round(bet_cost, 6),
            "risk_penalty": round(risk_penalty, 6),
            "watch_penalty": round(watch_penalty, 6),
            "raw_reward": round(raw_reward, 6),
            "reward_mode": self.reward_mode,
            "risk_reasons": tuple((*risk_reasons, *watch_reasons) or ("none",)),
            "balance": round(self.balance, 6),
            "drawdown": round(self._current_drawdown(), 6),
            "outcome_is_straight": bool(row["outcome_is_straight"]),
            "pressure_score": self._pressure_score(row),
            "straight_rate": round(self._straight_rate(row), 12),
            "losing_streak": self.losing_streak,
            "watch_streak": self.watch_streak,
            "bet_streak": self.bet_streak,
            "episode_bets": self.episode_bets,
            "episode_steps": self.episode_steps,
            "bet_rate": round(bet_rate, 6),
            "game_objective_enabled": self.game_objective.enabled,
            "game_objective_reason": objective_reason,
            "game_objective_reward": round(objective_reward, 6),
            "watch_penalty_start": self.training_risk.watch_penalty_start,
            "cooldown_remaining": self.cooldown_remaining,
            "cooldown_remaining_before": cooldown_before_decrement,
            "gate_reasons": tuple((*risk_reasons, *watch_reasons) or ("allowed",)),
            "action_forced_to_watch": False,
            "action_forced_to_bet": False,
            "reward_reason": reward_reason,
        }
        return obs, float(reward), bool(terminated), bool(truncated), dict(self.last_info)

    def step_package(self, directions: tuple[int, ...], amount_bin: int = 0, *, stake_per_play: float | None = None):
        return self._step_package(
            tuple(int(direction) for direction in directions),
            int(amount_bin),
            stake_per_play=stake_per_play,
        )

    def _step_package(self, directions: tuple[int, ...], amount_bin: int, *, stake_per_play: float | None = None):
        directions = tuple(direction for direction in directions if direction != 0)
        if not directions:
            return self.step([0, amount_bin])

        row = self.history[self.current_step]
        amount_ratio = self._amount_ratio(amount_bin)
        if stake_per_play is None:
            stake_per_play = self._stake_for(directions[0], amount_ratio)
        else:
            stake_per_play = min(max(self.min_stake, float(stake_per_play)), self.max_stake)
        if stake_per_play * len(directions) > self.balance:
            stake_per_play = 0.0
        settlements = [
            self._settle_bet(direction, stake_per_play, row, amount_ratio)
            for direction in directions
        ]
        profit_loss = sum(settlement.profit_loss for settlement in settlements)
        settlement_reward = sum(settlement.reward for settlement in settlements)
        stake = stake_per_play * len(directions)
        bet_cost = self.training_risk.bet_cost * len(directions) if stake > 0.0 else 0.0
        risk_penalties = [self._risk_penalty(direction, stake_per_play, row) for direction in directions]
        risk_penalty = sum(penalty for penalty, _reasons in risk_penalties)
        risk_reasons = tuple(reason for _penalty, reasons in risk_penalties for reason in reasons)
        watch_penalty, watch_reasons = self._watch_penalty(0 if stake <= 0.0 else directions[0])

        self.balance += profit_loss
        self.peak_balance = max(self.peak_balance, self.balance)
        self.losing_streak = self.losing_streak + 1 if profit_loss < 0.0 else 0
        raw_reward = settlement_reward - bet_cost - risk_penalty - watch_penalty
        reward = self._training_reward(
            settlement_reward=settlement_reward,
            bet_cost=bet_cost,
            risk_penalty=risk_penalty,
            watch_penalty=watch_penalty,
        )
        reward_reason = "win" if profit_loss > 0.0 else "loss" if profit_loss < 0.0 else "push"
        if bet_cost:
            reward_reason = f"{reward_reason}_bet_cost"
        if risk_penalty:
            reward_reason = f"{reward_reason}_risk_penalty"
        if watch_penalty:
            reward_reason = f"{reward_reason}_watch_penalty"

        terminated = self.balance <= 0.0
        if terminated:
            self.balance = max(0.0, self.balance)
            raw_reward = apply_bankruptcy_penalty(raw_reward, bankrupt=True, penalty=self.bankruptcy_penalty)
            reward = self._bankruptcy_reward(reward)
            reward_reason = f"{reward_reason}_bankrupt"

        cooldown_before_decrement = self.cooldown_remaining
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        if profit_loss < 0.0 and self.training_risk.cooldown_steps_after_loss > 0:
            self.cooldown_remaining = max(self.cooldown_remaining, self.training_risk.cooldown_steps_after_loss)

        self.watch_streak = 0
        self.bet_streak += 1
        if stake > 0.0:
            self.episode_bets += 1

        self.episode_steps += 1
        self.current_step += 1
        bet_rate = self.episode_bets / max(self.episode_steps, 1)
        truncated = self.current_step >= len(self.history)
        objective_reward, objective_reason, objective_done = self._game_objective_reward(
            terminated=terminated,
            truncated=truncated,
            bet_rate=bet_rate,
        )
        if objective_reward:
            raw_reward += objective_reward
            reward += objective_reward
            reward_reason = f"{reward_reason}_{objective_reason}"
        if objective_done == "pass":
            terminated = True
            truncated = False
        elif objective_done in {"fail", "max_steps"}:
            truncated = True

        obs = np.zeros(8, dtype=np.float32) if (terminated or truncated) else self._get_obs()
        labels = tuple(DIRECTION_LABELS[direction] for direction in directions)
        self.last_info = {
            "issue": row.get("issue"),
            "step": self.current_step - 1,
            "intended_direction": "+".join(labels),
            "should_bet": bool(stake > 0.0),
            "direction": "+".join(labels),
            "bet_kind": "package",
            "selected_gross_odds": None,
            "selected_net_profit_multiplier": None,
            "selected_loss_multiplier": None,
            "amount_ratio": round(amount_ratio, 6),
            "min_stake": round(self.min_stake, 6),
            "max_stake": round(self.max_stake, 6),
            "max_payout": round(self.max_payout, 6),
            "stake": round(stake, 6),
            "stake_per_play": round(stake_per_play, 6),
            "package_size": len(directions),
            "package_directions": labels,
            "package_profit_losses": tuple(round(settlement.profit_loss, 6) for settlement in settlements),
            "profit_loss": round(profit_loss, 6),
            "bet_cost": round(bet_cost, 6),
            "risk_penalty": round(risk_penalty, 6),
            "watch_penalty": round(watch_penalty, 6),
            "raw_reward": round(raw_reward, 6),
            "reward_mode": self.reward_mode,
            "risk_reasons": tuple((*risk_reasons, *watch_reasons) or ("none",)),
            "balance": round(self.balance, 6),
            "drawdown": round(self._current_drawdown(), 6),
            "outcome_is_straight": bool(row["outcome_is_straight"]),
            "pressure_score": self._pressure_score(row),
            "straight_rate": round(self._straight_rate(row), 12),
            "losing_streak": self.losing_streak,
            "watch_streak": self.watch_streak,
            "bet_streak": self.bet_streak,
            "episode_bets": self.episode_bets,
            "episode_steps": self.episode_steps,
            "bet_rate": round(bet_rate, 6),
            "game_objective_enabled": self.game_objective.enabled,
            "game_objective_reason": objective_reason,
            "game_objective_reward": round(objective_reward, 6),
            "watch_penalty_start": self.training_risk.watch_penalty_start,
            "cooldown_remaining": self.cooldown_remaining,
            "cooldown_remaining_before": cooldown_before_decrement,
            "gate_reasons": tuple((*risk_reasons, *watch_reasons) or ("allowed",)),
            "action_forced_to_watch": False,
            "action_forced_to_bet": False,
            "reward_reason": reward_reason,
        }
        return obs, float(reward), bool(terminated), bool(truncated), dict(self.last_info)

    def _game_objective_reward(
        self,
        *,
        terminated: bool,
        truncated: bool,
        bet_rate: float,
    ) -> tuple[float, str, str | None]:
        config = self.game_objective
        if not config.enabled:
            return 0.0, "none", None

        pass_balance = self.initial_balance * config.pass_balance_multiplier
        fail_balance = self.initial_balance * config.fail_balance_multiplier
        if self.balance >= pass_balance:
            return config.pass_reward + self._bet_rate_bonus(bet_rate), "game_pass", "pass"
        if self.balance <= fail_balance:
            return -config.fail_penalty, "game_fail", "fail"
        if config.max_bets > 0 and self.episode_bets > config.max_bets:
            return -config.bet_limit_penalty, "game_bet_limit", "fail"
        if self.episode_steps >= config.max_steps or terminated or truncated:
            progress_reward = (self.balance - self.initial_balance) * config.completion_reward_multiplier
            return progress_reward + self._bet_rate_bonus(bet_rate), "game_complete", "max_steps"
        return 0.0, "none", None

    def _bet_rate_bonus(self, bet_rate: float) -> float:
        config = self.game_objective
        if not config.enabled or config.target_bet_rate <= 0.0:
            return 0.0
        miss_ratio = abs(bet_rate - config.target_bet_rate) / max(config.target_bet_rate, 1e-9)
        return config.bet_rate_bonus * max(0.0, 1.0 - miss_ratio)

    def _training_reward(
        self,
        *,
        settlement_reward: float,
        bet_cost: float,
        risk_penalty: float,
        watch_penalty: float,
    ) -> float:
        if self.reward_mode == "pnl":
            return settlement_reward - bet_cost - risk_penalty - watch_penalty
        pnl_component = (settlement_reward - bet_cost - risk_penalty) / max(self.initial_balance, 1e-9) * 100.0
        return pnl_component - watch_penalty

    def _bankruptcy_reward(self, reward: float) -> float:
        if self.reward_mode == "pnl":
            return apply_bankruptcy_penalty(reward, bankrupt=True, penalty=self.bankruptcy_penalty)
        return reward - 100.0

    def render(self):
        print(self.last_info)

    def _get_obs(self) -> np.ndarray:
        row = self.history[self.current_step]
        values = [float(row["features"][name]) for name in FEATURE_COLUMNS]
        values[-1] = self.balance / max(self.initial_balance, 1e-9)
        return np.clip(np.array(values, dtype=np.float32), self.observation_space.low, self.observation_space.high)

    def _parse_action(self, action) -> tuple[bool, int | tuple[int, ...], int]:
        if isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], tuple):
            directions = tuple(int(np.clip(direction, 0, ACTION_SPACE_SIZE - 1)) for direction in action[0])
            amount_bin = int(np.clip(action[1], 0, 99))
            return bool(directions), directions, amount_bin
        action_array = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_array.size == 2:
            direction = int(np.clip(action_array[0], 0, ACTION_SPACE_SIZE - 1))
            amount_bin = int(np.clip(action_array[1], 0, 99))
            return direction != 0, direction, amount_bin
        if action_array.size != 3:
            raise ValueError("action must contain [play, amount_bin]")
        # Legacy compatibility for old saved models that used [should_bet, play, amount_bin].
        should_bet = bool(int(np.clip(action_array[0], 0, 1)))
        direction = int(np.clip(action_array[1], 0, ACTION_SPACE_SIZE - 1))
        amount_bin = int(np.clip(action_array[2], 0, 99))
        return should_bet, direction, amount_bin

    @staticmethod
    def _amount_ratio(amount_bin: int) -> float:
        return (amount_bin + 1) / 1000.0

    def _stake_for(self, direction: int, amount_ratio: float) -> float:
        if direction == 0 or self.balance < self.min_stake:
            return 0.0
        raw_stake = self.balance * amount_ratio
        return min(self.balance, max(self.min_stake, raw_stake), self.max_stake)

    def _settle_bet(self, direction: int, stake: float, row: dict[str, Any], amount_ratio: float) -> Settlement:
        if direction in (0, 1, 2):
            return settle_action(
                direction=direction,
                stake=stake,
                outcome_is_straight=bool(row["outcome_is_straight"]),
                pressure_score=self._pressure_score(row),
                losing_streak=self.losing_streak,
                amount_ratio=amount_ratio,
                current_omission=float(row["features"].get("current_omission", 0.0)),
                omission_avg_ratio=float(row["features"].get("omission_avg_ratio", 0.0)),
                straight_profit_multiplier=self._profit_multiplier(row, 1),
                non_straight_profit_multiplier=self._profit_multiplier(row, 2),
                non_straight_loss_multiplier=self._loss_multiplier(row, 2),
            )

        if stake <= 0.0:
            return Settlement(0.0, 0.0, self.losing_streak, "watch")

        spec = BET_SPECS[direction]
        if spec.kind == "non_straight_only":
            if bool(row["outcome_is_straight"]):
                profit_loss = -stake * NON_STRAIGHT_ONLY_LOSS_MULTIPLIER
                return Settlement(profit_loss, profit_loss, self.losing_streak + 1, "loss")
            profit_loss = stake * NON_STRAIGHT_ONLY_PROFIT_MULTIPLIER
            return Settlement(profit_loss, profit_loss, 0, "win")

        matched, push, gross_odds = self._match_bet(direction, row, stake)
        if push:
            return Settlement(0.0, 0.0, self.losing_streak, "push")
        if matched:
            gross_return = min(stake * gross_odds, self._max_payout_for(direction))
            profit_loss = gross_return - stake
            return Settlement(profit_loss, profit_loss, 0, "win")

        profit_loss = -stake
        return Settlement(profit_loss, profit_loss, self.losing_streak + 1, "loss")

    def _selected_odds(self, direction: int, row: dict[str, Any], stake: float) -> dict[str, float | str | None]:
        spec = BET_SPECS[direction]
        if direction == 0:
            return {
                "kind": spec.kind,
                "gross_odds": None,
                "net_profit_multiplier": None,
                "loss_multiplier": None,
            }
        if direction == 1:
            net = self._profit_multiplier(row, 1)
            return {
                "kind": spec.kind,
                "gross_odds": round(net + 1.0, 6),
                "net_profit_multiplier": round(net, 6),
                "loss_multiplier": 1.0,
            }
        if direction == 2:
            net = self._profit_multiplier(row, 2)
            return {
                "kind": spec.kind,
                "gross_odds": round(net + 1.0, 6),
                "net_profit_multiplier": round(net, 6),
                "loss_multiplier": round(self._loss_multiplier(row, 2), 6),
            }
        if spec.kind == "non_straight_only":
            return {
                "kind": spec.kind,
                "gross_odds": spec.gross_odds,
                "net_profit_multiplier": NON_STRAIGHT_ONLY_PROFIT_MULTIPLIER,
                "loss_multiplier": NON_STRAIGHT_ONLY_LOSS_MULTIPLIER,
            }

        matched, push, gross_odds = self._match_bet(direction, row, stake)
        return {
            "kind": spec.kind,
            "gross_odds": round(gross_odds, 6),
            "net_profit_multiplier": 0.0 if push else round(gross_odds - 1.0, 6),
            "loss_multiplier": 1.0,
        }

    def _max_payout_for(self, direction: int) -> float:
        if direction in (0, 1, 2):
            return self.max_payout
        return min(self.max_payout, BET_SPECS[direction].max_payout)

    def _match_bet(self, direction: int, row: dict[str, Any], stake: float) -> tuple[bool, bool, float]:
        spec = BET_SPECS[direction]
        digits = self._digits(row)
        total = self._sum(row, digits)

        if spec.kind == "sum_exact":
            return total == int(spec.target), False, spec.gross_odds
        if spec.kind == "sum_simple":
            matched = self._category_matches(total, str(spec.target))
            return matched, False, spec.gross_odds
        if spec.kind == "sum_combo":
            target = str(spec.target)
            matched = self._category_matches(total, target)
            return matched, False, spec.gross_odds
        if spec.kind == "pattern":
            target = str(spec.target)
            if target == "pair":
                return self._is_pair(digits), False, spec.gross_odds
            if target == "leopard":
                return len(digits) == 3 and len(set(digits)) == 1, False, spec.gross_odds
            if target == "extreme_small":
                return 0 <= total <= 5, False, spec.gross_odds
            if target == "extreme_big":
                return 22 <= total <= 27, False, spec.gross_odds
        if spec.kind == "digit_simple":
            position, target = spec.target
            return len(digits) > position and self._digit_category_matches(digits[position], target), False, spec.gross_odds
        if spec.kind == "digit_combo":
            position, target = spec.target
            return len(digits) > position and self._digit_category_matches(digits[position], target), False, spec.gross_odds
        if spec.kind == "digit_exact":
            position, digit = spec.target
            return len(digits) > position and digits[position] == digit, False, spec.gross_odds

        return False, False, spec.gross_odds

    @staticmethod
    def _digits(row: dict[str, Any]) -> list[int]:
        digits = row.get("digits")
        if not digits and isinstance(row.get("raw"), dict):
            digits = row["raw"].get("digits")
        return [int(value) for value in digits] if isinstance(digits, list) else []

    @staticmethod
    def _sum(row: dict[str, Any], digits: list[int]) -> int:
        value = row.get("sum")
        if value is None and isinstance(row.get("raw"), dict):
            value = row["raw"].get("sum")
        if value is None:
            value = sum(digits)
        return int(value)

    @staticmethod
    def _category_matches(value: int, target: str) -> bool:
        return (
            (target == "big" and value >= 14)
            or (target == "small" and value <= 13)
            or (target == "odd" and value % 2 == 1)
            or (target == "even" and value % 2 == 0)
            or (target == "small_odd" and value <= 13 and value % 2 == 1)
            or (target == "small_even" and value <= 13 and value % 2 == 0)
            or (target == "big_odd" and value >= 14 and value % 2 == 1)
            or (target == "big_even" and value >= 14 and value % 2 == 0)
        )

    @staticmethod
    def _digit_category_matches(value: int, target: str) -> bool:
        return (
            (target == "big" and value >= 5)
            or (target == "small" and value <= 4)
            or (target == "odd" and value % 2 == 1)
            or (target == "even" and value % 2 == 0)
            or (target == "small_odd" and value <= 4 and value % 2 == 1)
            or (target == "small_even" and value <= 4 and value % 2 == 0)
            or (target == "big_odd" and value >= 5 and value % 2 == 1)
            or (target == "big_even" and value >= 5 and value % 2 == 0)
        )

    @staticmethod
    def _is_pair(digits: list[int]) -> bool:
        return len(digits) == 3 and len(set(digits)) == 2

    def _profit_multiplier(self, row: dict[str, Any], direction: int) -> float:
        if direction == 1:
            return self._row_multiplier(row, "straight", self.straight_profit_multiplier)
        direct = row.get("non_straight_profit_multiplier")
        if direct is not None:
            return float(direct)
        return self.non_straight_profit_multiplier

    def _loss_multiplier(self, row: dict[str, Any], direction: int) -> float:
        if direction != 2:
            return 1.0
        direct = row.get("non_straight_loss_multiplier")
        if direct is not None:
            return float(direct)
        odds = row.get("non_straight_odds")
        if odds is not None:
            return float(odds)
        return self.non_straight_loss_multiplier

    def _risk_penalty(self, direction: int, stake: float, row: dict[str, Any]) -> tuple[float, tuple[str, ...]]:
        config = self.training_risk
        if not config.enabled or direction == 0:
            return 0.0, ()

        reasons: list[str] = []
        penalty = 0.0
        straight_rate = self._straight_rate(row)
        if self.cooldown_remaining > 0:
            reasons.append("cooldown_violation")
            penalty += stake * config.cooldown_penalty_multiplier * self.cooldown_remaining

        drawdown = self._current_drawdown()
        if drawdown >= config.max_drawdown:
            reasons.append("drawdown_exposure")
            penalty += stake * config.drawdown_penalty_multiplier * (
                (drawdown - config.max_drawdown) / max(config.max_drawdown, 1e-9)
            )

        in_gray_zone = config.non_straight_max_rate < straight_rate < config.straight_min_rate
        if in_gray_zone:
            reasons.append("gray_zone_bet")
            penalty += stake * config.gray_zone_penalty_multiplier
        if direction == 1 and straight_rate < config.straight_min_rate:
            reasons.append("straight_below_threshold")
            penalty += stake * config.threshold_penalty_multiplier * (
                (config.straight_min_rate - straight_rate) / max(config.straight_min_rate, 1e-9)
            )
        if direction == 2 and straight_rate > config.non_straight_max_rate:
            reasons.append("non_straight_above_threshold")
            penalty += stake * config.threshold_penalty_multiplier * (
                (straight_rate - config.non_straight_max_rate) / max(config.non_straight_max_rate, 1e-9)
            )

        if direction == 2:
            worst_case_loss = stake * self._loss_multiplier(row, 2)
            exposure_ratio = worst_case_loss / max(self.initial_balance, 1e-9)
            if exposure_ratio > config.non_straight_exposure_limit:
                reasons.append("non_straight_exposure_high")
                penalty += stake * config.non_straight_exposure_penalty_multiplier * (
                    exposure_ratio - config.non_straight_exposure_limit
                )

        expected_roi = self._expected_roi(direction, row)
        if expected_roi < 0.0:
            reasons.append("negative_expected_edge")
            penalty += stake * abs(expected_roi) * config.negative_edge_penalty_multiplier

        stake_ratio = stake / max(self.balance, 1e-9)
        if stake_ratio > config.high_stake_ratio:
            reasons.append("high_stake")
            penalty += stake * config.high_stake_penalty_multiplier * (
                (stake_ratio - config.high_stake_ratio) / max(config.high_stake_ratio, 1e-9)
            )

        next_bet_streak = self.bet_streak + 1
        if config.frequent_bet_penalty_start > 0 and next_bet_streak >= config.frequent_bet_penalty_start:
            reasons.append("frequent_bet")
            excess = next_bet_streak - config.frequent_bet_penalty_start + 1
            penalty += stake * config.frequent_bet_penalty_multiplier * excess

        next_steps = self.current_step + 1
        next_bets = self.episode_bets + 1
        next_bet_rate = next_bets / max(next_steps, 1)
        if (
            config.max_bet_rate > 0.0
            and next_steps >= config.bet_rate_penalty_min_steps
            and next_bet_rate > config.max_bet_rate
        ):
            reasons.append("bet_rate_too_high")
            penalty += stake * config.bet_rate_penalty_multiplier * (
                (next_bet_rate - config.max_bet_rate) / max(config.max_bet_rate, 1e-9)
            )

        return penalty, tuple(reasons)

    def _watch_penalty(self, direction: int) -> tuple[float, tuple[str, ...]]:
        config = self.training_risk
        if not config.enabled or direction != 0:
            return 0.0, ()
        reasons: list[str] = []
        penalty = 0.0
        next_watch_streak = self.watch_streak + 1
        if config.watch_penalty_start > 0 and next_watch_streak >= config.watch_penalty_start:
            reasons.append("excessive_watch")
            penalty += config.excessive_watch_penalty

        next_steps = self.current_step + 1
        current_bet_rate = self.episode_bets / max(next_steps, 1)
        if (
            config.min_bet_rate > 0.0
            and next_steps >= config.low_bet_rate_penalty_min_steps
            and current_bet_rate < config.min_bet_rate
        ):
            reasons.append("bet_rate_too_low")
            penalty += self.initial_balance * config.low_bet_rate_penalty_multiplier * (
                (config.min_bet_rate - current_bet_rate) / max(config.min_bet_rate, 1e-9)
            )
        return penalty, tuple(reasons)

    def _expected_roi(self, direction: int, row: dict[str, Any]) -> float:
        if direction == 0:
            return 0.0
        straight_rate = self._straight_rate(row)
        if direction == 1:
            return straight_rate * self._profit_multiplier(row, 1) - (1.0 - straight_rate)
        if direction == 2:
            return (1.0 - straight_rate) * self._profit_multiplier(row, 2) - straight_rate * self._loss_multiplier(row, 2)

        spec = BET_SPECS[direction]
        return self._theoretical_roi(spec)

    @classmethod
    def _theoretical_roi(cls, spec: BetSpec) -> float:
        if spec.kind == "sum_exact":
            return _sum_probability(int(spec.target)) * spec.gross_odds - 1.0
        if spec.kind == "sum_simple":
            target = str(spec.target)
            probability = sum(_sum_probability(total) for total in range(28) if cls._category_matches(total, target))
            return probability * spec.gross_odds - 1.0
        if spec.kind == "sum_combo":
            target = str(spec.target)
            probability = sum(_sum_probability(total) for total in range(28) if cls._category_matches(total, target))
            return probability * spec.gross_odds - 1.0
        if spec.kind == "pattern":
            if spec.target == "pair":
                return 0.27 * spec.gross_odds - 1.0
            if spec.target == "leopard":
                return 0.01 * spec.gross_odds - 1.0
            if spec.target == "extreme_small":
                probability = sum(_sum_probability(total) for total in range(0, 6))
                return probability * spec.gross_odds - 1.0
            if spec.target == "extreme_big":
                probability = sum(_sum_probability(total) for total in range(22, 28))
                return probability * spec.gross_odds - 1.0
        if spec.kind == "digit_simple":
            return 0.5 * spec.gross_odds - 1.0
        if spec.kind == "digit_combo":
            _, target = spec.target
            probability = 0.2 if target in {"small_odd", "big_even"} else 0.3
            return probability * spec.gross_odds - 1.0
        if spec.kind == "digit_exact":
            return 0.1 * spec.gross_odds - 1.0
        return -1.0

    def _current_drawdown(self) -> float:
        return max(0.0, (self.peak_balance - self.balance) / max(self.peak_balance, 1e-9))

    @staticmethod
    def _straight_rate(row: dict[str, Any]) -> float:
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        return float(np.clip(features.get("cond_straight_rate", row.get("cond_straight_rate", 0.0)), 0.0, 1.0))

    @staticmethod
    def _row_multiplier(row: dict[str, Any], name: str, default: float) -> float:
        direct = row.get(f"{name}_profit_multiplier")
        if direct is not None:
            return float(direct)
        odds = row.get(f"{name}_odds")
        if odds is not None:
            return max(0.0, float(odds) - 1.0)
        return default

    @staticmethod
    def _pressure_score(row: dict[str, Any]) -> float:
        return float(np.clip(row["features"].get("pressure_score", row.get("pressure_score", 0.0)), 0.0, 100.0))
