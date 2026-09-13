from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed

from .casino_env import BET_SPECS, CasinoEnv, DIRECTION_LABELS, GameObjectiveConfig, TrainingGateConfig
from .config import LOG_DIR, MODEL_DIR
from .data_adapter import FEATURE_COLUMNS, load_history
from .evaluate_current import _format_table
from .file_policy import resolve_history_path, resolve_log_path, resolve_model_path
from .training_utils import resolve_device


DQN_MODEL_PATH = MODEL_DIR / "dqn_capital_game_v1.zip"
DQN_MIN_STAKE = 30.0
DQN_FIXED_STAKE_AMOUNTS = (30.0, 40.0, 50.0, 75.0, 100.0)
# Backward-compatible public name used by existing experiment/test tooling.
DQN_STAKE_AMOUNTS = DQN_FIXED_STAKE_AMOUNTS
DQN_STAKE_RATIOS = (0.08, 0.10, 0.15)
DQN_AMOUNT_BINS = tuple(range(len(DQN_FIXED_STAKE_AMOUNTS) + len(DQN_STAKE_RATIOS)))
DQN_LOOKBACK_WINDOWS = (100, 200, 400, 1000)
DQN_LOOKBACK_FEATURES = tuple(
    feature
    for window in DQN_LOOKBACK_WINDOWS
    for feature in (f"straight_rate_{window}", f"pressure_avg_{window}")
)
DQN_PLAY_LABELS = ("small_odd", "big_even", "small_even", "big_odd")
DQN_WAIT_ACTIONS = 1
GOOD_OPPORTUNITY_THRESHOLD = 0.3
WAIT_HIGH_OPPORTUNITY_THRESHOLD = 0.4
WAIT_MEDIUM_OPPORTUNITY_THRESHOLD = 0.2
DQN_NET_ARCH = [256, 256, 128]
DQN_CHASE_PENALTY_STREAK = 3
DQN_HARD_CHASE_PENALTY_STREAK = 5
DQN_SPRINT_BALANCE_MULTIPLIER = 1.15
DQN_LOW_OPPORTUNITY_WAIT_REWARD = 0.5
DQN_BET_ACTION_COST = 2.0
DQN_LOW_OPPORTUNITY_BET_COST = 8.0
DQN_PASS_BONUS_REWARD = 3500.0
DQN_PROFIT_LOCK_MULTIPLIER = 2.0
DQN_PROFIT_LOCK_WAIT_REWARD = 2000.0
DQN_PROFIT_LOCK_FAILED_SPRINT_PENALTY = 500.0
DQN_PASS_GAP_SINGLE_PLAY_HINT = 150.0
DQN_PASS_GAP_SINGLE_PLAY_HINT_REWARD = 5.0
DQN_EDGE_WAIT_REWARD = 5.0
DQN_WEAK_EDGE_WAIT_REWARD = 1.0
DQN_NEGATIVE_EDGE_BET_COST = 40.0
DQN_WEAK_EDGE_BET_COST = 5.0
DQN_WEAK_EDGE_THRESHOLD = 0.1
DQN_BET_STREAK_SCORE_STEP = 0.05
DQN_BET_STREAK_MAX_EXTRA_SCORE = 0.15
DQN_BET_STREAK_MISS_COST = 10.0
DQN_NEGATIVE_EDGE_LOSS_MULTIPLIER = 1.5
DQN_RISK_LOSS_DRAWDOWN_MULTIPLIER = 1.0
DQN_RISK_LOSS_STREAK_STEP = 0.25
DQN_LOW_BET_RATE_WAIT_PENALTY_MULTIPLIER = 0.25
DQN_REWARD_OBJECTIVES = ("capital", "risk", "profit", "balance")
DQN_RISK_WAIT_REWARD = 0.05
DQN_RISK_WAIT_PRESSURE_REWARD = 2.0
DQN_RISK_CLEAN_BET_REWARD = 4.0
DQN_RISK_BET_VIOLATION_COST = 60.0
DQN_RISK_IGNORED_GATE_REASONS = {"negative_expected_edge"}
DQN_RISK_TARGET_BETS = 30
DQN_RISK_MAX_BETS = 45
DQN_RISK_TARGET_BET_RATE = 0.15
DQN_RISK_PASS_BALANCE = 3000.0
DQN_RISK_CASHOUT_BALANCE = 2000.0
DQN_RISK_MIN_CASHOUT_BETS = 15
DQN_RISK_MAX_STEPS = 200
DQN_RISK_FLOOR_BALANCE = 500.0
DQN_MAX_STEPS = 200


def dqn_lookback_features(lookback_windows: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(
        feature
        for window in lookback_windows
        for feature in (f"straight_rate_{window}", f"pressure_avg_{window}")
    )


def max_lookback_window(lookback_windows: tuple[int, ...]) -> int:
    return max(lookback_windows, default=0)


def parse_lookback_windows(value: str | None) -> tuple[int, ...]:
    if value is None:
        return DQN_LOOKBACK_WINDOWS
    if not value.strip():
        return ()
    windows = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if any(window <= 0 for window in windows):
        raise ValueError("lookback windows must be positive integers")
    return windows


@dataclass(frozen=True)
class DqnActionSpec:
    label: str
    directions: tuple[int, ...]
    amount_bin: int
    stake_amount: float | None = None
    stake_ratio: float | None = None

    @property
    def is_wait(self) -> bool:
        return not self.directions

    @property
    def direction(self) -> int:
        return self.directions[0] if self.directions else 0

    @property
    def stake_per_play(self) -> float | None:
        """Compatibility view for fixed-stake action specs."""

        if self.stake_amount is not None:
            return self.stake_amount
        # Ratio actions still expose a finite comparison value for legacy
        # metadata/tests; execution continues to use ``stake_ratio``.
        return DQN_FIXED_STAKE_AMOUNTS[self.amount_bin % len(DQN_FIXED_STAKE_AMOUNTS)]


def build_dqn_action_specs() -> tuple[DqnActionSpec, ...]:
    specs: list[DqnActionSpec] = [
        DqnActionSpec(f"WAIT_{index}", (), 0) for index in range(DQN_WAIT_ACTIONS)
    ]
    play_directions = tuple(
        next(key for key, value in DIRECTION_LABELS.items() if value == label)
        for label in DQN_PLAY_LABELS
    )
    for size in (1, 2, 3):
        for directions in itertools.combinations(play_directions, size):
            label = "+".join(DIRECTION_LABELS[direction] for direction in directions)
            amount_bin = 0
            for stake_amount in DQN_FIXED_STAKE_AMOUNTS:
                specs.append(DqnActionSpec(label, directions, amount_bin, stake_amount, None))
                amount_bin += 1
            for stake_ratio in DQN_STAKE_RATIOS:
                specs.append(DqnActionSpec(label, directions, amount_bin, None, stake_ratio))
                amount_bin += 1
    return tuple(specs)


DQN_ACTION_SPECS = build_dqn_action_specs()


class DqnCapitalEnv(gym.Wrapper):
    def __init__(
        self,
        env: CasinoEnv,
        *,
        action_specs: tuple[DqnActionSpec, ...] = DQN_ACTION_SPECS,
        random_start: bool = False,
        wait_reward: float = 0.0,
        low_bet_rate_wait_penalty: float = 1.0,
        lookback_windows: tuple[int, ...] = DQN_LOOKBACK_WINDOWS,
        reward_objective: str = "capital",
        risk_target_bets: int = DQN_RISK_TARGET_BETS,
        risk_max_bets: int = DQN_RISK_MAX_BETS,
        risk_min_cashout_bets: int = DQN_RISK_MIN_CASHOUT_BETS,
        risk_pass_balance: float = DQN_RISK_PASS_BALANCE,
        risk_cashout_balance: float = DQN_RISK_CASHOUT_BALANCE,
    ) -> None:
        super().__init__(env)
        self.action_specs = action_specs
        self.lookback_windows = lookback_windows
        self.lookback_features = dqn_lookback_features(lookback_windows)
        self.reward_objective = reward_objective
        self.risk_target_bets = int(risk_target_bets)
        self.risk_max_bets = int(risk_max_bets)
        self.risk_min_cashout_bets = int(risk_min_cashout_bets)
        self.risk_pass_balance = float(risk_pass_balance)
        self.risk_cashout_balance = float(risk_cashout_balance)
        self.random_start = random_start
        self.wait_reward = float(wait_reward)
        self.low_bet_rate_wait_penalty = float(low_bet_rate_wait_penalty)
        self.profit_lock_seen = False
        self.action_space = spaces.Discrete(len(action_specs))
        aux_low = np.zeros(6 + len(self.lookback_features), dtype=np.float32)
        aux_high = np.array(
            [10.0, 1.0, 10.0, 1.0, 10.0, 1.0, *([1.0] * len(self.lookback_features))],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.concatenate([env.observation_space.low, aux_low]).astype(np.float32),
            high=np.concatenate([env.observation_space.high, aux_high]).astype(np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if options is None and self.random_start:
            max_start = max(0, len(self.env.history) - self.env.game_objective.max_steps - 1)
            min_start = max_lookback_window(self.lookback_windows)
            rng = np.random.default_rng(seed)
            options = {
                "start_step": (
                    int(rng.integers(min_start, max_start + 1))
                    if max_start >= min_start
                    else 0
                )
            }
        obs, info = self.env.reset(seed=seed, options=options)
        self.profit_lock_seen = False
        return self._augment_obs(obs), info

    def step(self, action):
        action_index = int(action)
        spec = self.action_specs[action_index]
        edge_score = self._historical_edge_score(spec)
        if self.reward_objective == "risk" and spec.is_wait and self._must_catch_up_bets():
            action_index = self._best_catch_up_action()
            spec = self.action_specs[action_index]
            edge_score = self._historical_edge_score(spec)
        if self._risk_cashout_active() and spec.is_wait:
            return self._risk_cashout_wait_result(action_index, spec, edge_score)
        if self.reward_objective != "risk" and self._profit_lock_active() and spec.is_wait:
            info = {
                **self.env.last_info,
                "game_objective_reason": "dqn_profit_lock_cashout",
                "balance": round(self.env.balance, 6),
                "profit_loss": 0.0,
                "stake": 0.0,
            }
            return self._augment_obs(self.env._get_obs()), DQN_PROFIT_LOCK_WAIT_REWARD, False, True, {
                **info,
                "dqn_action": action_index,
                "dqn_action_label": spec.label,
                "dqn_directions": spec.directions,
                "dqn_amount_bin": spec.amount_bin,
                "dqn_profit_lock_active": True,
                "dqn_profit_lock_seen": True,
                "dqn_historical_edge_score": round(edge_score, 6),
                "dqn_adjusted_reward": DQN_PROFIT_LOCK_WAIT_REWARD,
            }
        cashout_sprint = self._risk_cashout_active() and not spec.is_wait
        extra_sprint = (
            self.reward_objective != "risk"
            and self._profit_lock_active()
            and self.env.episode_bets >= self.env.game_objective.max_bets
        )
        obs, reward, terminated, truncated, info = self.env.step_package(
            spec.directions,
            spec.amount_bin,
            stake_per_play=self._stake_per_play(spec),
        )
        if self._profit_lock_active():
            self.profit_lock_seen = True
        adjusted_reward, forced_done, force_reason = self._shape_reward(info, spec, edge_score)
        if extra_sprint and not spec.is_wait:
            adjusted_reward, force_reason = self._profit_lock_sprint_reward(info)
            terminated = False
            truncated = True
        if cashout_sprint and str(info.get("game_objective_reason", "none")) != "game_pass":
            adjusted_reward += self._risk_cashout_sprint_reward(info)
            force_reason = "risk_cashout_sprint_failed"
            terminated = False
            truncated = True
        if forced_done:
            truncated = True
            info = {**info, "game_objective_reason": force_reason}
        elif force_reason:
            info = {**info, "game_objective_reason": force_reason}
        info = {
            **info,
            "dqn_action": action_index,
            "dqn_action_label": spec.label,
            "dqn_directions": spec.directions,
            "dqn_amount_bin": spec.amount_bin,
            "dqn_profit_lock_active": self._profit_lock_active(),
            "dqn_profit_lock_seen": self.profit_lock_seen,
            "dqn_historical_edge_score": round(edge_score, 6),
            "dqn_adjusted_reward": round(adjusted_reward, 6),
        }
        return self._augment_obs(obs), adjusted_reward, terminated, truncated, info

    def _must_catch_up_bets(self) -> bool:
        if self.env.balance < self.env.initial_balance * 0.7:
            return False
        if self.env._current_drawdown() >= 0.45:
            return False
        if self._risk_cashout_active():
            return self.env.episode_bets < self.risk_min_cashout_bets
        steps_left = max(0, self.env.game_objective.max_steps - self.env.episode_steps)
        target_left = max(0, self.risk_target_bets - self.env.episode_bets)
        if target_left <= 0:
            return False
        return target_left >= steps_left

    def _best_catch_up_action(self) -> int:
        candidates = [
            (index, spec)
            for index, spec in enumerate(self.action_specs)
            if not spec.is_wait and len(spec.directions) == 1 and spec.stake_amount == self.env.min_stake
        ]
        if not candidates:
            return next(index for index, spec in enumerate(self.action_specs) if not spec.is_wait)
        return max(candidates, key=lambda item: self._catch_up_reference_score(item[1]))[0]

    def _catch_up_reference_score(self, spec: DqnActionSpec, *, lookback: int = 1000) -> float:
        current = max(0, int(self.env.current_step))
        sample = self.env.history[max(0, current - lookback):current]
        if spec.is_wait or not sample:
            return 0.0
        stake_per_play = self._stake_per_play(spec)
        pnl_total = 0.0
        current_loss_streak = 0
        max_loss_streak = 0
        for row in sample:
            pnl = sum(
                self.env._settle_bet(direction, stake_per_play, row, 0.0).profit_loss
                for direction in spec.directions
            )
            pnl_total += pnl
            if pnl > 0.0:
                current_loss_streak = 0
            elif pnl < 0.0:
                current_loss_streak += 1
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        avg_pnl = pnl_total / max(len(sample), 1)
        return avg_pnl - 0.5 * max_loss_streak

    def _profit_lock_active(self) -> bool:
        return self.env.balance >= self.env.initial_balance * DQN_PROFIT_LOCK_MULTIPLIER

    def _risk_cashout_active(self) -> bool:
        return (
            self.reward_objective == "risk"
            and self.risk_cashout_balance <= self.env.balance < self.risk_pass_balance
        )

    def _risk_cashout_wait_result(self, action_index: int, spec: DqnActionSpec, edge_score: float):
        if self.env.episode_bets < self.risk_min_cashout_bets:
            deficit = self.risk_min_cashout_bets - self.env.episode_bets
            info = {
                **self.env.last_info,
                "game_objective_reason": "risk_cashout_under_min_bets",
                "balance": round(self.env.balance, 6),
                "profit_loss": 0.0,
                "stake": 0.0,
                "episode_bets": self.env.episode_bets,
                "episode_steps": self.env.episode_steps,
            }
            reward = -1200.0 - 150.0 * deficit
            return self._augment_obs(self.env._get_obs()), reward, False, True, {
                **info,
                "dqn_action": action_index,
                "dqn_action_label": spec.label,
                "dqn_directions": spec.directions,
                "dqn_amount_bin": spec.amount_bin,
                "dqn_profit_lock_active": False,
                "dqn_profit_lock_seen": self.profit_lock_seen,
                "dqn_historical_edge_score": round(edge_score, 6),
                "dqn_adjusted_reward": round(reward, 6),
            }
        info = {
            **self.env.last_info,
            "game_objective_reason": "risk_cashout",
            "balance": round(self.env.balance, 6),
            "profit_loss": 0.0,
            "stake": 0.0,
            "episode_bets": self.env.episode_bets,
            "episode_steps": self.env.episode_steps,
        }
        return self._augment_obs(self.env._get_obs()), 1000.0, False, True, {
            **info,
            "dqn_action": action_index,
            "dqn_action_label": spec.label,
            "dqn_directions": spec.directions,
            "dqn_amount_bin": spec.amount_bin,
            "dqn_profit_lock_active": False,
            "dqn_profit_lock_seen": self.profit_lock_seen,
            "dqn_historical_edge_score": round(edge_score, 6),
            "dqn_adjusted_reward": 1000.0,
        }

    def _risk_cashout_sprint_reward(self, info: dict[str, Any]) -> float:
        balance = float(info.get("balance", self.env.balance))
        if balance >= self.risk_cashout_balance:
            return 100.0
        return -500.0

    def _profit_lock_sprint_reward(self, info: dict[str, Any]) -> tuple[float, str]:
        wealth = float(info.get("balance", self.env.balance))
        pass_balance = self.env.initial_balance * self.env.game_objective.pass_balance_multiplier
        cashout_balance = self.env.initial_balance * DQN_PROFIT_LOCK_MULTIPLIER
        if wealth >= pass_balance:
            return DQN_PASS_BONUS_REWARD, "game_pass"
        if wealth >= cashout_balance:
            return DQN_PROFIT_LOCK_WAIT_REWARD, "dqn_profit_lock_cashout"
        return -DQN_PROFIT_LOCK_FAILED_SPRINT_PENALTY, "dqn_profit_lock_failed_sprint"

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        max_bets = max(self.env.game_objective.max_bets, 1)
        max_steps = max(self.env.game_objective.max_steps, 1)
        lookback = self._lookback_features()
        aux = np.array(
            [
                self.env.balance / max(self.env.initial_balance, 1e-9),
                self.env._current_drawdown(),
                10.0 * max(0, max_bets - self.env.episode_bets) / max_bets,
                self.env.episode_bets / max(self.env.episode_steps, 1),
                min(self.env.losing_streak / 10.0, 10.0),
                self.env.episode_steps / max_steps,
                *lookback,
            ],
            dtype=np.float32,
        )
        return np.clip(np.concatenate([obs.astype(np.float32), aux]), self.observation_space.low, self.observation_space.high)

    def _lookback_features(self) -> list[float]:
        current = max(0, int(self.env.current_step))
        values: list[float] = []
        for window in self.lookback_windows:
            start = max(0, current - window)
            sample = self.env.history[start:current]
            if not sample:
                values.extend([0.0, 0.0])
                continue
            straight_rate = sum(1 for row in sample if bool(row["outcome_is_straight"])) / len(sample)
            pressure_avg = sum(
                float(row["features"].get("pressure_score", row.get("pressure_score", 0.0)))
                for row in sample
            ) / len(sample)
            values.extend([straight_rate, pressure_avg / 100.0])
        return values

    def _historical_edge_score(self, spec: DqnActionSpec) -> float:
        if spec.is_wait:
            bet_specs = [item for item in self.action_specs if not item.is_wait]
            if not bet_specs:
                return 0.0
            return max(self._historical_edge_score(item) for item in bet_specs)

        current = max(0, int(self.env.current_step))
        if current <= 0:
            return 0.0
        stake_per_play = self._stake_per_play(spec)
        total_stake = stake_per_play * max(len(spec.directions), 1)
        if total_stake <= 0.0:
            return 0.0

        window_edges: list[float] = []
        for window in self.lookback_windows:
            start = max(0, current - window)
            sample = self.env.history[start:current]
            if not sample:
                continue
            pnl_total = 0.0
            for row in sample:
                pnl_total += sum(
                    self.env._settle_bet(direction, stake_per_play, row, 0.0).profit_loss
                    for direction in spec.directions
                )
            window_edges.append(pnl_total / (len(sample) * total_stake))
        if not window_edges:
            return 0.0
        return float(np.clip(min(window_edges), -1.0, 1.0))

    def _stake_per_play(self, spec: DqnActionSpec) -> float:
        if spec.is_wait:
            return 0.0
        if spec.stake_amount is not None:
            return min(self.env.max_stake, max(self.env.min_stake, float(spec.stake_amount)))
        ratio = float(spec.stake_ratio or 0.0)
        stake = self.env.balance * ratio
        return min(self.env.max_stake, max(self.env.min_stake, stake))

    def _shape_reward(
        self,
        info: dict[str, Any],
        spec: DqnActionSpec,
        historical_edge_score: float,
    ) -> tuple[float, bool, str | None]:
        if self.reward_objective == "risk":
            reward = self._risk_control_reward(info, spec)
            force_reason = self._risk_force_reason(info)
            forced_done = force_reason == "risk_balance_floor_breach"
            if force_reason == "risk_balance_floor_breach":
                reward -= 2500.0 + 5.0 * max(0.0, DQN_RISK_FLOOR_BALANCE - float(info.get("balance", self.env.balance)))
            elif force_reason == "risk_cashout_under_min_bets":
                reward -= 1200.0 + 150.0 * max(0, self.risk_min_cashout_bets - self.env.episode_bets)
            elif force_reason == "risk_under_target_bets":
                reward -= 600.0 * (self.risk_target_bets - self.env.episode_bets)
            elif force_reason == "risk_over_max_bets":
                reward -= 1000.0 * (self.env.episode_bets - self.risk_max_bets)
            elif str(info.get("game_objective_reason", "none")) == "game_pass":
                reward += 3000.0
            elif str(info.get("game_objective_reason", "none")) == "game_complete":
                reward += 1000.0
            return reward, forced_done, force_reason

        if self.reward_objective == "profit":
            return self._profit_only_reward(info, spec), False, None
        if self.reward_objective == "balance":
            return self._balance_first_reward(info, spec, historical_edge_score), False, None

        pnl = float(info.get("profit_loss", 0.0))
        wealth_after = float(info.get("balance", self.env.balance))
        opportunity_score = self._opportunity_score(info)
        if spec.is_wait:
            if self._wait_makes_pass_unreachable(wealth_after):
                return -500.0, True, "dqn_unreachable_wait"
            shaped = (
                self.wait_reward
                + self._wait_opportunity_reward(opportunity_score)
                + self._edge_wait_reward(historical_edge_score)
                - self._low_bet_rate_wait_penalty()
            )
            if self._profit_lock_active():
                shaped += DQN_PROFIT_LOCK_WAIT_REWARD
        else:
            shaped = self._bet_opportunity_reward(info, pnl, opportunity_score, historical_edge_score)

        reason = info.get("game_objective_reason")
        if reason == "game_pass":
            return float(shaped + DQN_PASS_BONUS_REWARD), False, None
        if reason in {"game_fail", "game_bet_limit", "game_complete"}:
            return float(shaped + self._terminal_objective_reward(info)), False, None
        return float(shaped), False, None

    def _profit_only_reward(self, info: dict[str, Any], spec: DqnActionSpec) -> float:
        pnl = float(info.get("profit_loss", 0.0))
        reward = pnl
        if spec.is_wait:
            reward -= 0.2
        reason = str(info.get("game_objective_reason", "none"))
        if reason in {"game_pass", "game_complete", "game_fail"}:
            reward += 2.0 * (float(info.get("balance", self.env.balance)) - self.env.initial_balance)
        return reward

    def _balance_first_reward(self, info: dict[str, Any], spec: DqnActionSpec, historical_edge_score: float) -> float:
        balance = float(info.get("balance", self.env.balance))
        pnl = float(info.get("profit_loss", 0.0))
        drawdown = max(0.0, float(info.get("drawdown", 0.0)))
        reward = pnl
        if spec.is_wait:
            reward += 0.1 if historical_edge_score <= 0.0 else -0.2
        elif historical_edge_score <= 0.0:
            reward -= 60.0 + 120.0 * abs(historical_edge_score)
        elif historical_edge_score < DQN_WEAK_EDGE_THRESHOLD:
            reward -= 15.0
        else:
            reward += 4.0 * min(historical_edge_score, 0.5)
        if balance < 600.0:
            reward -= 10.0 * (600.0 - balance)
        elif balance < 700.0:
            reward -= 2.0 * (700.0 - balance)
        if drawdown > 0.5:
            reward -= 1600.0 * (drawdown - 0.5)
        elif drawdown > 0.4:
            reward -= 300.0 * (drawdown - 0.4)
        elif drawdown > 0.3:
            reward -= 80.0 * (drawdown - 0.3)
        reason = str(info.get("game_objective_reason", "none"))
        if reason in {"game_pass", "game_complete", "game_fail", "game_bet_limit"}:
            reward += 3.0 * (balance - self.env.initial_balance)
            if balance < 600.0:
                reward -= 3500.0
            elif balance < 700.0:
                reward -= 1000.0
            if drawdown > 0.6:
                reward -= 1500.0
            elif drawdown > 0.5:
                reward -= 600.0
        return reward

    def _risk_control_reward(self, info: dict[str, Any], spec: DqnActionSpec) -> float:
        pressure = self._risk_pressure_score(info)
        bet_budget_pressure = self._bet_budget_pressure()
        bet_deficit_pressure = self._bet_deficit_pressure()
        schedule_deficit = self._bet_schedule_deficit()
        schedule_surplus = self._bet_schedule_surplus()
        balance = float(info.get("balance", self.env.balance))
        pass_reached = balance >= self.risk_pass_balance
        cashout_reached = balance >= self.risk_cashout_balance
        floor_gap = max(0.0, DQN_RISK_FLOOR_BALANCE - balance)
        if spec.is_wait:
            if pass_reached:
                return 2000.0
            if cashout_reached:
                if self.env.episode_bets < self.risk_min_cashout_bets:
                    return -1000.0 - 150.0 * (self.risk_min_cashout_bets - self.env.episode_bets)
                return 500.0
            if floor_gap > 0.0:
                return 10.0
            if balance < self.env.initial_balance * 0.7:
                return 20.0 + 20.0 * pressure
            if schedule_deficit > 0.25:
                return -35.0 - 15.0 * min(schedule_deficit, 5.0)
            return (
                DQN_RISK_WAIT_REWARD
                + DQN_RISK_WAIT_PRESSURE_REWARD * max(pressure, bet_budget_pressure)
                + min(schedule_surplus, 3.0)
            )

        bad_reasons = self._bad_gate_reasons(info)
        watch_penalty = max(0.0, float(info.get("watch_penalty", 0.0)))
        previous_balance = balance - float(info.get("profit_loss", 0.0))
        if floor_gap > 0.0:
            return -250.0 - 2.0 * floor_gap
        if balance < self.env.initial_balance * 0.7:
            return -220.0 - 250.0 * (0.7 - balance / max(self.env.initial_balance, 1e-9))
        if balance < self.env.initial_balance * 0.85 and pressure > 0.25:
            return -120.0 - 120.0 * pressure
        if previous_balance >= self.risk_pass_balance:
            return -DQN_RISK_BET_VIOLATION_COST * 4.0
        if previous_balance >= self.risk_cashout_balance:
            return -DQN_RISK_BET_VIOLATION_COST * 2.0
        if self.env.episode_bets > self.risk_max_bets:
            return -DQN_RISK_BET_VIOLATION_COST * 2.0
        if bad_reasons or watch_penalty > 0.0:
            scaled_penalty = min(100.0, watch_penalty / max(float(info.get("stake", 1.0)), 1.0))
            cooldown_cost = 40.0 if "cooldown_violation" in bad_reasons else 0.0
            return -DQN_RISK_BET_VIOLATION_COST - 10.0 * len(bad_reasons) - cooldown_cost - scaled_penalty
        package_size = max(1, len(spec.directions))
        package_cost = 14.0 * max(0, package_size - 1)
        schedule_reward = 35.0 * min(max(schedule_deficit, 0.5), 4.0)
        surplus_cost = 6.0 * min(schedule_surplus, 5.0)
        return (
            DQN_RISK_CLEAN_BET_REWARD * 1.5
            + schedule_reward
            + 18.0 * bet_deficit_pressure
            - pressure
            - 8.0 * bet_budget_pressure
            - surplus_cost
            - package_cost
        )

    def _risk_force_reason(self, info: dict[str, Any]) -> str | None:
        reason = str(info.get("game_objective_reason", "none"))
        if reason == "game_pass":
            return None
        balance = float(info.get("balance", self.env.balance))
        if balance < DQN_RISK_FLOOR_BALANCE:
            return "risk_balance_floor_breach"
        if reason != "game_complete":
            return None
        if balance >= self.risk_cashout_balance:
            if self.env.episode_bets < self.risk_min_cashout_bets:
                return "risk_cashout_under_min_bets"
            return "risk_cashout"
        if self.env.episode_bets < self.risk_target_bets:
            return "risk_under_target_bets"
        if self.env.episode_bets > self.risk_max_bets:
            return "risk_over_max_bets"
        return None

    def _bet_budget_pressure(self) -> float:
        used_ratio = self.env.episode_bets / max(self.risk_target_bets, 1)
        return float(np.clip((used_ratio - 0.9) / 0.1, 0.0, 1.0))

    def _bet_deficit_pressure(self) -> float:
        steps_left = max(0, self.env.game_objective.max_steps - self.env.episode_steps)
        target_left = max(0, self.risk_target_bets - self.env.episode_bets)
        if target_left <= 0:
            return 0.0
        required_rate = target_left / max(steps_left, 1)
        return float(np.clip(required_rate / 0.5, 0.0, 1.0))

    def _bet_schedule_deficit(self) -> float:
        expected_bets = self.env.episode_steps * (
            self.risk_target_bets / max(self.env.game_objective.max_steps, 1)
        )
        return max(0.0, expected_bets - self.env.episode_bets)

    def _bet_schedule_surplus(self) -> float:
        expected_bets = self.env.episode_steps * (
            self.risk_target_bets / max(self.env.game_objective.max_steps, 1)
        )
        return max(0.0, self.env.episode_bets - expected_bets)

    @staticmethod
    def _bad_gate_reasons(info: dict[str, Any]) -> tuple[str, ...]:
        raw_reasons = info.get("gate_reasons") or info.get("risk_reasons") or ()
        if isinstance(raw_reasons, str):
            raw_reasons = (raw_reasons,)
        return tuple(
            reason
            for reason in raw_reasons
            if reason not in {"allowed", "none"} and reason not in DQN_RISK_IGNORED_GATE_REASONS
        )

    @staticmethod
    def _risk_pressure_score(info: dict[str, Any]) -> float:
        drawdown = max(0.0, float(info.get("drawdown", 0.0)))
        losing_streak = max(0, int(info.get("losing_streak", 0)))
        cooldown = max(0, int(info.get("cooldown_remaining_before", info.get("cooldown_remaining", 0))))
        bet_streak = max(0, int(info.get("bet_streak", 0)))
        bet_rate = max(0.0, float(info.get("bet_rate", 0.0)))
        score = 0.0
        if cooldown > 0:
            score += 0.45
        score += min(0.25, drawdown / 0.35 * 0.25)
        score += min(0.15, losing_streak / 5.0 * 0.15)
        score += min(0.10, max(0, bet_streak - 2) / 5.0 * 0.10)
        if bet_rate > 0.35:
            score += min(0.20, (bet_rate - 0.35) / 0.65 * 0.20)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _opportunity_score(info: dict[str, Any]) -> float:
        pressure_score = float(info.get("pressure_score", 0.0))
        pressure_component = pressure_score / 100.0
        bet_rate = float(info.get("bet_rate", 0.0))
        target_rate = float(info.get("game_target_bet_rate", 0.2))
        target_rate = target_rate if target_rate > 0.0 else 0.2
        bet_rate_component = 1.0 - min(1.0, bet_rate / max(target_rate, 1e-9))
        drawdown_component = 1.0 - float(info.get("drawdown", 0.0))
        loss_streak_component = 1.0 - min(1.0, float(info.get("losing_streak", 0)) / 5.0)
        score = (
            0.55 * pressure_component
            + 0.20 * bet_rate_component
            + 0.15 * drawdown_component
            + 0.10 * loss_streak_component
        )
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _wait_opportunity_reward(opportunity_score: float) -> float:
        if opportunity_score > WAIT_HIGH_OPPORTUNITY_THRESHOLD:
            return -0.3
        if opportunity_score > WAIT_MEDIUM_OPPORTUNITY_THRESHOLD:
            return -0.1
        return DQN_LOW_OPPORTUNITY_WAIT_REWARD

    @staticmethod
    def _edge_wait_reward(historical_edge_score: float) -> float:
        if historical_edge_score <= 0.0:
            return DQN_EDGE_WAIT_REWARD
        if historical_edge_score <= DQN_WEAK_EDGE_THRESHOLD:
            return DQN_WEAK_EDGE_WAIT_REWARD
        return 0.0

    @staticmethod
    def _edge_bet_cost(historical_edge_score: float) -> float:
        if historical_edge_score <= 0.0:
            return DQN_NEGATIVE_EDGE_BET_COST
        if historical_edge_score <= DQN_WEAK_EDGE_THRESHOLD:
            return DQN_WEAK_EDGE_BET_COST
        return 0.0

    def _bet_opportunity_reward(
        self,
        info: dict[str, Any],
        pnl: float,
        opportunity_score: float,
        historical_edge_score: float,
    ) -> float:
        losing_streak = int(info.get("losing_streak", 0))
        action_cost = DQN_BET_ACTION_COST
        if opportunity_score <= WAIT_MEDIUM_OPPORTUNITY_THRESHOLD:
            action_cost += DQN_LOW_OPPORTUNITY_BET_COST
        action_cost += self._bet_streak_cost(info, opportunity_score)
        action_cost += self._edge_bet_cost(historical_edge_score)
        profit_lock_single_play_hint = self._pass_gap_single_play_hint(info)
        if pnl > 0.0:
            multiplier = 3.0 if opportunity_score > GOOD_OPPORTUNITY_THRESHOLD else 1.0
            if losing_streak >= DQN_CHASE_PENALTY_STREAK:
                multiplier *= 0.5
            reward = multiplier * pnl
            if float(info.get("balance", self.env.balance)) >= self.env.initial_balance * DQN_SPRINT_BALANCE_MULTIPLIER:
                reward += 2.0 * pnl
            return reward - action_cost + profit_lock_single_play_hint

        multiplier = 0.5 if opportunity_score > GOOD_OPPORTUNITY_THRESHOLD else 2.0
        if losing_streak >= DQN_CHASE_PENALTY_STREAK:
            multiplier *= 1.5
        multiplier *= self._risk_loss_multiplier(info, historical_edge_score)
        reward = -multiplier * abs(pnl)
        if losing_streak >= DQN_HARD_CHASE_PENALTY_STREAK:
            reward -= 2.0 * abs(pnl)
        return reward - action_cost + profit_lock_single_play_hint

    @staticmethod
    def _risk_loss_multiplier(info: dict[str, Any], historical_edge_score: float) -> float:
        drawdown = max(0.0, float(info.get("drawdown", 0.0)))
        losing_streak = max(0, int(info.get("losing_streak", 0)))
        multiplier = 1.0 + drawdown * DQN_RISK_LOSS_DRAWDOWN_MULTIPLIER
        multiplier += min(1.0, losing_streak * DQN_RISK_LOSS_STREAK_STEP)
        if historical_edge_score <= 0.0:
            multiplier *= DQN_NEGATIVE_EDGE_LOSS_MULTIPLIER
        return multiplier

    def _pass_gap_single_play_hint(self, info: dict[str, Any]) -> float:
        pass_balance = self.env.initial_balance * self.env.game_objective.pass_balance_multiplier
        balance = float(info.get("balance", self.env.balance))
        if pass_balance - balance > DQN_PASS_GAP_SINGLE_PLAY_HINT:
            return 0.0
        if int(info.get("package_size", 1)) != 1:
            return 0.0
        return DQN_PASS_GAP_SINGLE_PLAY_HINT_REWARD

    @staticmethod
    def _bet_streak_required_score(info: dict[str, Any]) -> float:
        bet_streak = max(0, int(info.get("bet_streak", 1)) - 1)
        extra = min(DQN_BET_STREAK_MAX_EXTRA_SCORE, bet_streak * DQN_BET_STREAK_SCORE_STEP)
        return GOOD_OPPORTUNITY_THRESHOLD + extra

    @classmethod
    def _bet_streak_cost(cls, info: dict[str, Any], opportunity_score: float) -> float:
        required_score = cls._bet_streak_required_score(info)
        if opportunity_score >= required_score:
            return 0.0
        miss_steps = (required_score - opportunity_score) / max(DQN_BET_STREAK_SCORE_STEP, 1e-9)
        return DQN_BET_STREAK_MISS_COST * miss_steps

    def _terminal_objective_reward(self, info: dict[str, Any]) -> float:
        wealth = float(info.get("balance", self.env.balance))
        if wealth >= self.env.initial_balance * self.env.game_objective.pass_balance_multiplier:
            return DQN_PASS_BONUS_REWARD
        if self.env.game_objective.max_bets > 0 and int(info.get("episode_bets", self.env.episode_bets)) > self.env.game_objective.max_bets:
            return -500.0
        target_ratio = self.env.game_objective.pass_balance_multiplier
        return -100.0 * (target_ratio - wealth / max(self.env.initial_balance, 1e-9))

    def _wait_makes_pass_unreachable(self, wealth: float) -> bool:
        pass_balance = self.env.initial_balance * self.env.game_objective.pass_balance_multiplier
        wealth_gap = pass_balance - wealth
        if wealth_gap <= 0.0:
            return False
        max_bets = self.env.game_objective.max_bets
        bets_left = max(0, max_bets - self.env.episode_bets)
        periods_left = max(0, self.env.game_objective.max_steps - self.env.episode_steps)
        available_bets = min(bets_left, periods_left)
        max_possible_gain = available_bets * (wealth * 0.02)
        return wealth_gap > max_possible_gain + 50.0

    def _low_bet_rate_wait_penalty(self) -> float:
        target = self.env.game_objective.target_bet_rate
        if self.low_bet_rate_wait_penalty <= 0.0 or target <= 0.0:
            return 0.0
        expected_bets = self.env.episode_steps * target
        if self.env.episode_bets >= expected_bets:
            return 0.0
        shortfall = (expected_bets - self.env.episode_bets) / max(expected_bets, 1.0)
        return self.low_bet_rate_wait_penalty * DQN_LOW_BET_RATE_WAIT_PENALTY_MULTIPLIER * min(1.0, max(0.0, shortfall))


@dataclass(frozen=True)
class DqnTrainSettings:
    model_path: Path = DQN_MODEL_PATH
    total_timesteps: int = 100_000
    initial_balance: float = 1000.0
    max_stake: float = 100.0
    min_stake: float = DQN_MIN_STAKE
    max_steps: int = DQN_MAX_STEPS
    device: str = "auto"
    seed: int = 42
    learning_rate: float = 0.0001
    buffer_size: int = 100_000
    learning_starts: int = 1000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = -1
    target_update_interval: int = 2000
    exploration_fraction: float = 0.5
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    max_grad_norm: float = 10.0
    verbose: int = 1
    check_env: bool = True
    resume: bool = False
    wait_reward: float = 0.0
    low_bet_rate_wait_penalty: float = 1.0
    lookback_windows: tuple[int, ...] = DQN_LOOKBACK_WINDOWS
    reward_objective: str = "capital"
    risk_target_bets: int = DQN_RISK_TARGET_BETS
    risk_max_bets: int = DQN_RISK_MAX_BETS
    risk_min_cashout_bets: int = DQN_RISK_MIN_CASHOUT_BETS
    risk_max_steps: int = DQN_RISK_MAX_STEPS
    risk_target_bet_rate: float = DQN_RISK_TARGET_BET_RATE
    risk_pass_balance: float = DQN_RISK_PASS_BALANCE
    risk_cashout_balance: float = DQN_RISK_CASHOUT_BALANCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_path", resolve_model_path(self.model_path))
        if self.total_timesteps <= 0:
            raise ValueError("total_timesteps must be positive")
        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if self.max_stake <= 0:
            raise ValueError("max_stake must be positive")
        if self.min_stake <= 0:
            raise ValueError("min_stake must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.learning_starts < 0:
            raise ValueError("learning_starts cannot be negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.train_freq <= 0:
            raise ValueError("train_freq must be positive")
        if self.gradient_steps < -1:
            raise ValueError("gradient_steps must be -1 or non-negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.wait_reward < 0:
            raise ValueError("wait_reward cannot be negative")
        if self.low_bet_rate_wait_penalty < 0:
            raise ValueError("low_bet_rate_wait_penalty cannot be negative")
        if any(window <= 0 for window in self.lookback_windows):
            raise ValueError("lookback_windows must contain positive integers")
        if self.reward_objective not in DQN_REWARD_OBJECTIVES:
            raise ValueError(f"reward_objective must be one of: {', '.join(DQN_REWARD_OBJECTIVES)}")
        if self.risk_target_bets <= 0:
            raise ValueError("risk_target_bets must be positive")
        if self.risk_max_bets < self.risk_target_bets:
            raise ValueError("risk_max_bets must be >= risk_target_bets")
        if self.risk_min_cashout_bets <= 0:
            raise ValueError("risk_min_cashout_bets must be positive")
        if self.risk_max_steps <= 0:
            raise ValueError("risk_max_steps must be positive")
        if self.risk_target_bet_rate <= 0:
            raise ValueError("risk_target_bet_rate must be positive")
        if self.risk_pass_balance <= self.initial_balance:
            raise ValueError("risk_pass_balance must be greater than initial_balance")
        if self.risk_cashout_balance <= self.initial_balance:
            raise ValueError("risk_cashout_balance must be greater than initial_balance")


@dataclass(frozen=True)
class DqnHistoryPreflight:
    label: str
    rows: int
    required_rows: int
    max_lookback: int
    max_steps: int
    complete_start_min: int
    complete_start_max: int
    ok: bool


@dataclass(frozen=True)
class DqnRiskGameObjectiveConfig(GameObjectiveConfig):
    explicit_max_bets: int = DQN_RISK_MAX_BETS

    @property
    def max_bets(self) -> int:
        return max(0, int(self.explicit_max_bets))


def build_dqn_game_objective(max_steps: int = DQN_MAX_STEPS) -> GameObjectiveConfig:
    return GameObjectiveConfig(
        enabled=True,
        max_steps=max_steps,
        pass_balance_multiplier=3.0,
        fail_balance_multiplier=0.7,
        target_bet_rate=0.22,
        pass_reward=1000.0,
        fail_penalty=1000.0,
        bet_limit_penalty=20000.0,
        completion_reward_multiplier=1.0,
        bet_rate_bonus=0.0,
    )


def build_risk_control_objective(settings: DqnTrainSettings) -> GameObjectiveConfig:
    return DqnRiskGameObjectiveConfig(
        enabled=True,
        max_steps=settings.risk_max_steps,
        pass_balance_multiplier=settings.risk_pass_balance / settings.initial_balance,
        fail_balance_multiplier=0.0,
        target_bet_rate=settings.risk_target_bet_rate,
        pass_reward=0.0,
        fail_penalty=0.0,
        bet_limit_penalty=500.0,
        completion_reward_multiplier=0.0,
        bet_rate_bonus=0.0,
        explicit_max_bets=settings.risk_max_bets,
    )


def dqn_required_history_rows(
    objective: GameObjectiveConfig | None = None,
    *,
    lookback_windows: tuple[int, ...] = DQN_LOOKBACK_WINDOWS,
) -> int:
    objective = objective or build_dqn_game_objective()
    return max_lookback_window(lookback_windows) + objective.max_steps + 1


def inspect_dqn_history(
    history_source,
    *,
    label: str = "history",
    lookback_windows: tuple[int, ...] = DQN_LOOKBACK_WINDOWS,
) -> DqnHistoryPreflight:
    rows = load_history(history_source)
    objective = build_dqn_game_objective()
    max_lookback = max_lookback_window(lookback_windows)
    max_start = max(0, len(rows) - objective.max_steps - 1)
    return DqnHistoryPreflight(
        label=label,
        rows=len(rows),
        required_rows=dqn_required_history_rows(objective, lookback_windows=lookback_windows),
        max_lookback=max_lookback,
        max_steps=objective.max_steps,
        complete_start_min=max_lookback,
        complete_start_max=max_start,
        ok=max_start >= max_lookback,
    )


def preflight_dqn_training(
    history_source,
    *,
    eval_history_source=None,
    lookback_windows: tuple[int, ...] = DQN_LOOKBACK_WINDOWS,
) -> dict[str, Any]:
    reports = [inspect_dqn_history(history_source, label="train", lookback_windows=lookback_windows)]
    if eval_history_source is not None:
        reports.append(inspect_dqn_history(eval_history_source, label="eval", lookback_windows=lookback_windows))
    failures = [report for report in reports if not report.ok]
    if failures:
        details = [
            (
                f"{report.label} has {report.rows} rows; DQN requires at least "
                f"{report.required_rows} rows for lookback={report.max_lookback} "
                f"and max_steps={report.max_steps}"
            )
            for report in failures
        ]
        raise ValueError("DQN preflight failed:\n- " + "\n- ".join(details))
    return {report.label: asdict(report) for report in reports}


def build_dqn_env(
    history_source,
    *,
    settings: DqnTrainSettings,
    random_start: bool,
) -> DqnCapitalEnv:
    rows = load_history(history_source)
    risk_mode = settings.reward_objective == "risk"
    training_gate = (
        TrainingGateConfig(
            enabled=True,
            min_bet_rate=0.0,
            watch_penalty_start=0,
            high_stake_ratio=max(DQN_STAKE_RATIOS),
        )
        if risk_mode
        else TrainingGateConfig(enabled=False)
    )
    env = CasinoEnv(
        rows,
        initial_balance=settings.initial_balance,
        max_stake=settings.max_stake,
        min_stake=settings.min_stake,
        reward_mode="normalized",
        training_gate=training_gate,
        game_objective=build_risk_control_objective(settings) if risk_mode else build_dqn_game_objective(settings.max_steps),
    )
    return DqnCapitalEnv(
        env,
        random_start=random_start,
        wait_reward=settings.wait_reward,
        low_bet_rate_wait_penalty=settings.low_bet_rate_wait_penalty,
        lookback_windows=settings.lookback_windows,
        reward_objective=settings.reward_objective,
        risk_target_bets=settings.risk_target_bets,
        risk_max_bets=settings.risk_max_bets,
        risk_min_cashout_bets=settings.risk_min_cashout_bets,
        risk_pass_balance=settings.risk_pass_balance,
        risk_cashout_balance=settings.risk_cashout_balance,
    )


def build_dqn_kwargs(settings: DqnTrainSettings) -> dict[str, Any]:
    return {
        "device": resolve_device(settings.device),
        "seed": settings.seed,
        "learning_rate": settings.learning_rate,
        "buffer_size": settings.buffer_size,
        "learning_starts": settings.learning_starts,
        "batch_size": settings.batch_size,
        "gamma": settings.gamma,
        "train_freq": settings.train_freq,
        "gradient_steps": settings.gradient_steps,
        "target_update_interval": settings.target_update_interval,
        "exploration_fraction": settings.exploration_fraction,
        "exploration_initial_eps": settings.exploration_initial_eps,
        "exploration_final_eps": settings.exploration_final_eps,
        "max_grad_norm": settings.max_grad_norm,
        "policy_kwargs": {"net_arch": DQN_NET_ARCH},
        "verbose": settings.verbose,
    }


def metadata_path_for(model_path: str | Path) -> Path:
    path = resolve_model_path(model_path)
    return path.with_name(f"{path.stem}.metadata.json")


def write_training_metadata(settings: DqnTrainSettings, *, history_rows: int, resolved_device: str) -> Path:
    metadata_path = metadata_path_for(settings.model_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_path": str(settings.model_path),
        "history_rows": history_rows,
        "device": resolved_device,
        "feature_columns": [
            *FEATURE_COLUMNS,
            "wealth_ratio",
            "max_drawdown",
            "remaining_bet_ratio",
            "bet_rate",
            "loss_streak_ratio",
            "period_ratio",
            *dqn_lookback_features(settings.lookback_windows),
        ],
        "action_space": f"Discrete({len(DQN_ACTION_SPECS)})",
        "actions": [
            {
                **asdict(spec),
                "direction_labels": [DIRECTION_LABELS[direction] for direction in spec.directions],
            }
            for spec in DQN_ACTION_SPECS
        ],
        "plays": {str(key): asdict(value) for key, value in BET_SPECS.items() if value.label in DQN_PLAY_LABELS},
        "game_objective": asdict(build_dqn_game_objective(settings.max_steps)),
        "lookback_windows": list(settings.lookback_windows),
        "required_history_rows": dqn_required_history_rows(lookback_windows=settings.lookback_windows),
        "settings": {**asdict(settings), "model_path": str(settings.model_path)},
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def train_model(
    history_source,
    *,
    model_path: str | Path = DQN_MODEL_PATH,
    total_timesteps: int = 100_000,
    initial_balance: float = 1000.0,
    max_stake: float = 100.0,
    min_stake: float = DQN_MIN_STAKE,
    max_steps: int = DQN_MAX_STEPS,
    check: bool = True,
    device: str = "auto",
    seed: int = 42,
    verbose: int = 1,
    resume: bool = False,
    wait_reward: float = 0.0,
    low_bet_rate_wait_penalty: float = 1.0,
    lookback_windows: tuple[int, ...] = DQN_LOOKBACK_WINDOWS,
    reward_objective: str = "capital",
    risk_target_bets: int = DQN_RISK_TARGET_BETS,
    risk_max_bets: int = DQN_RISK_MAX_BETS,
    risk_min_cashout_bets: int = DQN_RISK_MIN_CASHOUT_BETS,
    risk_max_steps: int = DQN_RISK_MAX_STEPS,
    risk_target_bet_rate: float = DQN_RISK_TARGET_BET_RATE,
    risk_pass_balance: float = DQN_RISK_PASS_BALANCE,
    risk_cashout_balance: float = DQN_RISK_CASHOUT_BALANCE,
    combo: bool = False,
) -> Path:
    if combo:
        from .train_combo import train_dqn_combo

        target = Path(model_path)
        if target == Path(DQN_MODEL_PATH):
            target = MODEL_DIR / "dqn_combo_balance_v1.zip"
        return train_dqn_combo(
            history_source,
            model_path=target,
            total_timesteps=total_timesteps,
            initial_balance=initial_balance,
            seed=seed,
            device=device,
            check=check,
        )
    settings = DqnTrainSettings(
        model_path=Path(model_path),
        total_timesteps=total_timesteps,
        initial_balance=initial_balance,
        max_stake=max_stake,
        min_stake=min_stake,
        max_steps=max_steps,
        check_env=check,
        device=device,
        seed=seed,
        verbose=verbose,
        resume=resume,
        wait_reward=wait_reward,
        low_bet_rate_wait_penalty=low_bet_rate_wait_penalty,
        lookback_windows=lookback_windows,
        reward_objective=reward_objective,
        risk_target_bets=risk_target_bets,
        risk_max_bets=risk_max_bets,
        risk_min_cashout_bets=risk_min_cashout_bets,
        risk_max_steps=risk_max_steps,
        risk_target_bet_rate=risk_target_bet_rate,
        risk_pass_balance=risk_pass_balance,
        risk_cashout_balance=risk_cashout_balance,
    )
    preflight_dqn_training(history_source, lookback_windows=settings.lookback_windows)
    env = build_dqn_env(history_source, settings=settings, random_start=True)
    if settings.check_env:
        check_env(env, warn=True)

    resolved_device = resolve_device(settings.device)
    set_random_seed(settings.seed)
    if settings.resume:
        if not settings.model_path.exists():
            raise FileNotFoundError(f"cannot resume missing model: {settings.model_path}")
        model = DQN.load(settings.model_path, env=env, device=resolved_device)
    else:
        model = DQN("MlpPolicy", env, **build_dqn_kwargs(settings))
    model.learn(total_timesteps=settings.total_timesteps)

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(settings.model_path)
    write_training_metadata(settings, history_rows=len(env.env.history), resolved_device=resolved_device)
    return settings.model_path


def evaluate_model(
    *,
    model_path: str | Path,
    history_source,
    episodes: int = 20,
    initial_balance: float = 1000.0,
    max_stake: float = 100.0,
    min_stake: float = DQN_MIN_STAKE,
    max_steps: int = DQN_MAX_STEPS,
    device: str = "auto",
    wait_reward: float = 0.0,
    low_bet_rate_wait_penalty: float = 1.0,
    lookback_windows: tuple[int, ...] = DQN_LOOKBACK_WINDOWS,
    reward_objective: str = "capital",
    risk_target_bets: int = DQN_RISK_TARGET_BETS,
    risk_max_bets: int = DQN_RISK_MAX_BETS,
    risk_min_cashout_bets: int = DQN_RISK_MIN_CASHOUT_BETS,
    risk_max_steps: int = DQN_RISK_MAX_STEPS,
    risk_target_bet_rate: float = DQN_RISK_TARGET_BET_RATE,
    risk_pass_balance: float = DQN_RISK_PASS_BALANCE,
    risk_cashout_balance: float = DQN_RISK_CASHOUT_BALANCE,
    output_path: str | Path = LOG_DIR / "dqn_capital_game_v1_evaluation.json",
) -> dict[str, Any]:
    settings = DqnTrainSettings(
        model_path=Path(model_path),
        initial_balance=initial_balance,
        max_stake=max_stake,
        min_stake=min_stake,
        max_steps=max_steps,
        device=device,
        wait_reward=wait_reward,
        low_bet_rate_wait_penalty=low_bet_rate_wait_penalty,
        lookback_windows=lookback_windows,
        reward_objective=reward_objective,
        risk_target_bets=risk_target_bets,
        risk_max_bets=risk_max_bets,
        risk_min_cashout_bets=risk_min_cashout_bets,
        risk_max_steps=risk_max_steps,
        risk_target_bet_rate=risk_target_bet_rate,
        risk_pass_balance=risk_pass_balance,
        risk_cashout_balance=risk_cashout_balance,
        check_env=False,
        verbose=0,
    )
    preflight_dqn_training(history_source, lookback_windows=settings.lookback_windows)
    env = build_dqn_env(history_source, settings=settings, random_start=False)
    model = DQN.load(resolve_model_path(model_path), env=env, device=resolve_device(device))
    starts = _episode_starts(
        len(env.env.history),
        episodes,
        build_risk_control_objective(settings).max_steps if reward_objective == "risk" else build_dqn_game_objective(settings.max_steps).max_steps,
        min_start=max_lookback_window(settings.lookback_windows),
    )
    episode_reports = [_evaluate_episode(model, env, start) for start in starts]
    summary = _summarize_episodes(episode_reports)
    report = {
        "model_path": str(resolve_model_path(model_path)),
        "history_source": str(resolve_history_path(history_source)) if isinstance(history_source, (str, Path)) else "memory",
        "episodes": episode_reports,
        "summary": summary,
    }
    output = resolve_log_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def evaluate_walk_forward_model(
    *,
    model_path: str | Path,
    history_source,
    initial_balance: float = 1000.0,
    max_stake: float = 100.0,
    min_stake: float = DQN_MIN_STAKE,
    max_steps: int = DQN_MAX_STEPS,
    device: str = "auto",
    wait_reward: float = 0.0,
    low_bet_rate_wait_penalty: float = 1.0,
    lookback_windows: tuple[int, ...] = DQN_LOOKBACK_WINDOWS,
    reward_objective: str = "capital",
    risk_target_bets: int = DQN_RISK_TARGET_BETS,
    risk_max_bets: int = DQN_RISK_MAX_BETS,
    risk_min_cashout_bets: int = DQN_RISK_MIN_CASHOUT_BETS,
    risk_max_steps: int = DQN_RISK_MAX_STEPS,
    risk_target_bet_rate: float = DQN_RISK_TARGET_BET_RATE,
    risk_pass_balance: float = DQN_RISK_PASS_BALANCE,
    risk_cashout_balance: float = DQN_RISK_CASHOUT_BALANCE,
    start_step: int | None = None,
    output_path: str | Path = LOG_DIR / "dqn_capital_game_v1_evaluation.json",
) -> dict[str, Any]:
    settings = DqnTrainSettings(
        model_path=Path(model_path),
        initial_balance=initial_balance,
        max_stake=max_stake,
        min_stake=min_stake,
        max_steps=max_steps,
        device=device,
        wait_reward=wait_reward,
        low_bet_rate_wait_penalty=low_bet_rate_wait_penalty,
        lookback_windows=lookback_windows,
        reward_objective=reward_objective,
        risk_target_bets=risk_target_bets,
        risk_max_bets=risk_max_bets,
        risk_min_cashout_bets=risk_min_cashout_bets,
        risk_max_steps=risk_max_steps,
        risk_target_bet_rate=risk_target_bet_rate,
        risk_pass_balance=risk_pass_balance,
        risk_cashout_balance=risk_cashout_balance,
        check_env=False,
        verbose=0,
    )
    preflight_dqn_training(history_source, lookback_windows=settings.lookback_windows)
    env = build_dqn_env(history_source, settings=settings, random_start=False)
    objective = build_risk_control_objective(settings) if reward_objective == "risk" else build_dqn_game_objective(settings.max_steps)
    model = DQN.load(resolve_model_path(model_path), env=env, device=resolve_device(device))
    start = _single_reference_window_start(
        len(env.env.history),
        max_steps=objective.max_steps,
        lookback=max_lookback_window(settings.lookback_windows),
        start_step=start_step,
    )
    episode_reports = [_evaluate_episode(model, env, start)]
    summary = _summarize_episodes(episode_reports)
    report = {
        "mode": "single_reference_window",
        "model_path": str(resolve_model_path(model_path)),
        "history_source": str(resolve_history_path(history_source)) if isinstance(history_source, (str, Path)) else "memory",
        "lookback": max_lookback_window(settings.lookback_windows),
        "max_steps": objective.max_steps,
        "start_step": start,
        "context_start": start - max_lookback_window(settings.lookback_windows),
        "context_end": start - 1,
        "eval_start": start,
        "eval_end": start + objective.max_steps - 1,
        "episodes": episode_reports,
        "summary": summary,
    }
    output = resolve_log_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _episode_starts(history_rows: int, episodes: int, max_steps: int, *, min_start: int = 0) -> list[int]:
    max_start = max(0, history_rows - max_steps - 1)
    if max_start < min_start:
        return [0]
    if episodes <= 1 or max_start == 0:
        return [min_start]
    return [int(round(value)) for value in np.linspace(min_start, max_start, episodes)]


def _single_reference_window_start(
    history_rows: int,
    *,
    max_steps: int,
    lookback: int,
    start_step: int | None = None,
) -> int:
    max_start = history_rows - max_steps
    if max_start < lookback:
        raise ValueError(f"history requires at least {lookback + max_steps} rows for one reference window")
    start = lookback if start_step is None else int(start_step)
    if start < lookback:
        raise ValueError(f"start_step must be at least lookback={lookback}")
    if start > max_start:
        raise ValueError(f"start_step must be <= {max_start} for max_steps={max_steps}")
    return start


def _evaluate_episode(model: DQN, env: DqnCapitalEnv, start_step: int) -> dict[str, Any]:
    obs, info = env.reset(options={"start_step": start_step})
    terminated = False
    truncated = False
    steps = 0
    bets = 0
    wins = 0
    losses = 0
    wait_actions = 0
    history_reference_bets = 0
    history_bad_bets = 0
    history_win_rate_total = 0.0
    history_avg_pnl_total = 0.0
    history_max_loss_streak = 0
    risk_violation_steps = 0
    risky_bets = 0
    clean_bets = 0
    max_drawdown = 0.0
    final_info = dict(info)
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action_index = int(action)
        spec = env.action_specs[action_index]
        if spec.is_wait:
            wait_actions += 1
        else:
            reference = _selected_action_history_reference(env, spec)
            history_reference_bets += 1
            history_win_rate_total += reference["history_win_rate"]
            history_avg_pnl_total += reference["history_avg_pnl"]
            history_max_loss_streak = max(history_max_loss_streak, int(reference["history_max_loss_streak"]))
            if reference["history_avg_pnl"] <= 0.0:
                history_bad_bets += 1
        obs, reward, terminated, truncated, final_info = env.step(action_index)
        steps += 1
        bad_gate_reasons = env._bad_gate_reasons(final_info)
        if bad_gate_reasons:
            risk_violation_steps += 1
        if float(final_info.get("stake", 0.0)) > 0.0:
            bets += 1
            if bad_gate_reasons:
                risky_bets += 1
            else:
                clean_bets += 1
            pnl = float(final_info.get("profit_loss", 0.0))
            if pnl > 0.0:
                wins += 1
            elif pnl < 0.0:
                losses += 1
        max_drawdown = max(max_drawdown, float(final_info.get("drawdown", 0.0)))

    reason = str(final_info.get("game_objective_reason", "none"))
    return {
        "start_step": start_step,
        "steps": steps,
        "bets": bets,
        "wait_rate": round(wait_actions / max(steps, 1), 6),
        "bet_rate": round(bets / max(steps, 1), 6),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / max(bets, 1), 6),
        "history_reference_bets": history_reference_bets,
        "history_bad_bets": history_bad_bets,
        "history_bad_bet_rate": round(history_bad_bets / max(history_reference_bets, 1), 6),
        "history_win_rate_avg": round(history_win_rate_total / max(history_reference_bets, 1), 6),
        "history_avg_pnl": round(history_avg_pnl_total / max(history_reference_bets, 1), 6),
        "history_max_loss_streak": history_max_loss_streak,
        "risk_violation_steps": risk_violation_steps,
        "risky_bets": risky_bets,
        "clean_bets": clean_bets,
        "risky_bet_rate": round(risky_bets / max(bets, 1), 6),
        "clean_bet_rate": round(clean_bets / max(bets, 1), 6),
        "final_balance": float(final_info.get("balance", env.env.balance)),
        "pnl": round(float(final_info.get("balance", env.env.balance)) - env.env.initial_balance, 6),
        "max_drawdown": round(max_drawdown, 6),
        "loss_streak": int(final_info.get("losing_streak", 0)),
        "result": _episode_result(reason),
        "reason": reason,
    }


def _selected_action_history_reference(env: DqnCapitalEnv, spec: DqnActionSpec, *, lookback: int = 1000) -> dict[str, float | int]:
    current = max(0, int(env.env.current_step))
    sample = env.env.history[max(0, current - lookback):current]
    if spec.is_wait or not sample:
        return {
            "history_samples": len(sample),
            "history_win_rate": 0.0,
            "history_avg_pnl": 0.0,
            "history_max_loss_streak": 0,
        }

    wins = 0
    pnl_total = 0.0
    current_loss_streak = 0
    max_loss_streak = 0
    if spec.stake_amount is not None:
        stake_per_play = min(env.env.max_stake, max(env.env.min_stake, float(spec.stake_amount)))
    else:
        stake_per_play = min(env.env.max_stake, max(env.env.min_stake, env.env.balance * float(spec.stake_ratio or 0.0)))
    for row in sample:
        pnl = sum(
            env.env._settle_bet(direction, stake_per_play, row, 0.0).profit_loss
            for direction in spec.directions
        )
        pnl_total += pnl
        if pnl > 0.0:
            wins += 1
            current_loss_streak = 0
        elif pnl < 0.0:
            current_loss_streak += 1
            max_loss_streak = max(max_loss_streak, current_loss_streak)
    return {
        "history_samples": len(sample),
        "history_win_rate": round(wins / len(sample), 6),
        "history_avg_pnl": round(pnl_total / len(sample), 6),
        "history_max_loss_streak": max_loss_streak,
    }


def _episode_result(reason: str) -> str:
    if reason == "game_pass":
        return "pass"
    if reason == "risk_cashout":
        return "cashout"
    if reason == "risk_cashout_sprint_failed":
        return "cashout_sprint_failed"
    if reason == "game_bet_limit":
        return "violation"
    if reason in {
        "risk_under_target_bets",
        "risk_over_max_bets",
        "risk_cashout_under_min_bets",
        "risk_balance_floor_breach",
    }:
        return "violation"
    if reason == "game_fail":
        return "fail"
    return "complete"


def _summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    count = max(len(episodes), 1)
    return {
        "episodes": len(episodes),
        "pass_rate": round(sum(1 for item in episodes if item["result"] == "pass") / count, 6),
        "cashout_rate": round(sum(1 for item in episodes if item["result"] == "cashout") / count, 6),
        "cashout_sprint_failed_rate": round(
            sum(1 for item in episodes if item["result"] == "cashout_sprint_failed") / count,
            6,
        ),
        "fail_rate": round(sum(1 for item in episodes if item["result"] == "fail") / count, 6),
        "violation_rate": round(sum(1 for item in episodes if item["result"] == "violation") / count, 6),
        "avg_bets": round(sum(float(item["bets"]) for item in episodes) / count, 6),
        "avg_bet_rate": round(sum(float(item["bet_rate"]) for item in episodes) / count, 6),
        "avg_wait_rate": round(sum(float(item["wait_rate"]) for item in episodes) / count, 6),
        "history_reference_bets": sum(int(item.get("history_reference_bets", 0)) for item in episodes),
        "history_bad_bets": sum(int(item.get("history_bad_bets", 0)) for item in episodes),
        "history_bad_bet_rate": round(
            sum(int(item.get("history_bad_bets", 0)) for item in episodes)
            / max(sum(float(item["bets"]) for item in episodes), 1.0),
            6,
        ),
        "history_win_rate_avg": round(
            sum(float(item.get("history_win_rate_avg", 0.0)) for item in episodes) / count,
            6,
        ),
        "history_avg_pnl": round(
            sum(float(item.get("history_avg_pnl", 0.0)) for item in episodes) / count,
            6,
        ),
        "history_max_loss_streak": max((int(item.get("history_max_loss_streak", 0)) for item in episodes), default=0),
        "risk_violation_steps": sum(int(item.get("risk_violation_steps", 0)) for item in episodes),
        "risky_bets": sum(int(item.get("risky_bets", 0)) for item in episodes),
        "clean_bets": sum(int(item.get("clean_bets", 0)) for item in episodes),
        "risky_bet_rate": round(
            sum(int(item.get("risky_bets", 0)) for item in episodes)
            / max(sum(float(item["bets"]) for item in episodes), 1.0),
            6,
        ),
        "clean_bet_rate": round(
            sum(int(item.get("clean_bets", 0)) for item in episodes)
            / max(sum(float(item["bets"]) for item in episodes), 1.0),
            6,
        ),
        "bet_win_rate": round(
            sum(float(item["wins"]) for item in episodes) / max(sum(float(item["bets"]) for item in episodes), 1.0),
            6,
        ),
        "avg_pnl": round(sum(float(item["pnl"]) for item in episodes) / count, 6),
        "avg_final_balance": round(sum(float(item["final_balance"]) for item in episodes) / count, 6),
        "max_drawdown": round(max((float(item["max_drawdown"]) for item in episodes), default=0.0), 6),
        "max_loss_streak": max((int(item["loss_streak"]) for item in episodes), default=0),
    }


def format_summary_table(report: dict[str, Any]) -> str:
    summary = report["summary"]
    rows = [
        ["episodes", str(summary["episodes"])],
        ["pass_rate", f"{summary['pass_rate'] * 100:.2f}%"],
        ["cashout_rate", f"{summary.get('cashout_rate', 0.0) * 100:.2f}%"],
        ["cashout_sprint_failed_rate", f"{summary.get('cashout_sprint_failed_rate', 0.0) * 100:.2f}%"],
        ["violation_rate", f"{summary['violation_rate'] * 100:.2f}%"],
        ["avg_bets", f"{summary['avg_bets']:.2f}"],
        ["avg_bet_rate", f"{summary['avg_bet_rate'] * 100:.2f}%"],
        ["avg_wait_rate", f"{summary['avg_wait_rate'] * 100:.2f}%"],
        ["risk_violation_steps", str(summary.get("risk_violation_steps", 0))],
        ["risky_bets", str(summary.get("risky_bets", 0))],
        ["risky_bet_rate", f"{summary.get('risky_bet_rate', 0.0) * 100:.2f}%"],
        ["clean_bets", str(summary.get("clean_bets", 0))],
        ["clean_bet_rate", f"{summary.get('clean_bet_rate', 0.0) * 100:.2f}%"],
        ["history_bad_bet_rate", f"{summary.get('history_bad_bet_rate', 0.0) * 100:.2f}%"],
        ["history_win_rate_avg", f"{summary.get('history_win_rate_avg', 0.0) * 100:.2f}%"],
        ["history_avg_pnl", f"{summary.get('history_avg_pnl', 0.0):.2f}"],
        ["history_max_loss_streak", str(summary.get("history_max_loss_streak", 0))],
        ["bet_win_rate", f"{summary['bet_win_rate'] * 100:.2f}%"],
        ["avg_final_balance", f"{summary['avg_final_balance']:.2f}"],
        ["avg_pnl", f"{summary['avg_pnl']:.2f}"],
        ["max_drawdown", f"{summary['max_drawdown'] * 100:.2f}%"],
        ["max_loss_streak", str(summary["max_loss_streak"])],
    ]
    return "\n".join(_format_table(["指标", "DQN"], rows, right_align={1}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a discrete DQN capital-game model.")
    parser.add_argument("history", help="CSV/JSON/JSONL training history file")
    parser.add_argument("--eval-history", help="Optional CSV/JSON/JSONL evaluation history file")
    parser.add_argument("--model", default=str(DQN_MODEL_PATH))
    parser.add_argument("--combo", action="store_true", help="Train the PC28 three-base-market combo schema.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--initial-balance", type=float, default=1000.0)
    parser.add_argument("--max-stake", type=float, default=100.0)
    parser.add_argument("--min-stake", type=float, default=DQN_MIN_STAKE)
    parser.add_argument("--max-steps", type=int, default=DQN_MAX_STEPS)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-reward", type=float, default=0.0)
    parser.add_argument("--low-bet-rate-wait-penalty", type=float, default=1.0)
    parser.add_argument("--reward-objective", choices=DQN_REWARD_OBJECTIVES, default="capital")
    parser.add_argument("--risk-target-bets", type=int, default=DQN_RISK_TARGET_BETS)
    parser.add_argument("--risk-max-bets", type=int, default=DQN_RISK_MAX_BETS)
    parser.add_argument("--risk-min-cashout-bets", type=int, default=DQN_RISK_MIN_CASHOUT_BETS)
    parser.add_argument("--risk-max-steps", type=int, default=DQN_RISK_MAX_STEPS)
    parser.add_argument("--risk-target-bet-rate", type=float, default=DQN_RISK_TARGET_BET_RATE)
    parser.add_argument("--risk-pass-balance", type=float, default=DQN_RISK_PASS_BALANCE)
    parser.add_argument("--risk-cashout-balance", type=float, default=DQN_RISK_CASHOUT_BALANCE)
    parser.add_argument(
        "--lookback-windows",
        default=None,
        help="Comma-separated DQN lookback windows; pass an empty string to train on short data without lookback features.",
    )
    parser.add_argument("--skip-check-env", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="Validate DQN history windows without training.")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate an existing DQN model without training.")
    parser.add_argument("--walk-forward-eval", action="store_true", help="Evaluate one 1000-context + 200-step DQN reference window.")
    parser.add_argument("--start-step", type=int, default=None, help="Evaluation window start; defaults to the first full-lookback step.")
    parser.add_argument("--output", default=str(LOG_DIR / "dqn_capital_game_v1_evaluation.json"))
    parser.add_argument("--table-output", default=str(LOG_DIR / "dqn_capital_game_v1_evaluation.md"))
    args = parser.parse_args()
    try:
        lookback_windows = parse_lookback_windows(args.lookback_windows)
    except ValueError as exc:
        parser.error(str(exc))

    if args.combo:
        if args.eval_only or args.walk_forward_eval or args.preflight_only:
            parser.error("combo CLI currently supports training only; use ai_agent.backtest_combo and ai_agent.infer_combo for evaluation")
        output_model = train_model(
            args.history,
            model_path=args.model,
            total_timesteps=args.timesteps,
            initial_balance=args.initial_balance,
            check=not args.skip_check_env,
            device=args.device,
            seed=args.seed,
            combo=True,
        )
        print(f"saved combo model: {output_model}")
        print(f"saved combo metadata: {metadata_path_for(output_model)}")
        return

    try:
        preflight = preflight_dqn_training(
            args.history,
            eval_history_source=args.eval_history or args.history,
            lookback_windows=lookback_windows,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.eval_only:
        output_model = resolve_model_path(args.model)
    else:
        output_model = train_model(
            args.history,
            model_path=args.model,
            total_timesteps=args.timesteps,
            initial_balance=args.initial_balance,
            max_stake=args.max_stake,
            min_stake=args.min_stake,
            max_steps=args.max_steps,
            check=not args.skip_check_env,
            device=args.device,
            seed=args.seed,
            verbose=args.verbose,
            resume=args.resume,
            wait_reward=args.wait_reward,
            low_bet_rate_wait_penalty=args.low_bet_rate_wait_penalty,
            lookback_windows=lookback_windows,
            reward_objective=args.reward_objective,
            risk_target_bets=args.risk_target_bets,
            risk_max_bets=args.risk_max_bets,
            risk_min_cashout_bets=args.risk_min_cashout_bets,
            risk_max_steps=args.risk_max_steps,
            risk_target_bet_rate=args.risk_target_bet_rate,
            risk_pass_balance=args.risk_pass_balance,
            risk_cashout_balance=args.risk_cashout_balance,
            combo=args.combo,
        )
    if args.walk_forward_eval:
        report = evaluate_walk_forward_model(
            model_path=output_model,
            history_source=args.eval_history or args.history,
            initial_balance=args.initial_balance,
            max_stake=args.max_stake,
            min_stake=args.min_stake,
            max_steps=args.max_steps,
            device=args.device,
            wait_reward=args.wait_reward,
            low_bet_rate_wait_penalty=args.low_bet_rate_wait_penalty,
            lookback_windows=lookback_windows,
            reward_objective=args.reward_objective,
            risk_target_bets=args.risk_target_bets,
            risk_max_bets=args.risk_max_bets,
            risk_min_cashout_bets=args.risk_min_cashout_bets,
            risk_max_steps=args.risk_max_steps,
            risk_target_bet_rate=args.risk_target_bet_rate,
            risk_pass_balance=args.risk_pass_balance,
            risk_cashout_balance=args.risk_cashout_balance,
            start_step=args.start_step,
            output_path=args.output,
        )
    else:
        report = evaluate_model(
            model_path=output_model,
            history_source=args.eval_history or args.history,
            episodes=args.episodes,
            initial_balance=args.initial_balance,
            max_stake=args.max_stake,
            min_stake=args.min_stake,
            max_steps=args.max_steps,
            device=args.device,
            wait_reward=args.wait_reward,
            low_bet_rate_wait_penalty=args.low_bet_rate_wait_penalty,
            lookback_windows=lookback_windows,
            reward_objective=args.reward_objective,
            risk_target_bets=args.risk_target_bets,
            risk_max_bets=args.risk_max_bets,
            risk_min_cashout_bets=args.risk_min_cashout_bets,
            risk_max_steps=args.risk_max_steps,
            risk_target_bet_rate=args.risk_target_bet_rate,
            risk_pass_balance=args.risk_pass_balance,
            risk_cashout_balance=args.risk_cashout_balance,
            output_path=args.output,
        )
    table = format_summary_table(report)
    table_path = resolve_log_path(args.table_output)
    table_path.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"saved model: {output_model}")
    print(f"saved metadata: {metadata_path_for(output_model)}")
    print(f"saved evaluation: {resolve_log_path(args.output)}")
    print(f"saved table: {table_path}")


if __name__ == "__main__":
    main()
