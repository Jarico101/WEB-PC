"""Gymnasium environment for the PC28 three-base-market combo contract."""

from __future__ import annotations

from typing import Any, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data_adapter import COMBO_FEATURE_COLUMNS, clean_combo_history
from .markets_combo import ExtremeRanges, ZeroPolicy
from .portfolio_combo import BASE_COMBINATIONS, BASE_ONLY_EXTENSION_PAIRS, MODE_IDS, TRAINING_MODE_IDS, PortfolioError, PortfolioLimits, WAIT_BASE_ACTION_ID, action_nvec, build_decision, extension_pairs
from .reward import ComboRewardConfig, calculate_combo_reward
from .settlement_combo import settle_combo
from .probability_combo import ComboProbabilityModel
from .statistical_combo_policy import OpportunityGateConfig, gate_model_action


def extension_sets(
    *,
    extreme_ranges: ExtremeRanges | None,
    extensions_enabled: bool = False,
) -> tuple[tuple[str, ...], ...]:
    """Compatibility facade for the canonical portfolio action enumeration."""

    return extension_pairs(extreme_ranges=extreme_ranges, extensions_enabled=extensions_enabled)


class ComboCasinoEnv(gym.Env):
    """Permits three base markets and, only when enabled, optional extensions."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        history: list[dict[str, Any]],
        *,
        initial_balance: float = 10_000.0,
        zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
        extreme_ranges: ExtremeRanges | None = None,
        portfolio_limits: PortfolioLimits = PortfolioLimits(),
        reward_config: ComboRewardConfig = ComboRewardConfig(),
        allow_shadow_mode: bool = False,
        allow_extensions: bool = False,
        extension_options: tuple[tuple[str, ...], ...] | None = None,
        probability_model: ComboProbabilityModel | None = None,
        probability_validation: dict[str, Any] | None = None,
        gate_config: OpportunityGateConfig = OpportunityGateConfig(),
        random_start_on_reset: bool = False,
        episode_horizon: int | None = None,
        include_previous_combo_observation: bool = False,
    ) -> None:
        super().__init__()
        if initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        self.history = clean_combo_history(history, zero_policy=zero_policy, min_rows=1)
        self.initial_balance = float(initial_balance)
        self.zero_policy = zero_policy
        self.extreme_ranges = extreme_ranges
        self.portfolio_limits = portfolio_limits
        self.reward_config = reward_config
        self.allow_shadow_mode = allow_shadow_mode
        self.allow_extensions = allow_extensions
        self.mode_options = MODE_IDS if allow_shadow_mode else TRAINING_MODE_IDS
        self.probability_model = probability_model
        self.probability_validation = probability_validation or {}
        self.gate_config = gate_config
        if episode_horizon is not None and episode_horizon <= 0:
            raise ValueError("episode_horizon must be positive when configured")
        self.random_start_on_reset = random_start_on_reset
        self.episode_horizon = episode_horizon
        self.include_previous_combo_observation = include_previous_combo_observation
        default_extension_sets = extension_pairs(
            extreme_ranges=extreme_ranges,
            extensions_enabled=allow_extensions,
        )
        self.extension_sets = extension_options if extension_options is not None else default_extension_sets
        if not self.extension_sets or self.extension_sets[0] != ():
            raise ValueError("extension_options must start with the empty extension selection")
        if extension_options is not None and not allow_extensions:
            raise ValueError("custom extension_options require explicit allow_extensions=True")
        self.action_space = spaces.MultiDiscrete((
            action_nvec(
                extreme_ranges=extreme_ranges,
                include_shadow=allow_shadow_mode,
                extensions_enabled=allow_extensions,
            )[0],
            len(self.extension_sets),
            len(TRAINING_MODE_IDS),
            len(self.mode_options),
        ))
        # Prior statistics, capital ratio, drawdown, loss streak and, for
        # explicitly enabled diversity experiments, the prior base-combo ID.
        observation_context_size = 1 if include_previous_combo_observation else 0
        self.observation_space = spaces.Box(
            low=np.zeros(len(COMBO_FEATURE_COLUMNS) + 3 + observation_context_size, dtype=np.float32),
            high=np.array([1.0] * len(COMBO_FEATURE_COLUMNS) + [10.0, 1.0, 1.0] + [1.0] * observation_context_size, dtype=np.float32),
            dtype=np.float32,
        )
        self.current_step = 0
        self.episode_start_step = 0
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.losing_streak = 0
        self.max_losing_streak = 0
        self.cooldown_remaining = 0
        self.executed_orders = 0
        self.last_executed_base_id: int | None = None
        self.last_info: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        supplied_start_step = (options or {}).get("start_step")
        if supplied_start_step is not None:
            start_step = int(supplied_start_step)
        elif self.random_start_on_reset:
            latest_start = max(0, len(self.history) - (self.episode_horizon or 1))
            start_step = int(self.np_random.integers(0, latest_start + 1))
        else:
            start_step = 0
        self.current_step = min(max(start_step, 0), len(self.history) - 1)
        self.episode_start_step = self.current_step
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.losing_streak = 0
        self.max_losing_streak = 0
        self.cooldown_remaining = 0
        self.executed_orders = 0
        self.last_executed_base_id = None
        self.last_info = {"balance": self.balance, "step": self.current_step}
        self.last_info["features"] = dict(self.history[self.current_step]["features"])
        return self._get_obs(), dict(self.last_info)

    def step(self, action: Sequence[int]):
        action_values = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_values.size != 4:
            raise ValueError("combo action must have four axes")
        action_values = np.minimum(np.maximum(action_values, 0), self.action_space.nvec - 1)
        base_id, extension_set_id, profile_id, mode_id = (int(value) for value in action_values)
        row = self.history[self.current_step]
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            if self.cooldown_remaining == 0:
                self.losing_streak = 0
            return self._wait_step(
                row,
                np.asarray([WAIT_BASE_ACTION_ID, 0, 0, 0], dtype=np.int64),
                gate_reason="losing_streak_cooldown",
            )
        if base_id == WAIT_BASE_ACTION_ID:
            # MultiDiscrete emits every axis together.  Once the base axis says
            # wait, the remaining axes are intentionally ignored: no order can
            # be created from them.
            return self._wait_step(row, np.asarray([WAIT_BASE_ACTION_ID, 0, 0, 0], dtype=np.int64))
        gate_info = None
        if self.probability_model is not None and not self.allow_shadow_mode:
            gate_info = gate_model_action(
                action_values,
                self.probability_model,
                features=row["features"],
                balance=self.balance,
                drawdown=self._drawdown(),
                losing_streak=self.losing_streak,
                probability_validation=self.probability_validation,
                issue=row["issue"],
                as_of_issue=row["issue"],
                extreme_ranges=self.extreme_ranges,
                zero_policy=self.zero_policy,
                limits=self.portfolio_limits,
                config=self.gate_config,
                extension_options=self.extension_sets,
            )
            if gate_info.wait:
                return self._wait_step(
                    row,
                    np.asarray([WAIT_BASE_ACTION_ID, 0, 0, 0], dtype=np.int64),
                    gate_info=gate_info,
                )
        if self.gate_config.target_bet_rate is not None:
            observed_issues = self.current_step - self.episode_start_step + 1
            target_orders = int(np.ceil(observed_issues * self.gate_config.target_bet_rate))
            if self.executed_orders >= target_orders:
                return self._wait_step(
                    row,
                    np.asarray([WAIT_BASE_ACTION_ID, 0, 0, 0], dtype=np.int64),
                    gate_reason="target_bet_rate_limit",
                )
        # Use the exact frozen-model probabilities that approved the action.
        probabilities = gate_info.probabilities if gate_info is not None else self._probabilities()
        try:
            decision = build_decision(
                action=(base_id, extension_set_id, profile_id, mode_id),
                balance=self.balance,
                probabilities=probabilities,
                limits=self.portfolio_limits,
                issue=row["issue"],
                as_of_issue=row["issue"],
                extension_options=self.extension_sets,
                mode_options=self.mode_options,
            )
        except PortfolioError as exc:
            if str(exc) != "balance cannot fund the minimum legal combo portfolio":
                raise
            return self._insufficient_balance_step(row, action_values)
        settlement = settle_combo(
            decision,
            outcome_sum=int(row["sum"]),
            balance_before=self.balance,
            zero_policy=self.zero_policy,
            extreme_ranges=self.extreme_ranges,
            limits=self.portfolio_limits,
        )
        self.balance = settlement.balance_after
        self.executed_orders += 1
        self.peak_balance = max(self.peak_balance, self.balance)
        self.losing_streak = self.losing_streak + 1 if settlement.balance_delta < 0 else 0
        self.max_losing_streak = max(self.max_losing_streak, self.losing_streak)
        if self.losing_streak >= self.gate_config.max_losing_streak:
            self.cooldown_remaining = self.gate_config.cooldown_rounds
        drawdown = self._drawdown()
        reward_breakdown = calculate_combo_reward(
            settlement,
            initial_balance=self.initial_balance,
            drawdown=drawdown,
            losing_streak=self.losing_streak,
            config=self.reward_config,
        )
        repeated_base_combo = self.last_executed_base_id == base_id
        repeat_combo_penalty = self.reward_config.repeat_base_combo_penalty if repeated_base_combo else 0.0
        state_target_base_id = self._state_target_base_combo(row)
        state_combo_aligned = base_id == state_target_base_id
        state_alignment_bonus = self.reward_config.state_combo_alignment_bonus if state_combo_aligned else 0.0
        reward = reward_breakdown.reward - repeat_combo_penalty + state_alignment_bonus
        self.last_executed_base_id = base_id
        self.current_step += 1
        terminated = self.balance <= 0.0
        truncated = self._episode_finished()
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated or truncated else self._get_obs()
        self.last_info = {
            **settlement.to_dict(),
            "issue": row["issue"],
            "features": dict(row["features"]),
            "step": self.current_step - 1,
            "balance": self.balance,
            "drawdown": drawdown,
            "losing_streak": self.losing_streak,
            "max_losing_streak": self.max_losing_streak,
            "cooldown_remaining": self.cooldown_remaining,
            "executed_orders": self.executed_orders,
            "action": action_values.tolist(),
            "order_executed": True,
            "termination_reason": None,
            "probability_gate": gate_info.to_dict() if gate_info is not None else None,
            "reward": reward,
            "repeated_base_combo": repeated_base_combo,
            "repeat_combo_penalty": repeat_combo_penalty,
            "state_target_base_combo": state_target_base_id,
            "state_combo_aligned": state_combo_aligned,
            "state_alignment_bonus": state_alignment_bonus,
            "reward_breakdown": {
                "realized_pnl_component": reward_breakdown.realized_pnl_component,
                "shadow_pnl_component": reward_breakdown.shadow_pnl_component,
                "drawdown_penalty": reward_breakdown.drawdown_penalty,
                "losing_streak_penalty": reward_breakdown.losing_streak_penalty,
                "calibration_penalty": reward_breakdown.calibration_penalty,
                "calibration_error": reward_breakdown.calibration_error,
            },
        }
        return obs, float(reward), bool(terminated), bool(truncated), dict(self.last_info)

    def _wait_step(
        self,
        row: dict[str, Any],
        action_values: np.ndarray,
        *,
        gate_reason: str | None = None,
        gate_info: Any | None = None,
    ):
        """Advance one issue without creating a virtual order or settlement."""

        self.current_step += 1
        truncated = self._episode_finished()
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if truncated else self._get_obs()
        self.last_info = {
            "issue": row["issue"],
            "features": dict(row["features"]),
            "step": self.current_step - 1,
            "balance": self.balance,
            "balance_before": self.balance,
            "balance_after": self.balance,
            "drawdown": self._drawdown(),
            "losing_streak": self.losing_streak,
            "max_losing_streak": self.max_losing_streak,
            "cooldown_remaining": self.cooldown_remaining,
            "executed_orders": self.executed_orders,
            "action": action_values.tolist(),
            "wait": True,
            "order_executed": False,
            "termination_reason": None,
            "wait_reason": gate_info.reason if gate_info is not None else (gate_reason or "model_requested_wait"),
            "probability_gate": gate_info.to_dict() if gate_info is not None else ({"wait": True, "reason": gate_reason} if gate_reason else None),
            "total_stake": 0.0,
            "total_return": 0.0,
            "net_profit": 0.0,
            "is_shadow": False,
            "reward": 0.0,
            "reward_breakdown": {
                "realized_pnl_component": 0.0,
                "shadow_pnl_component": 0.0,
                "drawdown_penalty": 0.0,
                "losing_streak_penalty": 0.0,
                "calibration_penalty": 0.0,
                "calibration_error": 0.0,
            },
        }
        return obs, 0.0, False, bool(truncated), dict(self.last_info)

    def _insufficient_balance_step(self, row: dict[str, Any], action_values: np.ndarray):
        """End the episode when the chosen legal portfolio cannot be funded."""

        reward = -1.0
        self.last_info = {
            "issue": row["issue"],
            "step": self.current_step,
            "balance": self.balance,
            "balance_before": self.balance,
            "balance_after": self.balance,
            "action": action_values.tolist(),
            "order_executed": False,
            "insufficient_balance": True,
            "termination_reason": "insufficient_balance",
            "total_stake": 0.0,
            "total_return": 0.0,
            "net_profit": 0.0,
            "is_shadow": False,
            "reward": reward,
            "reward_breakdown": {
                "realized_pnl_component": 0.0,
                "shadow_pnl_component": 0.0,
                "drawdown_penalty": 0.0,
                "losing_streak_penalty": 0.0,
                "calibration_penalty": 0.0,
                "calibration_error": 0.0,
            },
        }
        return np.zeros(self.observation_space.shape, dtype=np.float32), reward, True, False, dict(self.last_info)

    def _probabilities(self) -> dict[str, float]:
        row = self.history[self.current_step]
        features = row["features"]
        probabilities = {
            "small_odd": float(features["base_rate_small_odd_100"]),
            "small_even": float(features["base_rate_small_even_100"]),
            "big_odd": float(features["base_rate_big_odd_100"]),
            "big_even": float(features["base_rate_big_even_100"]),
        }
        probabilities.update(
            {
                f"sum_{total:02d}": float(features[f"sum_rate_{total:02d}_100"])
                for total in range(28)
            }
        )
        if self.extreme_ranges is not None:
            probabilities["extreme_small"] = sum(
                probabilities[f"sum_{total:02d}"]
                for total in range(self.extreme_ranges.small_min, self.extreme_ranges.small_max + 1)
            )
            probabilities["extreme_big"] = sum(
                probabilities[f"sum_{total:02d}"]
                for total in range(self.extreme_ranges.big_min, self.extreme_ranges.big_max + 1)
            )
        return probabilities

    def _get_obs(self) -> np.ndarray:
        features = self.history[self.current_step]["features"]
        values = [float(features[column]) for column in COMBO_FEATURE_COLUMNS]
        values.extend(
            [
                self.balance / max(self.initial_balance, 1e-9),
                self._drawdown(),
                min(self.losing_streak / 20.0, 1.0),
            ]
        )
        if self.include_previous_combo_observation:
            values.append(1.0 if self.last_executed_base_id is None else self.last_executed_base_id / 4.0)
        return np.clip(np.asarray(values, dtype=np.float32), self.observation_space.low, self.observation_space.high)

    @staticmethod
    def _state_target_base_combo(row: dict[str, Any]) -> int:
        """Choose the three highest prior base frequencies with deterministic ties.

        The target reads only this row's causal feature snapshot (the 100 prior
        issues), never the row's outcome.  It is an experimental behaviour
        curriculum, not a next-outcome prediction target.
        """

        features = row["features"]
        markets = ("small_odd", "small_even", "big_odd", "big_even")
        selected = {
            market
            for _rate, market in sorted(
                ((float(features[f"base_rate_{market}_100"]), market) for market in markets),
                key=lambda item: (-item[0], item[1]),
            )[:3]
        }
        return next(index for index, combo in enumerate(BASE_COMBINATIONS) if set(combo) == selected)

    def _episode_finished(self) -> bool:
        if self.current_step >= len(self.history):
            return True
        return self.episode_horizon is not None and self.current_step - self.episode_start_step >= self.episode_horizon

    def _drawdown(self) -> float:
        return max(0.0, (self.peak_balance - self.balance) / max(self.peak_balance, 1e-9))
