"""Training entry points for the PC28 combo action contract.

This module intentionally uses a new model namespace. Existing models trained
against ``CasinoEnv``'s legacy action IDs are not compatible with this schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed

from .combo_env import ComboCasinoEnv
from .config import MODEL_DIR
from .data_adapter import COMBO_FEATURE_COLUMNS, COMBO_FEATURE_VERSION, clean_combo_history, load_history
from .markets_combo import COMBO_ACTION_SPACE_VERSION, COMBO_ODDS_VERSION, COMBO_RULE_VERSION, CONFIRMED_EXTREME_RANGES, ExtremeRanges, ZeroPolicy, extreme_ranges_metadata
from .portfolio_combo import action_nvec
from .probability_combo import (
    ProbabilityValidationThresholds,
    fit_and_qualify_combo_probability_model,
)
from .reward import ComboRewardConfig
from .statistical_combo_policy import OpportunityGateConfig


QUALIFIED_OPPORTUNITY_PROTOCOL_VERSION = "pc28-qualified-opportunity-v1"


class ComboDqnEnv(gym.Wrapper):
    """A lossless discrete projection for DQN, whose native action space is Discrete."""

    def __init__(self, env: ComboCasinoEnv) -> None:
        super().__init__(env)
        self.nvec = tuple(int(value) for value in env.action_space.nvec)
        self.action_space = spaces.Discrete(int(np.prod(self.nvec)))
        self.observation_space = env.observation_space

    def step(self, action: int):
        action_index = int(action)
        if not self.action_space.contains(action_index):
            raise ValueError("combo DQN action index out of range")
        multi_action = np.asarray(np.unravel_index(action_index, self.nvec), dtype=np.int64)
        obs, reward, terminated, truncated, info = self.env.step(multi_action)
        return obs, reward, terminated, truncated, {**info, "dqn_action": action_index, "combo_action": multi_action.tolist()}


def combo_action_to_dqn_index(action: tuple[int, int, int, int], nvec: tuple[int, int, int, int]) -> int:
    return int(np.ravel_multi_index(action, nvec))


def dqn_index_to_combo_action(index: int, nvec: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return tuple(int(value) for value in np.unravel_index(index, nvec))


def metadata_path_for(model_path: str | Path) -> Path:
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}.metadata.json")


def train_ppo_combo(
    history_source: Any,
    *,
    model_path: str | Path = MODEL_DIR / "ppo_combo_balance_v1.zip",
    total_timesteps: int = 10_000,
    initial_balance: float = 10_000.0,
    seed: int = 42,
    device: str = "auto",
    check: bool = True,
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    extreme_ranges: ExtremeRanges | None = CONFIRMED_EXTREME_RANGES,
    hyperparameters: Mapping[str, Any] | None = None,
    checkpoint_callback: BaseCallback | None = None,
    experiment_metadata: Mapping[str, Any] | None = None,
    gate_config: OpportunityGateConfig = OpportunityGateConfig(),
    allow_extensions: bool = False,
    random_start_on_reset: bool = True,
    episode_horizon: int | None = None,
    reward_config: ComboRewardConfig = ComboRewardConfig(
        repeat_base_combo_penalty=0.05,
        state_combo_alignment_bonus=0.25,
    ),
    include_previous_combo_observation: bool = False,
    probability_thresholds: ProbabilityValidationThresholds = ProbabilityValidationThresholds(),
) -> Path:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    rows = load_history(history_source)
    probability_model, probability_validation = _training_probability_gate(
        rows,
        zero_policy=zero_policy,
        thresholds=probability_thresholds,
        gate_config=gate_config,
        experiment_metadata=experiment_metadata,
    )
    effective_episode_horizon = episode_horizon if episode_horizon is not None else min(200, len(rows))
    env = ComboCasinoEnv(
        rows,
        initial_balance=initial_balance,
        zero_policy=zero_policy,
        extreme_ranges=extreme_ranges,
        probability_model=probability_model,
        probability_validation=probability_validation,
        gate_config=gate_config,
        allow_extensions=allow_extensions,
        random_start_on_reset=random_start_on_reset,
        episode_horizon=effective_episode_horizon,
        reward_config=reward_config,
        include_previous_combo_observation=include_previous_combo_observation,
    )
    if check:
        check_env(env, warn=True)
    set_random_seed(seed)
    resolved_hyperparameters = _ppo_hyperparameters(len(env.history), hyperparameters)
    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        seed=seed,
        **resolved_hyperparameters,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    diversity = _policy_diversity_report(model, env)
    if not diversity["passes"]:
        raise ValueError(
            "PPO candidate rejected: action policy collapsed; "
            f"unique_base_combos={diversity['unique_base_combos']} "
            f"unique_actions={diversity['unique_actions']} "
            f"max_action_share={diversity['max_action_share']} "
            f"state_alignment_rate={diversity['state_alignment_rate']} "
            f"unique_state_aligned_base_combos={diversity['unique_state_aligned_base_combos']}"
        )
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    metadata = {
        "model_path": str(output),
        "algorithm": "PPO",
        "rule_version": COMBO_RULE_VERSION,
        "odds_version": COMBO_ODDS_VERSION,
        "action_space_version": COMBO_ACTION_SPACE_VERSION,
        "feature_version": COMBO_FEATURE_VERSION,
        "feature_columns": list(COMBO_FEATURE_COLUMNS),
        "action_space": {"type": "MultiDiscrete", "nvec": [int(value) for value in env.action_space.nvec]},
        "portfolio_action_nvec": list(action_nvec(extreme_ranges=extreme_ranges, extensions_enabled=allow_extensions)),
        "extreme_ranges": extreme_ranges_metadata(extreme_ranges),
        "extensions_enabled": allow_extensions,
        "history_rows": len(env.history),
        "training_data_end_issue": env.history[-1]["issue"],
        "zero_policy": zero_policy.value,
        "training_hyperparameters": resolved_hyperparameters,
        "experiment": dict(experiment_metadata or {}),
        "environment_config": {
            "random_start_on_reset": random_start_on_reset,
            "episode_horizon": effective_episode_horizon,
            "repeat_base_combo_penalty": reward_config.repeat_base_combo_penalty,
            "state_combo_alignment_bonus": reward_config.state_combo_alignment_bonus,
            "include_previous_combo_observation": include_previous_combo_observation,
            "action_diversity_regularization": {
                "repeat_base_combo_penalty": reward_config.repeat_base_combo_penalty,
                "state_combo_alignment_bonus": reward_config.state_combo_alignment_bonus,
                "minimum_unique_base_combos": 2,
                "minimum_unique_actions": 3,
                "maximum_single_action_share": 0.90,
                "minimum_state_alignment_rate": 0.40,
                "minimum_state_aligned_base_combos": 3,
            },
        },
        "policy_diversity": diversity,
        "training_protocol": QUALIFIED_OPPORTUNITY_PROTOCOL_VERSION,
        "probability_gate": {
            "enabled": True,
            "qualification": probability_validation,
            "require_probability_validation": gate_config.require_probability_validation,
            "risk_controls": {
                "min_expected_return_ratio": gate_config.min_expected_return_ratio,
                "min_base_coverage": gate_config.min_base_coverage,
                "max_drawdown": gate_config.max_drawdown,
                "max_losing_streak": gate_config.max_losing_streak,
                "cooldown_rounds": gate_config.cooldown_rounds,
                "target_bet_rate": gate_config.target_bet_rate,
                "allow_negative_expected_return": gate_config.allow_negative_expected_return,
            },
        },
    }
    metadata_path_for(output).write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output


def train_dqn_combo(
    history_source: Any,
    *,
    model_path: str | Path = MODEL_DIR / "dqn_combo_balance_v1.zip",
    total_timesteps: int = 10_000,
    initial_balance: float = 10_000.0,
    seed: int = 42,
    device: str = "auto",
    check: bool = True,
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    extreme_ranges: ExtremeRanges | None = CONFIRMED_EXTREME_RANGES,
    hyperparameters: Mapping[str, Any] | None = None,
    checkpoint_callback: BaseCallback | None = None,
    experiment_metadata: Mapping[str, Any] | None = None,
    gate_config: OpportunityGateConfig = OpportunityGateConfig(),
    allow_extensions: bool = False,
    probability_thresholds: ProbabilityValidationThresholds = ProbabilityValidationThresholds(),
) -> Path:
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    rows = load_history(history_source)
    probability_model, probability_validation = _training_probability_gate(
        rows,
        zero_policy=zero_policy,
        thresholds=probability_thresholds,
        gate_config=gate_config,
        experiment_metadata=experiment_metadata,
    )
    base_env = ComboCasinoEnv(
        rows,
        initial_balance=initial_balance,
        zero_policy=zero_policy,
        extreme_ranges=extreme_ranges,
        probability_model=probability_model,
        probability_validation=probability_validation,
        gate_config=gate_config,
        allow_extensions=allow_extensions,
    )
    env = ComboDqnEnv(base_env)
    if check:
        check_env(env, warn=True)
    set_random_seed(seed)
    resolved_hyperparameters = _dqn_hyperparameters(len(base_env.history), total_timesteps, hyperparameters)
    model = DQN(
        "MlpPolicy",
        env,
        device=device,
        seed=seed,
        **resolved_hyperparameters,
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    metadata = {
        "model_path": str(output),
        "algorithm": "DQN",
        "rule_version": COMBO_RULE_VERSION,
        "odds_version": COMBO_ODDS_VERSION,
        "action_space_version": f"discrete-projection-{COMBO_ACTION_SPACE_VERSION}",
        "feature_version": COMBO_FEATURE_VERSION,
        "feature_columns": list(COMBO_FEATURE_COLUMNS),
        "action_space": {"type": "Discrete", "n": int(env.action_space.n), "multi_nvec": list(env.nvec)},
        "portfolio_action_nvec": list(action_nvec(extreme_ranges=extreme_ranges, extensions_enabled=allow_extensions)),
        "extreme_ranges": extreme_ranges_metadata(extreme_ranges),
        "extensions_enabled": allow_extensions,
        "history_rows": len(base_env.history),
        "training_data_end_issue": base_env.history[-1]["issue"],
        "zero_policy": zero_policy.value,
        "training_hyperparameters": resolved_hyperparameters,
        "experiment": dict(experiment_metadata or {}),
        "training_protocol": QUALIFIED_OPPORTUNITY_PROTOCOL_VERSION,
        "probability_gate": {
            "enabled": True,
            "qualification": probability_validation,
            "require_probability_validation": gate_config.require_probability_validation,
            "risk_controls": {
                "min_expected_return_ratio": gate_config.min_expected_return_ratio,
                "min_base_coverage": gate_config.min_base_coverage,
                "max_drawdown": gate_config.max_drawdown,
                "max_losing_streak": gate_config.max_losing_streak,
                "cooldown_rounds": gate_config.cooldown_rounds,
                "target_bet_rate": gate_config.target_bet_rate,
                "allow_negative_expected_return": gate_config.allow_negative_expected_return,
            },
        },
    }
    metadata_path_for(output).write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _ppo_hyperparameters(history_rows: int, custom: Mapping[str, Any] | None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "learning_rate": 3e-4,
        "n_steps": min(128, max(2, history_rows)),
        "batch_size": min(64, max(2, history_rows)),
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        # A non-zero entropy bonus prevents an early deterministic collapse to
        # one combo.  Post-training diversity checks remain mandatory.
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
    _apply_hyperparameters(params, custom, algorithm="ppo")
    params["n_steps"] = int(params["n_steps"])
    params["batch_size"] = int(params["batch_size"])
    params["n_epochs"] = int(params["n_epochs"])
    if params["n_steps"] < 2 or params["batch_size"] < 2 or params["batch_size"] > params["n_steps"] or params["n_epochs"] <= 0:
        raise ValueError("PPO n_steps, batch_size and n_epochs are invalid")
    return params


def _dqn_hyperparameters(history_rows: int, total_timesteps: int, custom: Mapping[str, Any] | None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "learning_rate": 1e-4,
        "learning_starts": 0,
        "buffer_size": max(1_000, total_timesteps * 2),
        "batch_size": min(32, max(2, history_rows)),
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "target_update_interval": 10_000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
    }
    _apply_hyperparameters(params, custom, algorithm="dqn")
    for key in ("learning_starts", "buffer_size", "batch_size", "train_freq", "gradient_steps", "target_update_interval"):
        params[key] = int(params[key])
    if any(params[key] <= 0 for key in ("buffer_size", "batch_size", "train_freq", "gradient_steps", "target_update_interval")):
        raise ValueError("DQN integer hyperparameters must be positive")
    if params["learning_starts"] < 0:
        raise ValueError("DQN learning_starts must not be negative")
    return params


def _apply_hyperparameters(params: dict[str, Any], custom: Mapping[str, Any] | None, *, algorithm: str) -> None:
    if custom is None:
        return
    unknown = set(custom) - set(params)
    if unknown:
        raise ValueError(f"unsupported {algorithm} hyperparameters: {', '.join(sorted(unknown))}")
    params.update(dict(custom))


def _policy_diversity_report(model: Any, env: ComboCasinoEnv) -> dict[str, Any]:
    """Audit deterministic PPO outputs and reject a single fixed strategy."""

    observations = []
    for row in env.history:
        values = [float(row["features"][column]) for column in COMBO_FEATURE_COLUMNS]
        values.extend([1.0, 0.0, 0.0])
        if env.include_previous_combo_observation:
            values.append(0.0)
        observations.append(np.asarray(values, dtype=np.float32))
    actions: list[tuple[int, ...]] = []
    for observation in observations:
        action, _ = model.predict(observation, deterministic=True)
        values = tuple(int(value) for value in np.asarray(action).reshape(-1))
        actions.append(values)
    counts: dict[str, int] = {}
    base_counts: dict[str, int] = {}
    state_targets = [env._state_target_base_combo(row) for row in env.history]
    aligned_base_counts: dict[str, int] = {}
    aligned = 0
    for action, target in zip(actions, state_targets):
        key = ",".join(str(value) for value in action)
        counts[key] = counts.get(key, 0) + 1
        base = str(action[0]) if action else "-1"
        base_counts[base] = base_counts.get(base, 0) + 1
        if action and action[0] == target:
            aligned += 1
            aligned_base_counts[base] = aligned_base_counts.get(base, 0) + 1
    total = max(len(actions), 1)
    max_share = max(counts.values(), default=0) / total
    unique_actions = len(counts)
    unique_base = len(base_counts)
    alignment_rate = aligned / total
    unique_aligned_base = len(aligned_base_counts)
    return {
        "rows": len(actions),
        "unique_actions": unique_actions,
        "unique_base_combos": unique_base,
        "action_counts": counts,
        "base_combo_counts": base_counts,
        "state_target_base_combo_counts": {str(target): state_targets.count(target) for target in sorted(set(state_targets))},
        "state_alignment_rate": round(alignment_rate, 6),
        "state_aligned_base_combo_counts": aligned_base_counts,
        "unique_state_aligned_base_combos": unique_aligned_base,
        "max_action_share": round(max_share, 6),
        "passes": bool(
            unique_base >= 2
            and unique_actions >= 3
            and max_share <= 0.90
            and alignment_rate >= 0.40
            and unique_aligned_base >= 3
        ),
    }


def _training_probability_gate(
    rows: Any,
    *,
    zero_policy: ZeroPolicy,
    thresholds: ProbabilityValidationThresholds,
    gate_config: OpportunityGateConfig,
    experiment_metadata: Mapping[str, Any] | None,
):
    """Build the frozen probability gate and block unqualified formal runs."""

    cleaned_rows = clean_combo_history(rows, zero_policy=zero_policy, min_rows=100)
    model, report = fit_and_qualify_combo_probability_model(
        cleaned_rows,
        zero_policy=zero_policy,
        thresholds=thresholds,
    )
    if gate_config.experimental_non_champion:
        if (experiment_metadata or {}).get("classification") != "EXPERIMENTAL_NON_CHAMPION":
            raise ValueError("experimental training requires classification=EXPERIMENTAL_NON_CHAMPION")
        return model, report
    if not report["passed"]:
        raise ValueError("formal PPO/DQN training blocked: independent probability qualification failed")
    return model, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PC28 combo PPO model in simulation.")
    parser.add_argument("history", help="CSV/JSON/JSONL history file")
    parser.add_argument("--model", default=str(MODEL_DIR / "ppo_combo_balance_v1.zip"))
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-check-env", action="store_true")
    args = parser.parse_args()
    output = train_ppo_combo(
        args.history,
        model_path=args.model,
        total_timesteps=args.timesteps,
        initial_balance=args.initial_balance,
        seed=args.seed,
        device=args.device,
        check=not args.skip_check_env,
    )
    print(f"saved model: {output}")
    print(f"saved metadata: {metadata_path_for(output)}")


if __name__ == "__main__":
    main()
