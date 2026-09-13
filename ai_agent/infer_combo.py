"""Inference helpers for models trained with the combo action schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from stable_baselines3 import DQN, PPO

from .combo_env import ComboCasinoEnv
from .data_adapter import COMBO_FEATURE_COLUMNS, COMBO_FEATURE_VERSION, clean_combo_history
from .markets_combo import COMBO_ACTION_SPACE_VERSION, COMBO_ODDS_VERSION, COMBO_RULE_VERSION, CONFIRMED_EXTREME_RANGES, ExtremeRanges, ZeroPolicy, extreme_ranges_metadata
from .portfolio_combo import BASE_COMBINATIONS, MODE_IDS, STAKE_PROFILES, WAIT_BASE_ACTION_ID, action_nvec, build_decision, extension_pairs
from .train_combo import (
    dqn_index_to_combo_action,
    metadata_path_for,
)


def validate_combo_model_metadata(
    model_path: str | Path,
    *,
    algorithm: str,
    expected_nvec: tuple[int, int, int, int] | None = None,
    zero_policy: ZeroPolicy | None = None,
    extreme_ranges: ExtremeRanges | None = None,
    extensions_enabled: bool = False,
) -> dict[str, Any]:
    """Reject a legacy or cross-algorithm model before inference."""

    path = metadata_path_for(model_path)
    if not path.exists():
        raise ValueError(f"combo model metadata not found: {path}")
    import json

    metadata = json.loads(path.read_text(encoding="utf-8"))
    expected_algorithm = algorithm.upper()
    if metadata.get("algorithm") != expected_algorithm:
        raise ValueError("model algorithm does not match the requested combo inference algorithm")
    if metadata.get("rule_version") != COMBO_RULE_VERSION:
        raise ValueError("model is not compatible with the PC28 combo rule version")
    if metadata.get("odds_version") != COMBO_ODDS_VERSION:
        raise ValueError("model odds version is not compatible with PC28 combo")
    expected_action_version = (
        COMBO_ACTION_SPACE_VERSION
        if expected_algorithm == "PPO"
        else f"discrete-projection-{COMBO_ACTION_SPACE_VERSION}"
    )
    if metadata.get("action_space_version") != expected_action_version:
        raise ValueError("model action schema is not compatible with PC28 combo")
    if metadata.get("feature_version") != COMBO_FEATURE_VERSION:
        raise ValueError("model feature version is not compatible with PC28 combo")
    if metadata.get("feature_columns") != list(COMBO_FEATURE_COLUMNS):
        raise ValueError("model feature columns are not compatible with PC28 combo")
    expected_nvec = expected_nvec or action_nvec(
        extreme_ranges=extreme_ranges,
        extensions_enabled=extensions_enabled,
    )
    action_space = metadata.get("action_space")
    if not isinstance(action_space, dict):
        raise ValueError("combo model metadata requires an action_space")
    actual_nvec = action_space.get("nvec", action_space.get("multi_nvec"))
    if actual_nvec != list(expected_nvec):
        raise ValueError("model action dimensions are not compatible with the active combo markets")
    if expected_algorithm == "PPO" and action_space.get("type") != "MultiDiscrete":
        raise ValueError("PPO combo model must declare a MultiDiscrete action space")
    if expected_algorithm == "DQN":
        if action_space.get("type") != "Discrete" or action_space.get("n") != _action_count(expected_nvec):
            raise ValueError("DQN combo model must declare the matching discrete action projection")
    if zero_policy is not None and metadata.get("zero_policy") != zero_policy.value:
        raise ValueError("model zero policy is not compatible with the active combo policy")
    if metadata.get("extreme_ranges") != extreme_ranges_metadata(extreme_ranges):
        raise ValueError("model extreme ranges are not compatible with the active combo markets")
    if bool(metadata.get("extensions_enabled", False)) != extensions_enabled:
        raise ValueError("model extension setting is not compatible with the active combo markets")
    return metadata


def action_to_combo_decision(
    action: list[int] | tuple[int, ...],
    *,
    balance: float,
    extreme_ranges: ExtremeRanges | None = CONFIRMED_EXTREME_RANGES,
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    issue: str | None = None,
    as_of_issue: str | None = None,
    model_version: str | None = None,
    probabilities: Mapping[str, float] | None = None,
    allow_extensions: bool = False,
) -> dict[str, Any]:
    """Decode an action and return the JSON-friendly legal decision."""

    if len(action) != 4:
        raise ValueError("combo action must contain four integers")
    base_id, extension_id, profile_id, mode_id = (int(value) for value in action)
    if base_id == WAIT_BASE_ACTION_ID:
        return {
            "wait": True,
            "selections": [],
            "mode": "wait",
            "issue": issue,
            "as_of_issue": as_of_issue,
            "model_version": model_version,
        }
    available_extensions = extension_pairs(
        extreme_ranges=extreme_ranges,
        extensions_enabled=allow_extensions,
    )
    if not 0 <= base_id < len(BASE_COMBINATIONS):
        raise ValueError("base combo action index out of range")
    if not 0 <= extension_id < len(available_extensions):
        raise ValueError("extension action index out of range for the configured markets")
    if not 0 <= profile_id < len(STAKE_PROFILES) or not 0 <= mode_id < len(MODE_IDS):
        raise ValueError("stake profile or mode action index out of range")
    extensions = available_extensions[extension_id]
    decision = build_decision(
        action=(base_id, extension_id, profile_id, mode_id),
        balance=balance,
        issue=issue,
        as_of_issue=as_of_issue,
        model_version=model_version,
        probabilities=probabilities,
        extension_options=available_extensions,
    )
    return decision.to_dict()


def predict_combo_decision(
    history_source: Any,
    *,
    model_path: str | Path,
    algorithm: str = "ppo",
    balance: float = 10_000.0,
    deterministic: bool = True,
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    extreme_ranges: ExtremeRanges | None = CONFIRMED_EXTREME_RANGES,
    allow_extensions: bool = False,
) -> dict[str, Any]:
    rows = clean_combo_history(history_source, zero_policy=zero_policy, min_rows=1)
    env = ComboCasinoEnv(
        rows,
        initial_balance=balance,
        zero_policy=zero_policy,
        extreme_ranges=extreme_ranges,
        allow_extensions=allow_extensions,
    )
    obs, _ = env.reset(options={"start_step": len(rows) - 1})
    normalized_algorithm = algorithm.lower()
    if normalized_algorithm == "ppo":
        validate_combo_model_metadata(
            model_path,
            algorithm="ppo",
            expected_nvec=tuple(int(value) for value in env.action_space.nvec),
            zero_policy=zero_policy,
            extreme_ranges=extreme_ranges,
            extensions_enabled=allow_extensions,
        )
        model = PPO.load(model_path)
        action, _ = model.predict(obs, deterministic=deterministic)
        action_values = tuple(int(value) for value in action)
    elif normalized_algorithm == "dqn":
        validate_combo_model_metadata(
            model_path,
            algorithm="dqn",
            expected_nvec=tuple(int(value) for value in env.action_space.nvec),
            zero_policy=zero_policy,
            extreme_ranges=extreme_ranges,
            extensions_enabled=allow_extensions,
        )
        model = DQN.load(model_path)
        action, _ = model.predict(obs, deterministic=deterministic)
        action_values = dqn_index_to_combo_action(int(action), tuple(int(value) for value in env.action_space.nvec))
    else:
        raise ValueError("algorithm must be ppo or dqn")
    decision = action_to_combo_decision(
        list(action_values),
        balance=balance,
        extreme_ranges=extreme_ranges,
        zero_policy=zero_policy,
        issue=_next_issue(rows[-1]["issue"]),
        as_of_issue=rows[-1]["issue"],
        model_version=Path(model_path).stem,
        probabilities=_selection_probabilities(rows[-1]["features"], extreme_ranges=extreme_ranges),
        allow_extensions=allow_extensions,
    )
    return {
        "issue": decision["issue"],
        "as_of_issue": rows[-1]["issue"],
        "action": list(action_values),
        "decision": decision,
        "wait": bool(decision.get("wait", False)),
        "rule_version": "pc28-combo-v1",
    }


def _next_issue(issue: str) -> str:
    """Derive the next numeric issue without discarding zero padding."""

    normalized = str(issue).strip()
    if not normalized.isdigit():
        raise ValueError("cannot derive target issue from a non-numeric latest issue")
    return str(int(normalized) + 1).zfill(len(normalized))


def _selection_probabilities(
    features: Mapping[str, Any],
    *,
    extreme_ranges: ExtremeRanges | None = None,
) -> dict[str, float]:
    """Use only pre-draw history features for JSON decision probabilities."""

    probabilities = {
        "small_odd": float(features.get("base_rate_small_odd_100", 0.0)),
        "small_even": float(features.get("base_rate_small_even_100", 0.0)),
        "big_odd": float(features.get("base_rate_big_odd_100", 0.0)),
        "big_even": float(features.get("base_rate_big_even_100", 0.0)),
    }
    probabilities.update(
        {
            f"sum_{total:02d}": float(features.get(f"sum_rate_{total:02d}_100", 0.0))
            for total in range(28)
        }
    )
    if extreme_ranges is not None:
        probabilities["extreme_small"] = sum(
            probabilities[f"sum_{total:02d}"]
            for total in range(extreme_ranges.small_min, extreme_ranges.small_max + 1)
        )
        probabilities["extreme_big"] = sum(
            probabilities[f"sum_{total:02d}"]
            for total in range(extreme_ranges.big_min, extreme_ranges.big_max + 1)
        )
    return probabilities


def _action_count(nvec: tuple[int, int, int, int]) -> int:
    result = 1
    for value in nvec:
        result *= value
    return result
