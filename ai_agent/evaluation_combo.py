"""Promotion gates for PC28 combo models and time-order walk-forward reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from stable_baselines3 import DQN, PPO

from .backtest_combo import ComboPolicy, combo_baseline_policy, run_combo_backtest
from .markets_combo import ExtremeRanges, ZeroPolicy
from .probability_combo import fit_and_qualify_combo_probability_model
from .statistical_combo_policy import OpportunityGateConfig
from .train_combo import dqn_index_to_combo_action


ComboPolicyFactory = Callable[[list[dict[str, Any]]], ComboPolicy]


@dataclass(frozen=True)
class ComboEvaluationThresholds:
    max_drawdown: float = 0.35
    max_exposure_ratio: float = 1.0
    min_executed_rounds: int = 30
    min_profit_window_rate: float = 0.60
    min_walk_forward_windows: int = 5
    max_extension_positive_profit_share: float = 0.75


def run_combo_walk_forward(
    history: list[dict[str, Any]],
    *,
    policy: ComboPolicy | None = None,
    policy_factory: ComboPolicyFactory | None = None,
    window_size: int,
    windows: int = 5,
    initial_balance: float = 10_000.0,
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    extreme_ranges: ExtremeRanges | None = None,
    probability_model: Any | None = None,
    probability_validation: dict[str, Any] | None = None,
    gate_config: OpportunityGateConfig = OpportunityGateConfig(),
    include_previous_combo_observation: bool = False,
) -> dict[str, Any]:
    """Evaluate a fixed policy over contiguous chronological windows.

    Model training must happen before this function is called.  This function
    never lets one evaluated window influence another.
    """

    if window_size <= 0 or windows <= 0:
        raise ValueError("window_size and windows must be positive")
    if (policy is None) == (policy_factory is None):
        raise ValueError("provide exactly one of policy or policy_factory")
    required = window_size * windows
    if len(history) < required:
        raise ValueError(f"need at least {required} rows for {windows} walk-forward windows")
    evaluation_start = len(history) - required
    results = []
    for index in range(windows):
        start = evaluation_start + index * window_size
        end = start + window_size
        segment = history[start:end]
        context = history[:end]
        train_rows = history[:start]
        active_policy = policy_factory(train_rows) if policy_factory is not None else policy
        assert active_policy is not None
        agent = run_combo_backtest(
            context,
            policy=active_policy,
            initial_balance=initial_balance,
            zero_policy=zero_policy,
            extreme_ranges=extreme_ranges,
            start_step=start,
            probability_model=probability_model,
            probability_validation=probability_validation,
            gate_config=gate_config,
            include_previous_combo_observation=include_previous_combo_observation,
        )
        baselines = {
            name: run_combo_backtest(
                context,
                policy=combo_baseline_policy(name, seed=index, extreme_ranges=extreme_ranges),
                initial_balance=initial_balance,
                zero_policy=zero_policy,
                extreme_ranges=extreme_ranges,
                start_step=start,
                probability_model=probability_model,
                probability_validation=probability_validation,
                gate_config=gate_config,
                include_previous_combo_observation=include_previous_combo_observation,
            )
            for name in ("fixed_defense", "fixed_balanced", "probability", "random")
        }
        results.append(
            {
                "window": index + 1,
                "train_rows": len(train_rows),
                "train_end_issue": train_rows[-1]["issue"] if train_rows else None,
                "test_start_issue": segment[0]["issue"],
                "test_end_issue": segment[-1]["issue"],
                "agent": agent,
                "baselines": baselines,
            }
        )
    return {"walk_forward": results, "window_size": window_size, "windows": windows}


def classify_combo_evaluation(
    report: dict[str, Any],
    *,
    thresholds: ComboEvaluationThresholds = ComboEvaluationThresholds(),
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
) -> dict[str, Any]:
    windows = [item for item in report.get("walk_forward", []) if isinstance(item, dict)]
    checks: list[dict[str, Any]] = []
    if len(windows) < thresholds.min_walk_forward_windows:
        checks.append(_check("minimum_windows", f">= {thresholds.min_walk_forward_windows}", len(windows), False))
    else:
        checks.append(_check("minimum_windows", f">= {thresholds.min_walk_forward_windows}", len(windows), True))
    passes = 0
    violations: list[dict[str, Any]] = []
    for item in windows:
        window_checks = _window_checks(item, thresholds)
        passed = all(check["passed"] for check in window_checks)
        if passed:
            passes += 1
        else:
            violations.append({"window": item.get("window"), "checks": [check for check in window_checks if not check["passed"]]})
    pass_rate = passes / max(len(windows), 1)
    checks.append(_check("profit_window_rate", f">= {thresholds.min_profit_window_rate:.0%}", round(pass_rate, 6), pass_rate >= thresholds.min_profit_window_rate))
    status = "PASS" if all(check["passed"] for check in checks) and not violations else "FAIL"
    return {
        "status": status,
        "thresholds": asdict(thresholds),
        "summary": {"windows": len(windows), "passed_windows": passes, "profit_window_rate": round(pass_rate, 6)},
        "checks": checks,
        "violations": violations,
    }


def evaluate_combo_model(
    history: list[dict[str, Any]],
    *,
    model_path: str | Path,
    algorithm: str,
    window_size: int,
    windows: int = 5,
    initial_balance: float = 10_000.0,
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    extreme_ranges: ExtremeRanges | None = None,
    gate_config: OpportunityGateConfig | None = None,
    include_previous_combo_observation: bool = False,
) -> dict[str, Any]:
    """Evaluate a stored combo model only on issues after its training cutoff.

    This is intentionally separate from model training.  It refuses a report
    if metadata cannot establish that all test windows are after the model's
    declared training-data cutoff.
    """

    from .infer_combo import validate_combo_model_metadata

    metadata = validate_combo_model_metadata(
        model_path,
        algorithm=algorithm,
        zero_policy=zero_policy,
        extreme_ranges=extreme_ranges,
    )
    cutoff = str(metadata.get("training_data_end_issue", ""))
    if not cutoff.isdigit():
        raise ValueError("combo model metadata requires a numeric training_data_end_issue")
    required = window_size * windows
    if len(history) < required:
        raise ValueError(f"need at least {required} rows for {windows} walk-forward windows")
    test_start = str(history[-required].get("issue", ""))
    if not test_start.isdigit() or int(test_start) <= int(cutoff):
        raise ValueError("walk-forward test period must start after the model training cutoff")

    policy = _loaded_combo_policy(model_path, algorithm=algorithm, nvec=tuple(int(value) for value in metadata["action_space"].get("nvec", metadata["action_space"].get("multi_nvec", []))))
    training_rows = [row for row in history if int(str(row.get("issue", "0"))) <= int(cutoff)]
    if not training_rows:
        raise ValueError("evaluation history does not contain the declared training range")
    probability_model, probability_validation = fit_and_qualify_combo_probability_model(
        training_rows,
        zero_policy=zero_policy,
    )
    gate_config = gate_config or OpportunityGateConfig()
    report = run_combo_walk_forward(
        history,
        policy=policy,
        window_size=window_size,
        windows=windows,
        initial_balance=initial_balance,
        zero_policy=zero_policy,
        extreme_ranges=extreme_ranges,
        probability_model=probability_model,
        probability_validation=probability_validation,
        gate_config=gate_config,
        include_previous_combo_observation=include_previous_combo_observation,
    )
    validation_start = len(history) - required
    continuous_agent = run_combo_backtest(
        history,
        policy=policy,
        initial_balance=initial_balance,
        zero_policy=zero_policy,
        extreme_ranges=extreme_ranges,
        start_step=validation_start,
        probability_model=probability_model,
        probability_validation=probability_validation,
        gate_config=gate_config,
        include_previous_combo_observation=include_previous_combo_observation,
    )
    continuous_baselines = {
        name: run_combo_backtest(
            history,
            policy=combo_baseline_policy(name, seed=0, extreme_ranges=extreme_ranges),
            initial_balance=initial_balance,
            zero_policy=zero_policy,
            extreme_ranges=extreme_ranges,
            start_step=validation_start,
            probability_model=probability_model,
            probability_validation=probability_validation,
            gate_config=gate_config,
            include_previous_combo_observation=include_previous_combo_observation,
        )
        for name in ("fixed_defense", "fixed_balanced", "probability", "random")
    }
    report["continuous_validation"] = {
        "start_issue": history[validation_start]["issue"],
        "end_issue": history[-1]["issue"],
        "initial_balance": initial_balance,
        "agent": continuous_agent,
        "baselines": continuous_baselines,
    }
    report["model"] = {
        "path": str(model_path),
        "algorithm": algorithm.upper(),
        "training_data_end_issue": cutoff,
        "rule_version": metadata.get("rule_version"),
        "odds_version": metadata.get("odds_version"),
        "action_space_version": metadata.get("action_space_version"),
        "extreme_ranges": metadata.get("extreme_ranges"),
    }
    report["probability_gate"] = {
        "training_rows": len(training_rows),
        "training_end_issue": training_rows[-1]["issue"],
        "qualification": probability_validation,
        "validation": probability_validation,
        "risk_controls": {
            "min_expected_return_ratio": gate_config.min_expected_return_ratio,
            "min_base_coverage": gate_config.min_base_coverage,
            "max_drawdown": gate_config.max_drawdown,
            "max_losing_streak": gate_config.max_losing_streak,
            "cooldown_rounds": gate_config.cooldown_rounds,
            "target_bet_rate": gate_config.target_bet_rate,
            "allow_negative_expected_return": gate_config.allow_negative_expected_return,
            "include_previous_combo_observation": include_previous_combo_observation,
        },
    }
    report["verdict"] = classify_combo_evaluation(report, zero_policy=zero_policy)
    return report


def _loaded_combo_policy(
    model_path: str | Path,
    *,
    algorithm: str,
    nvec: tuple[int, ...],
) -> ComboPolicy:
    normalized = algorithm.lower()
    if normalized == "ppo":
        model = PPO.load(model_path)

        def ppo_policy(obs: Any, _info: dict[str, Any]) -> list[int]:
            action, _ = model.predict(obs, deterministic=True)
            return [int(value) for value in action]

        return ppo_policy
    if normalized == "dqn":
        if len(nvec) != 4:
            raise ValueError("DQN combo metadata must contain four-dimensional multi_nvec")
        model = DQN.load(model_path)

        def dqn_policy(obs: Any, _info: dict[str, Any]) -> list[int]:
            action, _ = model.predict(obs, deterministic=True)
            return list(dqn_index_to_combo_action(int(action), tuple(int(value) for value in nvec)))

        return dqn_policy
    raise ValueError("algorithm must be ppo or dqn")


def _window_checks(item: dict[str, Any], thresholds: ComboEvaluationThresholds) -> list[dict[str, Any]]:
    agent = item.get("agent", {})
    baselines = item.get("baselines", {})
    agent_profit = _float(agent, "net_profit")
    strongest_baseline = max((_float(metrics, "net_profit") for metrics in baselines.values()), default=0.0)
    checks = [
        _check("beats_baselines", "> strongest baseline", agent_profit, agent_profit > strongest_baseline),
        _check("max_drawdown", f"<= {thresholds.max_drawdown:.0%}", _float(agent, "max_drawdown"), _float(agent, "max_drawdown") <= thresholds.max_drawdown),
        _check("max_exposure_ratio", f"<= {thresholds.max_exposure_ratio:.0%}", _float(agent, "max_exposure_ratio"), _float(agent, "max_exposure_ratio") <= thresholds.max_exposure_ratio),
        _check("executed_rounds", f">= {thresholds.min_executed_rounds}", int(_float(agent, "executed_rounds")), _float(agent, "executed_rounds") >= thresholds.min_executed_rounds),
        _check("extension_positive_profit_share", f"<= {thresholds.max_extension_positive_profit_share:.0%}", _float(agent, "extension_positive_profit_share"), _float(agent, "extension_positive_profit_share") <= thresholds.max_extension_positive_profit_share),
    ]
    return checks


def _check(name: str, threshold: str, value: Any, passed: bool) -> dict[str, Any]:
    return {"name": name, "threshold": threshold, "value": value, "passed": bool(passed)}


def _float(metrics: Any, name: str) -> float:
    try:
        return float(metrics.get(name, 0.0)) if isinstance(metrics, dict) else 0.0
    except (TypeError, ValueError):
        return 0.0
