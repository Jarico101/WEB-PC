"""Time-ordered backtesting for the PC28 combo environment."""

from __future__ import annotations

import random
from typing import Any, Callable, Sequence

from .combo_env import ComboCasinoEnv
from .markets_combo import ExtremeRanges, ZeroPolicy
from .portfolio_combo import BASE_COMBINATIONS, BASE_ONLY_EXTENSION_PAIRS, WAIT_BASE_ACTION_ID, action_nvec
from .probability_combo import ComboProbabilityModel
from .statistical_combo_policy import OpportunityGateConfig, choose_statistical_opportunity


ComboPolicy = Callable[[Any, dict[str, Any]], Sequence[int]]


def probability_gated_policy(
    model_policy: ComboPolicy,
    probability_model: ComboProbabilityModel,
    *,
    probability_validation: dict[str, Any],
    extreme_ranges: ExtremeRanges | None,
    gate_config: OpportunityGateConfig = OpportunityGateConfig(),
    include_previous_combo_observation: bool = False,
) -> ComboPolicy:
    """Turn a learned action into a probability/risk-gated action stream.

    The learned action remains auditable as ``model_action`` in the returned
    policy's info, while the gate can replace it with the canonical wait action
    or the highest expected-value legal portfolio.
    """

    def gated(obs: Any, info: dict[str, Any]) -> Sequence[int]:
        opportunity = choose_statistical_opportunity(
            probability_model,
            features=info.get("features", {}),
            balance=float(info.get("balance", 10_000.0)),
            drawdown=float(info.get("drawdown", 0.0)),
            losing_streak=int(info.get("losing_streak", 0)),
            probability_validation=probability_validation,
            issue=info.get("issue"),
            as_of_issue=info.get("issue"),
            extreme_ranges=extreme_ranges,
            config=gate_config,
        )
        if opportunity.wait:
            return [WAIT_BASE_ACTION_ID, 0, 0, 0]
        assert opportunity.action is not None
        return list(opportunity.action)

    return gated


def run_combo_backtest(
    history: Any,
    *,
    policy: ComboPolicy,
    initial_balance: float = 10_000.0,
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    extreme_ranges: ExtremeRanges | None = None,
    start_step: int = 0,
    allow_shadow_mode: bool = False,
    probability_model: ComboProbabilityModel | None = None,
    probability_validation: dict[str, Any] | None = None,
    gate_config: OpportunityGateConfig = OpportunityGateConfig(),
    include_previous_combo_observation: bool = False,
    allow_extensions: bool = False,
    extension_options: tuple[tuple[str, ...], ...] | None = None,
) -> dict[str, Any]:
    """Run one chronological simulation; policy sees no current result fields."""

    env = ComboCasinoEnv(
        history,
        initial_balance=initial_balance,
        zero_policy=zero_policy,
        extreme_ranges=extreme_ranges,
        allow_shadow_mode=allow_shadow_mode or bool(getattr(policy, "allow_shadow_mode", False)),
        probability_model=probability_model,
        probability_validation=probability_validation,
        gate_config=gate_config,
        include_previous_combo_observation=include_previous_combo_observation,
        allow_extensions=allow_extensions,
        extension_options=extension_options,
    )
    obs, info = env.reset(options={"start_step": start_step})
    equity_curve = [initial_balance]
    decisions: list[dict[str, Any]] = []
    executed_rounds = 0
    wins = 0
    losses = 0
    longest_losing_streak = 0
    total_exposure = 0.0
    max_exposure_ratio = 0.0
    extension_positive_profit = 0.0
    all_positive_profit = 0.0
    unconfirmed_zero_outcomes = 0
    wait_rounds = 0
    insufficient_balance_events = 0
    termination_reason: str | None = None
    terminated = truncated = False
    while not (terminated or truncated):
        action = list(policy(obs, dict(info)))
        obs, reward, terminated, truncated, info = env.step(action)
        settlement = info
        if settlement.get("wait", False):
            wait_rounds += 1
        if settlement.get("insufficient_balance", False):
            insufficient_balance_events += 1
        termination_reason = settlement.get("termination_reason") or termination_reason
        balance_before = float(settlement.get("balance_before", equity_curve[-1]))
        total_stake = float(settlement.get("total_stake", 0.0))
        total_exposure += total_stake
        max_exposure_ratio = max(max_exposure_ratio, total_stake / max(balance_before, 1e-9))
        if int(settlement.get("outcome_sum", -1)) == 0 and zero_policy == ZeroPolicy.EXCLUDE:
            unconfirmed_zero_outcomes += 1
        for item in settlement.get("items", []):
            contribution = float(item.get("net_contribution", 0.0))
            if contribution > 0:
                all_positive_profit += contribution
                if item.get("selection", {}).get("market_type") != "base":
                    extension_positive_profit += contribution
        if settlement.get("order_executed", True) and not settlement.get("is_shadow", False):
            executed_rounds += 1
            if float(settlement.get("net_profit", 0.0)) > 0:
                wins += 1
            elif float(settlement.get("net_profit", 0.0)) < 0:
                losses += 1
        longest_losing_streak = max(longest_losing_streak, int(info.get("losing_streak", 0)))
        equity_curve.append(float(info.get("balance", equity_curve[-1])))
        decisions.append(
            {
                "issue": info.get("issue"),
                "action": info.get("action"),
                "requested_action": list(action),
                "mode": info.get("decision", {}).get("mode"),
                "selections": info.get("decision", {}).get("selections", []),
                "outcome_sum": info.get("outcome_sum"),
                "balance_before": info.get("balance_before"),
                "total_stake": info.get("total_stake"),
                "total_return": info.get("total_return"),
                "net_profit": info.get("net_profit"),
                "balance": info.get("balance"),
                "is_shadow": info.get("is_shadow"),
                "order_executed": bool(info.get("order_executed", True)),
                "wait": bool(info.get("wait", False)),
                "wait_reason": info.get("wait_reason"),
                "cooldown_remaining": info.get("cooldown_remaining", 0),
                "probability_gate": info.get("probability_gate"),
                "termination_reason": info.get("termination_reason"),
                "reward": reward,
            }
        )
    return {
        "steps": len(decisions),
        "executed_rounds": executed_rounds,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / max(executed_rounds, 1), 6),
        "longest_losing_streak": longest_losing_streak,
        "final_balance": round(equity_curve[-1], 2),
        "net_profit": round(equity_curve[-1] - initial_balance, 2),
        "max_drawdown": round(_max_drawdown(equity_curve), 6),
        "total_exposure": round(total_exposure, 2),
        "max_exposure_ratio": round(max_exposure_ratio, 6),
        "extension_positive_profit_share": round(extension_positive_profit / max(all_positive_profit, 1e-9), 6),
        "unconfirmed_zero_outcomes": unconfirmed_zero_outcomes,
        "wait_rounds": wait_rounds,
        "insufficient_balance_events": insufficient_balance_events,
        "termination_reason": termination_reason,
        "equity_curve": [round(value, 2) for value in equity_curve],
        "decisions": decisions,
    }


def combo_baseline_policy(
    name: str,
    *,
    seed: int = 42,
    extreme_ranges: ExtremeRanges | None = None,
) -> ComboPolicy:
    """Return controls that obey the same three-base-item contract as the AI."""

    if name == "fixed_defense":
        return lambda obs, info: [0, 0, 0, 0]
    if name == "fixed_balanced":
        return lambda obs, info: [1, 0, 1, 1]
    if name == "shadow":
        def shadow_policy(_obs, _info):
            return [0, 0, 0, 3]

        setattr(shadow_policy, "allow_shadow_mode", True)
        return shadow_policy
    if name == "random":
        rng = random.Random(seed)
        nvec = action_nvec(extreme_ranges=extreme_ranges)
        return lambda obs, info: [rng.randrange(nvec[0]), rng.randrange(nvec[1]), rng.randrange(nvec[2]), rng.randrange(3)]
    if name == "probability":
        return _probability_baseline_action
    raise ValueError(f"unknown combo baseline: {name}")


def _probability_baseline_action(obs: Any, _info: dict[str, Any]) -> list[int]:
    """Choose three base markets by prior 100-period empirical frequency.

    It is deliberately simple and has no access to the current outcome.  It is
    a control, not a claim that historical frequency predicts the next result.
    """

    rates = [float(value) for value in obs[:4]]
    desired = {
        market
        for _score, market in sorted(
            zip(rates, ("small_odd", "small_even", "big_odd", "big_even")),
            reverse=True,
        )[:3]
    }
    combo_id = next(index for index, combo in enumerate(BASE_COMBINATIONS) if set(combo) == desired)
    return [combo_id, 0, 0, 0]


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    result = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            result = max(result, (peak - value) / peak)
    return result
