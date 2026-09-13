"""Chronological evaluation for the probability-first PC28 decision policy."""

from __future__ import annotations

from typing import Any, Sequence

from .data_adapter import clean_combo_history
from .markets_combo import ExtremeRanges, ZeroPolicy
from .online_combo import decision_from_dict
from .portfolio_combo import PortfolioLimits
from .probability_combo import ComboProbabilityModel
from .settlement_combo import settle_combo
from .statistical_combo_policy import OpportunityGateConfig, choose_statistical_opportunity


def run_statistical_combo_backtest(
    history: Sequence[dict[str, Any]] | Any,
    *,
    model: ComboProbabilityModel,
    probability_calibration: dict[str, Any],
    initial_balance: float = 10_000.0,
    start_step: int = 0,
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    extreme_ranges: ExtremeRanges | None,
    limits: PortfolioLimits = PortfolioLimits(),
    gate_config: OpportunityGateConfig = OpportunityGateConfig(),
) -> dict[str, Any]:
    """Settle only post-start rows, carrying one real virtual balance forward."""

    rows = clean_combo_history(history, zero_policy=zero_policy, min_rows=1)
    start = min(max(int(start_step), 0), len(rows))
    balance = float(initial_balance)
    peak = balance
    losing_streak = 0
    executed_rounds = 0
    wait_rounds = 0
    reasons: dict[str, int] = {}
    equity_curve = [balance]
    decisions: list[dict[str, Any]] = []
    for row in rows[start:]:
        drawdown = max(0.0, (peak - balance) / max(peak, 1e-9))
        opportunity = choose_statistical_opportunity(
            model,
            features=row["features"],
            balance=balance,
            drawdown=drawdown,
            losing_streak=losing_streak,
            probability_validation=probability_calibration,
            issue=row["issue"],
            as_of_issue=row["issue"],
            extreme_ranges=extreme_ranges,
            zero_policy=zero_policy,
            limits=limits,
            config=gate_config,
        )
        reasons[opportunity.reason] = reasons.get(opportunity.reason, 0) + 1
        if opportunity.wait:
            wait_rounds += 1
            equity_curve.append(balance)
            decisions.append({"issue": row["issue"], **opportunity.to_dict(), "balance": balance})
            continue
        assert opportunity.decision is not None
        settlement = settle_combo(
            decision_from_dict(opportunity.decision),
            outcome_sum=int(row["sum"]),
            balance_before=balance,
            zero_policy=zero_policy,
            extreme_ranges=extreme_ranges,
            limits=limits,
        )
        balance = settlement.balance_after
        peak = max(peak, balance)
        losing_streak = losing_streak + 1 if settlement.net_profit < 0 else 0
        executed_rounds += 1
        equity_curve.append(balance)
        decisions.append({
            "issue": row["issue"],
            **opportunity.to_dict(),
            "settlement": settlement.to_dict(),
            "balance": balance,
        })
    return {
        "steps": len(rows) - start,
        "executed_rounds": executed_rounds,
        "wait_rounds": wait_rounds,
        "wait_rate": round(wait_rounds / max(len(rows) - start, 1), 6),
        "final_balance": round(balance, 2),
        "net_profit": round(balance - initial_balance, 2),
        "max_drawdown": round(_max_drawdown(equity_curve), 6),
        "wait_reasons": reasons,
        "equity_curve": [round(value, 2) for value in equity_curve],
        "decisions": decisions,
    }


def _max_drawdown(values: list[float]) -> float:
    peak = values[0]
    result = 0.0
    for value in values:
        peak = max(peak, value)
        result = max(result, (peak - value) / max(peak, 1e-9))
    return result
