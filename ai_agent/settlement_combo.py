"""Independent per-market PC28 combo settlement."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from math import isfinite

from .markets_combo import ExtremeRanges, ZeroPolicy, market_hits, validate_sum
from .portfolio_combo import PortfolioLimits, validate_decision
from .schemas_combo import ComboDecision, ComboSettlement, DecisionMode, SettlementItem


def settle_combo(
    decision: ComboDecision,
    *,
    outcome_sum: int,
    balance_before: float,
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    extreme_ranges: ExtremeRanges | None = None,
    limits: PortfolioLimits = PortfolioLimits(),
) -> ComboSettlement:
    """Settle every selected market independently using gross (stake-inclusive) odds."""

    validate_decision(decision, limits=limits)
    outcome_sum = validate_sum(outcome_sum)
    if not isfinite(balance_before) or balance_before < 0:
        raise ValueError("balance_before must be finite and non-negative")
    proposed_stake = sum(selection.stake for selection in decision.selections)
    maximum_exposure = min(balance_before * limits.max_exposure_ratio, limits.max_round_stake)
    if proposed_stake > maximum_exposure + 1e-9:
        raise ValueError("combo decision exceeds the current one-round exposure budget")

    items: list[SettlementItem] = []
    total_stake = Decimal("0")
    total_return = Decimal("0")
    for selection in decision.selections:
        stake = _money(selection.stake)
        hit = market_hits(
            selection.market,
            outcome_sum,
            zero_policy=zero_policy,
            extreme_ranges=extreme_ranges,
        )
        returned = stake * Decimal(str(selection.odds)) if hit else Decimal("0")
        contribution = returned - stake
        total_stake += stake
        total_return += returned
        items.append(
            SettlementItem(
                selection=selection,
                hit=hit,
                return_amount=float(_money(returned)),
                net_contribution=float(_money(contribution)),
            )
        )

    total_stake = _money(total_stake)
    total_return = _money(total_return)
    net_profit = _money(total_return - total_stake)
    is_shadow = decision.mode == DecisionMode.SHADOW.value
    balance_delta = Decimal("0") if is_shadow else net_profit
    balance_after = _money(Decimal(str(balance_before)) + balance_delta)
    return ComboSettlement(
        decision=decision,
        outcome_sum=outcome_sum,
        items=tuple(items),
        total_stake=float(total_stake),
        total_return=float(total_return),
        net_profit=float(net_profit),
        balance_before=float(_money(Decimal(str(balance_before)))),
        balance_after=float(balance_after),
        balance_delta=float(balance_delta),
        is_shadow=is_shadow,
        shadow_net_profit=float(net_profit) if is_shadow else None,
    )


def _money(value: Decimal | float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
