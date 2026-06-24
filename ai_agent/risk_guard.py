from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskConfig:
    max_bet_ratio: float = 0.10
    loss_streak_reduce_threshold: int = 2
    loss_streak_stop_threshold: int = 3
    capital_reduce_ratio: float = 0.70
    capital_stop_ratio: float = 0.50
    reduced_max_bet_ratio: float = 0.02
    stale_after_seconds: int = 90
    high_pressure_score: float = 85.0


@dataclass(frozen=True)
class RuntimeState:
    issue: str
    balance: float
    initial_balance: float
    losing_streak: int
    pressure_score: float = 0.0
    data_age_seconds: float | None = None
    already_bet_issues: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    direction: int
    amount_ratio: float
    capped_amount_ratio: float
    reasons: tuple[str, ...]


def evaluate_risk(
    *,
    direction: int,
    amount_ratio: float,
    state: RuntimeState,
    config: RiskConfig = RiskConfig(),
) -> GuardDecision:
    reasons: list[str] = []
    capped = min(float(amount_ratio), config.max_bet_ratio)
    capital_ratio = state.balance / max(state.initial_balance, 1e-9)

    if direction == 0:
        return GuardDecision(True, direction, amount_ratio, 0.0, ("watch",))
    if state.issue in state.already_bet_issues:
        reasons.append("duplicate_issue")
    if state.data_age_seconds is not None and state.data_age_seconds > config.stale_after_seconds:
        reasons.append("stale_data")
    if capital_ratio < config.capital_stop_ratio:
        reasons.append("capital_below_stop")
    elif capital_ratio < config.capital_reduce_ratio:
        capped = min(capped, config.reduced_max_bet_ratio)
        reasons.append("capital_reduced")
    if state.losing_streak >= config.loss_streak_stop_threshold:
        reasons.append("losing_streak_stop")
    elif state.losing_streak >= config.loss_streak_reduce_threshold:
        capped = min(capped, config.reduced_max_bet_ratio)
        reasons.append("losing_streak_reduced")
    if state.pressure_score >= config.high_pressure_score:
        capped = min(capped, config.reduced_max_bet_ratio)
        reasons.append("high_pressure_capped")

    blocked = any(reason in reasons for reason in ("duplicate_issue", "stale_data", "capital_below_stop", "losing_streak_stop"))
    return GuardDecision(not blocked, direction, amount_ratio, capped, tuple(reasons or ("allowed",)))

