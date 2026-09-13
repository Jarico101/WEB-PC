"""Risk-gated PC28 opportunity selection with explicit wait decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from .markets_combo import BASE_MARKETS, ExtremeRanges, ZeroPolicy, market_odds
from .portfolio_combo import BASE_COMBINATIONS, BASE_ONLY_EXTENSION_PAIRS, WAIT_BASE_ACTION_ID, PortfolioError, PortfolioLimits, STAKE_PROFILES, build_decision, extension_pairs
from .probability_combo import ComboProbabilityModel


@dataclass(frozen=True)
class OpportunityGateConfig:
    """Hard risk controls shared by simulation policies.

    Formal candidates require independently qualified probabilities and a
    strictly positive expected-return threshold.  Forced-rate or negative-EV
    diagnostics need an explicit non-Champion classification.
    """

    min_expected_return_ratio: float = 0.02
    min_base_coverage: float = 0.60
    max_drawdown: float = 0.35
    max_losing_streak: int = 3
    cooldown_rounds: int = 10
    target_bet_rate: float | None = None
    allow_negative_expected_return: bool = False
    require_probability_validation: bool = True
    experimental_non_champion: bool = False

    def __post_init__(self) -> None:
        if self.allow_negative_expected_return and not self.experimental_non_champion:
            object.__setattr__(self, "experimental_non_champion", True)
        if not isfinite(self.min_expected_return_ratio) or not -1 <= self.min_expected_return_ratio <= 1:
            raise ValueError("expected return threshold must be finite and within [-1, 1]")
        if self.min_expected_return_ratio <= 0 and not self.experimental_non_champion:
            raise ValueError("formal training requires a positive expected-return threshold")
        if self.min_expected_return_ratio < 0 and not self.allow_negative_expected_return:
            raise ValueError("negative expected-return thresholds require explicit experimental opt-in")
        if self.allow_negative_expected_return and not self.experimental_non_champion:
            raise ValueError("negative expected-return mode is experimental and cannot produce a Champion")
        if self.target_bet_rate is not None and not self.experimental_non_champion:
            raise ValueError("forced target_bet_rate is experimental and cannot produce a Champion")
        if not self.require_probability_validation and not self.experimental_non_champion:
            raise ValueError("formal training requires independent probability qualification")
        if not 0 <= self.min_base_coverage <= 1:
            raise ValueError("base coverage threshold must be a probability")
        if not 0 < self.max_drawdown <= 1 or self.max_losing_streak < 0 or self.cooldown_rounds <= 0:
            raise ValueError("risk thresholds are invalid")
        if self.target_bet_rate is not None and not 0 < self.target_bet_rate <= 1:
            raise ValueError("target_bet_rate must be within (0, 1]")


@dataclass(frozen=True)
class StatisticalOpportunity:
    wait: bool
    reason: str
    action: tuple[int, int, int, int] | None
    decision: dict[str, Any] | None
    expected_profit: float
    expected_return_ratio: float
    base_coverage: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "action": list(self.action) if self.action is not None else None}


def gate_model_action(
    action: Sequence[int],
    model: ComboProbabilityModel,
    *,
    features: Mapping[str, Any],
    balance: float,
    drawdown: float,
    losing_streak: int,
    probability_validation: Mapping[str, Any] | None,
    issue: str | None,
    as_of_issue: str | None,
    extreme_ranges: ExtremeRanges | None,
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    limits: PortfolioLimits = PortfolioLimits(),
    config: OpportunityGateConfig = OpportunityGateConfig(),
    extension_options: tuple[tuple[str, ...], ...] = BASE_ONLY_EXTENSION_PAIRS,
) -> StatisticalOpportunity:
    """Validate a learned action without allowing it to bypass risk controls."""

    probabilities = model.predict_markets(features, zero_policy=zero_policy, extreme_ranges=extreme_ranges)
    if int(action[0]) == WAIT_BASE_ACTION_ID:
        return _wait("model_requested_wait", probabilities)
    if config.require_probability_validation and not bool((probability_validation or {}).get("passed", False)):
        return _wait("probability_validation_not_predictive", probabilities)
    if drawdown > config.max_drawdown:
        return _wait("drawdown_limit", probabilities)
    if losing_streak >= config.max_losing_streak:
        return _wait("losing_streak_limit", probabilities)
    try:
        decision = build_decision(
            action=action,
            balance=balance,
            probabilities=probabilities,
            limits=limits,
            issue=issue,
            as_of_issue=as_of_issue,
            extension_options=extension_options,
        )
    except PortfolioError:
        return _wait("illegal_or_unfunded_action", probabilities)
    total_stake = sum(item.stake for item in decision.selections)
    base_coverage = sum(
        probabilities[item.market]
        for item in decision.selections
        if item.market in {base.value for base in BASE_MARKETS}
    )
    if base_coverage < config.min_base_coverage:
        return _wait("base_coverage_below_threshold", probabilities)
    expected_profit = sum(
        item.stake * (probabilities[item.market] * market_odds(item.market) - 1.0)
        for item in decision.selections
    )
    expected_return_ratio = expected_profit / max(total_stake, 1e-9)
    if expected_return_ratio < config.min_expected_return_ratio:
        return _wait("expected_return_below_threshold", probabilities)
    return StatisticalOpportunity(
        wait=False,
        reason="model_action_approved",
        action=tuple(int(value) for value in action),
        decision=decision.to_dict(),
        expected_profit=round(expected_profit, 8),
        expected_return_ratio=round(expected_return_ratio, 8),
        base_coverage=base_coverage,
        probabilities=probabilities,
    )


def choose_statistical_opportunity(
    model: ComboProbabilityModel,
    *,
    features: Mapping[str, Any],
    balance: float,
    drawdown: float,
    losing_streak: int,
    probability_validation: Mapping[str, Any] | None,
    issue: str | None = None,
    as_of_issue: str | None = None,
    extreme_ranges: ExtremeRanges | None,
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    limits: PortfolioLimits = PortfolioLimits(),
    config: OpportunityGateConfig = OpportunityGateConfig(),
    extension_options: tuple[tuple[str, ...], ...] = BASE_ONLY_EXTENSION_PAIRS,
) -> StatisticalOpportunity:
    """Select the highest predicted-value legal portfolio or an auditable wait.

    The policy cannot use realized outcome fields.  The frozen probability
    report is an audit input in random-baseline risk training, or an explicit
    eligibility requirement when ``require_probability_validation`` is true.
    """

    probabilities = model.predict_markets(features, zero_policy=zero_policy, extreme_ranges=extreme_ranges)
    if config.require_probability_validation and not bool((probability_validation or {}).get("passed", False)):
        return _wait("probability_validation_not_predictive", probabilities)
    if balance * limits.max_exposure_ratio < limits.min_stake * 3:
        return _wait("insufficient_balance", probabilities)
    if drawdown > config.max_drawdown:
        return _wait("drawdown_limit", probabilities)
    if losing_streak >= config.max_losing_streak:
        return _wait("losing_streak_limit", probabilities)

    best: StatisticalOpportunity | None = None
    for base_id, base in enumerate(BASE_COMBINATIONS):
        base_coverage = sum(probabilities[market] for market in base)
        if base_coverage < config.min_base_coverage:
            continue
        for extension_id, _extensions in enumerate(extension_options):
            for profile_id, _profile in enumerate(STAKE_PROFILES):
                action = (base_id, extension_id, profile_id, 0)
                try:
                    decision = build_decision(
                        action=action,
                        balance=balance,
                        probabilities=probabilities,
                        limits=limits,
                        issue=issue,
                        as_of_issue=as_of_issue,
                        extension_options=extension_options,
                    )
                except PortfolioError as exc:
                    if str(exc) == "balance cannot fund the minimum legal combo portfolio":
                        continue
                    raise
                expected_profit = sum(
                    selection.stake * (probabilities[selection.market] * market_odds(selection.market) - 1.0)
                    for selection in decision.selections
                )
                total_stake = sum(selection.stake for selection in decision.selections)
                candidate = StatisticalOpportunity(
                    wait=False,
                    reason="eligible",
                    action=action,
                    decision=decision.to_dict(),
                    expected_profit=round(expected_profit, 8),
                    expected_return_ratio=round(expected_profit / max(total_stake, 1e-9), 8),
                    base_coverage=round(base_coverage, 8),
                    probabilities=probabilities,
                )
                if best is None or candidate.expected_profit > best.expected_profit:
                    best = candidate
    if best is None:
        return _wait("base_coverage_below_threshold", probabilities)
    if best.expected_return_ratio < config.min_expected_return_ratio:
        return _wait("expected_return_below_threshold", probabilities, best)
    return best


def _wait(
    reason: str,
    probabilities: dict[str, float],
    candidate: StatisticalOpportunity | None = None,
) -> StatisticalOpportunity:
    return StatisticalOpportunity(
        wait=True,
        reason=reason,
        action=None,
        decision=None,
        expected_profit=0.0 if candidate is None else candidate.expected_profit,
        expected_return_ratio=0.0 if candidate is None else candidate.expected_return_ratio,
        base_coverage=0.0 if candidate is None else candidate.base_coverage,
        probabilities=probabilities,
    )
