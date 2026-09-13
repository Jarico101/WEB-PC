"""Action decoding and hard portfolio constraints for PC28 combo simulation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from itertools import combinations
from math import isfinite
import random
from typing import Mapping, Sequence

from .markets_combo import BASE_MARKETS, EXTENSION_MARKETS, EXTREME_MARKETS, ExtremeRanges, is_base_market, is_extension_market, market_odds, market_type
from .schemas_combo import ComboDecision, DecisionMode, Selection, SelectionRole


class PortfolioError(ValueError):
    pass


BASE_COMBINATIONS: tuple[tuple[str, str, str], ...] = tuple(
    tuple(market.value for market in combo) for combo in combinations(BASE_MARKETS, 3)
)
# A wait has no selections.  It is encoded on the base axis so a betting action
# still always starts from one of the four legal three-base portfolios.
WAIT_BASE_ACTION_ID = len(BASE_COMBINATIONS)
EXTENSION_PAIRS: tuple[tuple[str, ...], ...] = ((),) + tuple((market,) for market in EXTENSION_MARKETS) + tuple(
    pair for pair in combinations(EXTENSION_MARKETS, 2)
)
BASE_ONLY_EXTENSION_PAIRS: tuple[tuple[str, ...], ...] = ((),)
# This restricted extension set is intentionally separate from the formal
# action space.  It is for explicitly labelled, non-Champion random simulation
# only: at most one of the relatively low-house-edge sums may be appended.
EXPERIMENTAL_SUM_EXTENSION_PAIRS: tuple[tuple[str, ...], ...] = (
    (),
    ("sum_08",),
    ("sum_09",),
    ("sum_18",),
    ("sum_19",),
)
STAKE_PROFILES: tuple[str, ...] = ("defense", "balanced", "attack")
# PPO and DQN may only produce a real virtual wager.  Shadow remains available
# for explicit offline analysis, but cannot be selected by a learned action.
TRAINING_MODE_IDS: tuple[str, ...] = (
    DecisionMode.DEFENSE.value,
    DecisionMode.BALANCED.value,
    DecisionMode.ATTACK.value,
)
MODE_IDS: tuple[str, ...] = tuple(mode.value for mode in DecisionMode)


@dataclass(frozen=True)
class StakeAllocationConfig:
    """Virtual-round budget fraction for each selectable stake profile."""

    defense_base_share: float = 0.50
    balanced_base_share: float = 0.75
    attack_base_share: float = 1.00

    def __post_init__(self) -> None:
        if any(not 0 < value <= 1 for value in self._values().values()):
            raise PortfolioError("stake profile base shares must be within (0, 1]")

    def base_share(self, profile: str) -> float:
        try:
            return self._values()[profile]
        except KeyError as exc:
            raise PortfolioError(f"unsupported stake profile: {profile}") from exc

    def _values(self) -> dict[str, float]:
        return {
            "defense": self.defense_base_share,
            "balanced": self.balanced_base_share,
            "attack": self.attack_base_share,
        }


@dataclass(frozen=True)
class PortfolioLimits:
    max_exposure_ratio: float = 1.0
    max_round_stake: float = 300.0
    max_total_selections: int = 5
    min_stake: float = 30.0
    max_stake: float = 300.0
    stake_allocations: StakeAllocationConfig = StakeAllocationConfig()

    def __post_init__(self) -> None:
        if not 0 < self.max_exposure_ratio <= 1:
            raise PortfolioError("max_exposure_ratio must be in (0, 1]")
        if not isfinite(self.max_round_stake) or self.max_round_stake <= 0:
            raise PortfolioError("max_round_stake must be positive and finite")
        if self.max_total_selections != 5:
            raise PortfolioError("PC28 combo supports exactly 3 through 5 total selections")
        if not isfinite(self.min_stake) or self.min_stake <= 0:
            raise PortfolioError("min_stake must be positive and finite")
        if not isfinite(self.max_stake) or self.max_stake < self.min_stake:
            raise PortfolioError("max_stake must be finite and at least min_stake")


def extension_pairs(
    *,
    extreme_ranges: ExtremeRanges | None = None,
    extensions_enabled: bool = False,
) -> tuple[tuple[str, ...], ...]:
    """Return enabled extension combinations, or the base-only contract by default."""

    if not extensions_enabled:
        return BASE_ONLY_EXTENSION_PAIRS

    disabled = {market.value for market in EXTREME_MARKETS} if extreme_ranges is None else set()
    return tuple(pair for pair in EXTENSION_PAIRS if not disabled.intersection(pair))


def decode_action(
    action: Sequence[int],
    *,
    extension_options: tuple[tuple[str, ...], ...] = BASE_ONLY_EXTENSION_PAIRS,
    mode_options: tuple[str, ...] = TRAINING_MODE_IDS,
) -> tuple[tuple[str, str, str], tuple[str, ...], str, str]:
    """Decode a four-axis action without representing illegal extension duplicates.

    Action layout: ``[base_combo_id, extension_pair_id, stake_profile_id, mode_id]``.
    """

    if len(action) != 4:
        raise PortfolioError("combo action must contain [base_combo, extensions, stake_profile, mode]")
    base_id, extension_id, profile_id, mode_id = (int(value) for value in action)
    if not 0 <= base_id < len(BASE_COMBINATIONS):
        raise PortfolioError("base combo action index out of range")
    if not 0 <= extension_id < len(extension_options):
        raise PortfolioError("extension action index out of range")
    if not 0 <= profile_id < len(STAKE_PROFILES):
        raise PortfolioError("stake profile action index out of range")
    if not 0 <= mode_id < len(mode_options):
        raise PortfolioError("mode action index out of range")
    try:
        return (
            BASE_COMBINATIONS[base_id],
            extension_options[extension_id],
            STAKE_PROFILES[profile_id],
            mode_options[mode_id],
        )
    except IndexError as exc:
        raise PortfolioError("combo action index out of range") from exc


def action_nvec(
    *,
    extreme_ranges: ExtremeRanges | None = None,
    include_shadow: bool = False,
    extensions_enabled: bool = False,
) -> tuple[int, int, int, int]:
    modes = MODE_IDS if include_shadow else TRAINING_MODE_IDS
    return (
        len(BASE_COMBINATIONS) + 1,
        len(extension_pairs(extreme_ranges=extreme_ranges, extensions_enabled=extensions_enabled)),
        len(STAKE_PROFILES),
        len(modes),
    )


def is_wait_action(action: Sequence[int]) -> bool:
    """Return whether an action is the one canonical no-order action."""

    if len(action) != 4:
        return False
    base_id, extension_id, profile_id, mode_id = (int(value) for value in action)
    return (base_id, extension_id, profile_id, mode_id) == (WAIT_BASE_ACTION_ID, 0, 0, 0)


def validate_decision(decision: ComboDecision, *, limits: PortfolioLimits = PortfolioLimits()) -> None:
    if decision.mode not in MODE_IDS:
        raise PortfolioError(f"unsupported decision mode: {decision.mode}")
    if not 3 <= len(decision.selections) <= limits.max_total_selections:
        raise PortfolioError("a combo decision must contain 3 to 5 selections")
    base = [selection for selection in decision.selections if is_base_market(selection.market)]
    extensions = [selection for selection in decision.selections if is_extension_market(selection.market)]
    if len(base) != 3 or len({selection.market for selection in base}) != 3:
        raise PortfolioError("a combo decision must contain exactly three distinct base markets")
    if len(extensions) > 2 or len({selection.market for selection in extensions}) != len(extensions):
        raise PortfolioError("a combo decision may contain at most two distinct extension markets")
    if len(base) + len(extensions) != len(decision.selections):
        raise PortfolioError("a combo decision contains an unsupported market")
    for selection in decision.selections:
        if selection.market_type != market_type(selection.market).value:
            raise PortfolioError("selection market_type does not match market")
        if selection.role not in {role.value for role in SelectionRole}:
            raise PortfolioError("selection role is unsupported")
        if (
            not isfinite(selection.stake)
            or selection.stake < limits.min_stake
            or selection.stake > limits.max_stake
            or float(selection.stake) != int(selection.stake)
        ):
            raise PortfolioError("each selection stake must be an integer within min_stake and max_stake")
        if selection.odds != market_odds(selection.market):
            raise PortfolioError("selection odds does not match canonical odds")
        if not isfinite(selection.probability) or not 0 <= selection.probability <= 1:
            raise PortfolioError("selection probability must be within [0, 1]")


def build_decision(
    *,
    action: Sequence[int],
    balance: float,
    probabilities: Mapping[str, float] | None = None,
    limits: PortfolioLimits = PortfolioLimits(),
    issue: str | None = None,
    model_version: str | None = None,
    as_of_issue: str | None = None,
    extension_options: tuple[tuple[str, ...], ...] = BASE_ONLY_EXTENSION_PAIRS,
    mode_options: tuple[str, ...] = TRAINING_MODE_IDS,
) -> ComboDecision:
    """Construct a legal budget-bounded decision from a discrete model action."""

    if not isfinite(balance) or balance <= 0:
        raise PortfolioError("balance must be positive and finite")
    base_id, extension_id, profile_id, mode_id = (int(value) for value in action)
    base, extensions, profile, mode = decode_action(
        action,
        extension_options=extension_options,
        mode_options=mode_options,
    )
    budget = min(balance * limits.max_exposure_ratio, limits.max_round_stake)
    if budget < limits.min_stake * (len(base) + len(extensions)):
        raise PortfolioError("balance cannot fund the minimum legal combo portfolio")
    selections: list[Selection] = []
    probability_map = probabilities or {}

    minimum_stake = Decimal(str(limits.min_stake))
    selection_count = len(base) + len(extensions)
    maximum_budget = int(Decimal(str(budget)).to_integral_value(rounding=ROUND_DOWN))
    minimum_budget = int(minimum_stake) * selection_count
    planned_budget = int(maximum_budget * limits.stake_allocations.base_share(profile))
    available_budget = max(minimum_budget, planned_budget)
    if available_budget > maximum_budget:
        raise PortfolioError("balance cannot fund the minimum legal combo portfolio")
    all_markets = (*base, *extensions)
    stakes = _mode_integer_stakes(
        available_budget,
        len(all_markets),
        int(limits.min_stake),
        int(limits.max_stake),
        mode=mode,
        seed=f"{issue}:{base_id}:{extension_id}:{profile_id}:{mode_id}",
    )
    for index, market in enumerate(base):
        role = SelectionRole.PRIMARY.value if index == 0 else SelectionRole.DEFENSE.value
        selections.append(_selection(market, role, stakes[index], probability_map))
    for index, market in enumerate(extensions):
        selections.append(_selection(market, SelectionRole.ATTACK.value, stakes[len(base) + index], probability_map))

    decision = ComboDecision(
        selections=tuple(selections),
        mode=mode,
        issue=issue,
        model_version=model_version,
        as_of_issue=as_of_issue,
    )
    validate_decision(decision, limits=limits)
    if sum(selection.stake for selection in decision.selections) > budget + 1e-9:
        raise PortfolioError("rounded stakes exceed the one-round budget")
    return decision


def _selection(market: str, role: str, stake: float, probabilities: Mapping[str, float]) -> Selection:
    probability = float(probabilities.get(market, 0.0))
    return Selection(
        market=market,
        market_type=market_type(market).value,
        role=role,
        stake=stake,
        odds=market_odds(market),
        probability=probability,
    )


def _split_stakes(amount: Decimal | float, count: int) -> tuple[float, ...]:
    """Split a monetary amount to cents without rounding above its allocation."""

    if count == 0:
        return ()
    cents = (Decimal(str(amount)) * 100).to_integral_value(rounding=ROUND_DOWN)
    whole, remainder = divmod(int(cents), count)
    return tuple((whole + (1 if index < remainder else 0)) / 100 for index in range(count))


def _split_integer_stakes(amount: Decimal | float, count: int, minimum: int) -> tuple[float, ...]:
    """Split a whole-dollar amount into integer stakes without violating minimums."""

    if count == 0:
        return ()
    whole = int(Decimal(str(amount)).to_integral_value(rounding=ROUND_DOWN))
    if whole < minimum * count:
        raise PortfolioError("balance cannot fund the minimum legal combo portfolio")
    quotient, remainder = divmod(whole, count)
    return tuple(float(quotient + (1 if index < remainder else 0)) for index in range(count))


def _random_integer_stakes(total: int, count: int, minimum: int, maximum: int, *, seed: str) -> tuple[float, ...]:
    """Distribute the fixed round total randomly within per-item bounds."""

    if count <= 0 or total < minimum * count or total > maximum * count:
        raise PortfolioError("round budget cannot satisfy the selected stake bounds")
    rng = random.Random(seed)
    stakes = [minimum] * count
    remaining = total - minimum * count
    for index in range(count - 1):
        capacity = maximum - minimum
        low = max(0, remaining - capacity * (count - index - 1))
        high = min(capacity, remaining)
        allocation = rng.randint(low, high)
        stakes[index] += allocation
        remaining -= allocation
    stakes[-1] += remaining
    rng.shuffle(stakes)
    return tuple(float(value) for value in stakes)


def _mode_integer_stakes(
    total: int,
    count: int,
    minimum: int,
    maximum: int,
    *,
    mode: str,
    seed: str,
) -> tuple[float, ...]:
    """Make the risk-mode axis change stake concentration, not only metadata."""

    if mode == DecisionMode.DEFENSE.value:
        return _split_integer_stakes(total, count, minimum)
    if mode in {DecisionMode.BALANCED.value, DecisionMode.SHADOW.value}:
        return _random_integer_stakes(total, count, minimum, maximum, seed=seed)
    if mode == DecisionMode.ATTACK.value:
        if total < minimum * count or total > maximum * count:
            raise PortfolioError("round budget cannot satisfy the selected stake bounds")
        # The first selected base market is the primary market.  Attack keeps
        # the other two legal defensive selections at their floor and gives the
        # primary most of the remaining permitted budget.
        primary_capacity = maximum - minimum
        primary_extra = min(primary_capacity, (total - minimum * count) * 2 // 3)
        remaining_total = total - minimum - primary_extra
        tail = _random_integer_stakes(
            remaining_total,
            count - 1,
            minimum,
            maximum,
            seed=seed,
        )
        return (float(minimum + primary_extra), *tail)
    raise PortfolioError(f"unsupported decision mode: {mode}")
