"""Canonical PC28 combo-market definitions.

All odds in this module are gross odds: a winning return includes the stake.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final


COMBO_RULE_VERSION: Final[str] = "pc28-combo-v1"
COMBO_ODDS_VERSION: Final[str] = "user-odds-1"
COMBO_ACTION_SPACE_VERSION: Final[str] = "multidiscrete-base4-wait-base-only-profile-real-mode-v6"


class ComboMarketError(ValueError):
    """Raised when a market or result is outside the PC28 combo contract."""


class MarketType(str, Enum):
    BASE = "base"
    SUM_EXACT = "sum_exact"
    EXTREME = "extreme"


class BaseMarket(str, Enum):
    SMALL_ODD = "small_odd"
    SMALL_EVEN = "small_even"
    BIG_ODD = "big_odd"
    BIG_EVEN = "big_even"


class ExtremeMarket(str, Enum):
    EXTREME_SMALL = "extreme_small"
    EXTREME_BIG = "extreme_big"


class ZeroPolicy(str, Enum):
    EXCLUDE = "exclude"
    SMALL = "small"


BASE_MARKETS: Final[tuple[BaseMarket, ...]] = (
    BaseMarket.SMALL_ODD,
    BaseMarket.SMALL_EVEN,
    BaseMarket.BIG_ODD,
    BaseMarket.BIG_EVEN,
)
EXTREME_MARKETS: Final[tuple[ExtremeMarket, ...]] = (
    ExtremeMarket.EXTREME_SMALL,
    ExtremeMarket.EXTREME_BIG,
)
BASE_ODDS: Final[dict[BaseMarket, float]] = {
    BaseMarket.SMALL_ODD: 3.69,
    BaseMarket.BIG_EVEN: 3.69,
    BaseMarket.SMALL_EVEN: 4.29,
    BaseMarket.BIG_ODD: 4.29,
}
SUM_EXACT_ODDS: Final[dict[int, float]] = {
    0: 888.0,
    1: 300.0,
    2: 150.0,
    3: 90.0,
    4: 60.0,
    5: 45.0,
    6: 33.0,
    7: 27.0,
    8: 22.0,
    9: 18.0,
    10: 15.0,
    11: 14.0,
    12: 13.0,
    13: 11.0,
    14: 11.0,
    15: 13.0,
    16: 14.0,
    17: 15.0,
    18: 18.0,
    19: 22.0,
    20: 27.0,
    21: 33.0,
    22: 45.0,
    23: 60.0,
    24: 90.0,
    25: 150.0,
    26: 300.0,
    27: 888.0,
}
EXTREME_ODDS: Final[float] = 16.0
EXTENSION_MARKETS: Final[tuple[str, ...]] = tuple(
    [f"sum_{total:02d}" for total in range(28)]
    + [market.value for market in EXTREME_MARKETS]
)


@dataclass(frozen=True)
class ExtremeRanges:
    """Configured inclusive sum ranges for the two extreme markets.

    The product owner must approve these values.  No implicit legacy boundary is
    imported from the old pattern-market implementation.
    """

    small_min: int
    small_max: int
    big_min: int
    big_max: int

    def __post_init__(self) -> None:
        values = (self.small_min, self.small_max, self.big_min, self.big_max)
        if any(not isinstance(value, int) or value < 0 or value > 27 for value in values):
            raise ComboMarketError("extreme boundaries must be integer sums from 0 through 27")
        if self.small_min > self.small_max or self.big_min > self.big_max:
            raise ComboMarketError("extreme range minimum cannot exceed maximum")
        if max(self.small_min, self.big_min) <= min(self.small_max, self.big_max):
            raise ComboMarketError("extreme small and extreme big ranges cannot overlap")


CONFIRMED_EXTREME_RANGES: Final[ExtremeRanges] = ExtremeRanges(0, 5, 22, 27)
CONFIRMED_ZERO_POLICY: Final[ZeroPolicy] = ZeroPolicy.EXCLUDE


def extreme_ranges_metadata(extreme_ranges: ExtremeRanges | None) -> dict[str, int] | None:
    if extreme_ranges is None:
        return None
    return {
        "small_min": extreme_ranges.small_min,
        "small_max": extreme_ranges.small_max,
        "big_min": extreme_ranges.big_min,
        "big_max": extreme_ranges.big_max,
    }


def validate_sum(total: int) -> int:
    if isinstance(total, bool) or not isinstance(total, int) or not 0 <= total <= 27:
        raise ComboMarketError("PC28 sum must be an integer from 0 through 27")
    return total


def base_market_for_sum(total: int, *, zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE) -> BaseMarket | None:
    """Return the sole basic market that hits for a sum, or none for excluded 0."""

    total = validate_sum(total)
    if total == 0 and zero_policy == ZeroPolicy.EXCLUDE:
        return None
    if total <= 13:
        return BaseMarket.SMALL_ODD if total % 2 else BaseMarket.SMALL_EVEN
    return BaseMarket.BIG_ODD if total % 2 else BaseMarket.BIG_EVEN


def market_type(market: str) -> MarketType:
    if market in {item.value for item in BASE_MARKETS}:
        return MarketType.BASE
    if market in {f"sum_{total:02d}" for total in SUM_EXACT_ODDS}:
        return MarketType.SUM_EXACT
    if market in {item.value for item in EXTREME_MARKETS}:
        return MarketType.EXTREME
    raise ComboMarketError(f"unsupported combo market: {market}")


def market_odds(market: str) -> float:
    kind = market_type(market)
    if kind == MarketType.BASE:
        return BASE_ODDS[BaseMarket(market)]
    if kind == MarketType.SUM_EXACT:
        return SUM_EXACT_ODDS[int(market.removeprefix("sum_"))]
    return EXTREME_ODDS


def is_base_market(market: str) -> bool:
    return market_type(market) == MarketType.BASE


def is_extension_market(market: str) -> bool:
    return market_type(market) in {MarketType.SUM_EXACT, MarketType.EXTREME}


def market_hits(
    market: str,
    total: int,
    *,
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    extreme_ranges: ExtremeRanges | None = None,
) -> bool:
    """Apply the only allowed PC28 combo-market settlement rule."""

    total = validate_sum(total)
    kind = market_type(market)
    if kind == MarketType.BASE:
        return base_market_for_sum(total, zero_policy=zero_policy) == BaseMarket(market)
    if kind == MarketType.SUM_EXACT:
        return total == int(market.removeprefix("sum_"))
    if extreme_ranges is None:
        raise ComboMarketError("extreme ranges must be configured before extreme market settlement")
    if market == ExtremeMarket.EXTREME_SMALL.value:
        return extreme_ranges.small_min <= total <= extreme_ranges.small_max
    return extreme_ranges.big_min <= total <= extreme_ranges.big_max
