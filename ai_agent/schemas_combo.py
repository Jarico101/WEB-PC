"""Typed, JSON-friendly records shared by combo training, APIs and backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DecisionMode(str, Enum):
    DEFENSE = "defense"
    BALANCED = "balanced"
    ATTACK = "attack"
    SHADOW = "shadow"


class SelectionRole(str, Enum):
    DEFENSE = "defense"
    PRIMARY = "primary"
    ATTACK = "attack"
    BALANCED = "balanced"


@dataclass(frozen=True)
class Selection:
    market: str
    market_type: str
    role: str
    stake: float
    odds: float
    probability: float
    hit: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComboDecision:
    selections: tuple[Selection, ...]
    mode: str
    issue: str | None = None
    model_version: str | None = None
    as_of_issue: str | None = None
    feature_snapshot_hash: str | None = None
    decision_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selections"] = [selection.to_dict() for selection in self.selections]
        return payload


@dataclass(frozen=True)
class SettlementItem:
    selection: Selection
    hit: bool
    return_amount: float
    net_contribution: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selection"] = self.selection.to_dict()
        return payload


@dataclass(frozen=True)
class ComboSettlement:
    decision: ComboDecision
    outcome_sum: int
    items: tuple[SettlementItem, ...]
    total_stake: float
    total_return: float
    net_profit: float
    balance_before: float
    balance_after: float
    balance_delta: float
    is_shadow: bool
    shadow_net_profit: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.to_dict()
        payload["items"] = [item.to_dict() for item in self.items]
        return payload


@dataclass(frozen=True)
class EpisodeMetrics:
    final_balance: float
    max_drawdown: float
    longest_losing_streak: int
    total_exposure: float
    executed_net_profit: float
    shadow_net_profit: float


@dataclass(frozen=True)
class ModelMetadata:
    rule_version: str
    odds_version: str
    action_space_version: str
    training_data_end_issue: str
    extra: dict[str, Any] = field(default_factory=dict)


# Public contract name used by API and task-plan consumers.
Settlement = ComboSettlement
