"""Local, simulation-only online settlement ledger for combo decisions."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .markets_combo import ExtremeRanges, ZeroPolicy
from .portfolio_combo import PortfolioLimits, validate_decision
from .schemas_combo import ComboDecision, ComboSettlement
from .settlement_combo import settle_combo


class OnlineLedgerError(ValueError):
    pass


@dataclass
class ComboOnlineLedger:
    balance: float = 10_000.0
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE
    extreme_ranges: ExtremeRanges | None = None
    limits: PortfolioLimits = field(default_factory=PortfolioLimits)
    log_path: Path | None = None
    pending: dict[str, ComboDecision] = field(default_factory=dict)
    settled_issues: set[str] = field(default_factory=set)
    decision_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.balance < 0:
            raise OnlineLedgerError("ledger balance cannot be negative")
        if self.log_path is not None:
            self.log_path = Path(self.log_path)

    def submit(self, decision: ComboDecision) -> None:
        issue = str(decision.issue or "").strip()
        if not issue:
            raise OnlineLedgerError("pending combo decision requires an issue")
        validate_decision(decision, limits=self.limits)
        decision = _with_decision_id(decision)
        if issue in self.pending or issue in self.settled_issues:
            raise OnlineLedgerError(f"duplicate combo issue: {issue}")
        if decision.decision_id in self.decision_ids:
            raise OnlineLedgerError(f"duplicate combo decision id: {decision.decision_id}")
        self.pending[issue] = decision
        self.decision_ids.add(str(decision.decision_id))
        self._append_event("decision", {"decision": decision.to_dict(), "balance_before": self.balance})

    def settle(self, issue: str, outcome_sum: int) -> ComboSettlement:
        issue = str(issue).strip()
        if issue in self.settled_issues:
            raise OnlineLedgerError(f"issue already settled: {issue}")
        decision = self.pending.pop(issue, None)
        if decision is None:
            raise OnlineLedgerError(f"no pending decision for issue: {issue}")
        result = settle_combo(
            decision,
            outcome_sum=outcome_sum,
            balance_before=self.balance,
            zero_policy=self.zero_policy,
            extreme_ranges=self.extreme_ranges,
            limits=self.limits,
        )
        self.balance = result.balance_after
        self.settled_issues.add(issue)
        self._append_event("settlement", result.to_dict())
        return result

    def restore(self) -> None:
        """Recover the last balance and settled issues from an existing JSONL log."""

        if self.log_path is None or not self.log_path.exists():
            return
        with self.log_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                event = payload.get("event", "settlement" if "balance_after" in payload else "")
                decision_payload = payload.get("decision")
                if not isinstance(decision_payload, dict):
                    continue
                issue = str(decision_payload.get("issue", "")).strip()
                if not issue:
                    continue
                if event == "decision" and issue not in self.settled_issues:
                    decision = _with_decision_id(decision_from_dict(decision_payload))
                    if decision.decision_id not in self.decision_ids:
                        self.pending[issue] = decision
                        self.decision_ids.add(str(decision.decision_id))
                elif event == "settlement":
                    self.pending.pop(issue, None)
                    self.settled_issues.add(issue)
                    if "balance_after" in payload:
                        self.balance = float(payload["balance_after"])

    def _append_event(self, event: str, payload: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True) + "\n")


def decision_from_dict(payload: dict[str, Any]) -> ComboDecision:
    from .schemas_combo import Selection

    selections = tuple(
        Selection(
            market=str(item["market"]),
            market_type=str(item["market_type"]),
            role=str(item["role"]),
            stake=float(item["stake"]),
            odds=float(item["odds"]),
            probability=float(item["probability"]),
            hit=item.get("hit"),
        )
        for item in payload.get("selections", [])
    )
    return ComboDecision(
        selections=selections,
        mode=str(payload["mode"]),
        issue=payload.get("issue"),
        model_version=payload.get("model_version"),
        as_of_issue=payload.get("as_of_issue"),
        feature_snapshot_hash=payload.get("feature_snapshot_hash"),
        decision_id=payload.get("decision_id"),
    )


def _with_decision_id(decision: ComboDecision) -> ComboDecision:
    if decision.decision_id:
        return decision
    payload = decision.to_dict()
    payload.pop("decision_id", None)
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return replace(decision, decision_id=sha256(encoded.encode("utf-8")).hexdigest()[:24])
