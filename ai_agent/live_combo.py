"""Simulation-only synchronizer for live PC28 results and combo decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .data_adapter import clean_combo_history
from .challenger_combo import ChallengerConfig, challenger_due
from .infer_combo import predict_combo_decision
from .online_combo import ComboOnlineLedger, OnlineLedgerError, decision_from_dict


ComboPredictor = Callable[..., dict[str, Any]]


@dataclass
class ComboLiveSynchronizer:
    ledger: ComboOnlineLedger
    model_path: str
    algorithm: str = "ppo"
    predictor: ComboPredictor = predict_combo_decision
    challenger_config: ChallengerConfig = ChallengerConfig()

    def process_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        rows = clean_combo_history(payload, zero_policy=self.ledger.zero_policy, min_rows=1)
        return self.process_rows(rows, prediction_source=payload)

    def process_rows(self, rows: list[dict[str, Any]], *, prediction_source: Any) -> dict[str, Any]:
        if not rows:
            raise ValueError("live combo synchronization requires at least one result row")
        diagnostics = issue_diagnostics(rows)
        settled: list[str] = []
        duplicates: list[str] = []
        for row in rows:
            issue = str(row["issue"])
            if issue in self.ledger.pending:
                self.ledger.settle(issue, int(row["sum"]))
                settled.append(issue)
            elif issue in self.ledger.settled_issues:
                duplicates.append(issue)

        prediction = None
        if not diagnostics["has_gap"]:
            latest = rows[-1]
            prediction = self.predictor(
                prediction_source,
                model_path=self.model_path,
                algorithm=self.algorithm,
                balance=self.ledger.balance,
                zero_policy=self.ledger.zero_policy,
                extreme_ranges=self.ledger.extreme_ranges,
            )
            target_issue = str(prediction["issue"])
            if prediction.get("wait", False):
                pass
            elif target_issue not in self.ledger.pending and target_issue not in self.ledger.settled_issues:
                self.ledger.submit(decision_from_dict(prediction["decision"]))
            else:
                duplicates.append(target_issue)
        return {
            "dry_run": True,
            "latest_issue": rows[-1]["issue"],
            "latest_time": rows[-1].get("time"),
            "settled_issues": settled,
            "duplicate_issues": sorted(set(duplicates)),
            "pending_issues": sorted(self.ledger.pending),
            "balance": self.ledger.balance,
            "prediction": prediction,
            "diagnostics": diagnostics,
            "challenger_due": challenger_due(len(self.ledger.settled_issues), config=self.challenger_config),
            "challenger_settled_rounds": len(self.ledger.settled_issues),
        }


def issue_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Surface duplicate/non-numeric/gapped live issue sequences without guessing."""

    issues = [str(row.get("issue", "")) for row in rows]
    duplicates = sorted({issue for issue in issues if issues.count(issue) > 1})
    numeric: list[int] = []
    non_numeric: list[str] = []
    for issue in issues:
        if issue.isdigit():
            numeric.append(int(issue))
        else:
            non_numeric.append(issue)
    missing_ranges: list[str] = []
    for previous, current in zip(numeric, numeric[1:]):
        if current <= previous:
            continue
        if current - previous > 1:
            start, end = previous + 1, current - 1
            missing_ranges.append(str(start) if start == end else f"{start}-{end}")
    return {
        "has_gap": bool(missing_ranges or non_numeric),
        "missing_ranges": missing_ranges,
        "duplicate_issues": duplicates,
        "non_numeric_issues": non_numeric,
        "checked": len(rows),
    }
