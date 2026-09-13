"""Recover public PC28 results into a validated, non-destructive training file."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib import parse, request

from .data_adapter import load_history


SOURCE_URL = "https://pc28.ai/ajax.php"


class HistorySyncError(ValueError):
    pass


def fetch_day(day: date, *, timeout: float = 30.0) -> list[dict[str, Any]]:
    body = parse.urlencode(
        {
            "action": "GetOpenedData",
            "cztype": "xsjnd",
            "kaishishijian": f"{day.isoformat()} 00:00:00",
            "jieshushijian": f"{day.isoformat()} 23:59:59",
        }
    ).encode("utf-8")
    response = request.urlopen(
        request.Request(SOURCE_URL, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}),
        timeout=timeout,
    )
    payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise HistorySyncError(f"source returned a non-list payload for {day.isoformat()}")
    return [_normalize_remote(row) for row in payload]


def sync_history(
    source: str | Path,
    destination: str | Path,
    manifest_path: str | Path,
    *,
    end_day: date,
    pause_seconds: float = 2.1,
) -> dict[str, Any]:
    local_rows = [_normalize_local(row) for row in load_history(source)]
    _validate_rows(local_rows, label="local history", require_contiguous=True)
    start_day = date.fromisoformat(str(local_rows[-1]["time"])[:10])
    remote_rows: list[dict[str, Any]] = []
    current_day = start_day
    while current_day <= end_day:
        remote_rows.extend(fetch_day(current_day))
        current_day += timedelta(days=1)
        if current_day <= end_day:
            time.sleep(pause_seconds)

    remote_by_issue = _dedupe(remote_rows, label="remote history")
    local_by_issue = _dedupe(local_rows, label="local history")
    for issue, remote in remote_by_issue.items():
        local = local_by_issue.get(issue)
        if local is not None and _core_fields(local) != _core_fields(remote):
            raise HistorySyncError(f"overlapping issue {issue} conflicts with local history")
    merged = sorted({**local_by_issue, **remote_by_issue}.values(), key=lambda row: int(row["issue"]))
    _validate_rows(merged, label="merged history", require_contiguous=True)
    if int(merged[-1]["issue"]) != 3_478_112:
        raise HistorySyncError(f"expected latest issue 3478112, got {merged[-1]['issue']}")

    destination = Path(destination)
    manifest_path = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest = {
        "source": SOURCE_URL,
        "local_source": str(source),
        "output": str(destination),
        "synced_at": datetime.now().astimezone().isoformat(),
        "local_rows": len(local_rows),
        "fetched_rows": len(remote_rows),
        "merged_rows": len(merged),
        "first_issue": merged[0]["issue"],
        "last_issue": merged[-1]["issue"],
        "sha256": digest,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def fetch_latest_training_history(
    destination: str | Path,
    manifest_path: str | Path,
    *,
    start_day: date,
    end_day: date,
    expected_last_issue: int | None = None,
    keep_rows: int = 10_000,
    pause_seconds: float = 2.1,
) -> dict[str, Any]:
    """Build a source-consistent recent training set without merging conflicts."""

    if keep_rows <= 0 or start_day > end_day:
        raise ValueError("invalid training history date range or row count")
    rows: list[dict[str, Any]] = []
    current_day = start_day
    while current_day <= end_day:
        rows.extend(fetch_day(current_day))
        current_day += timedelta(days=1)
        if current_day <= end_day:
            time.sleep(pause_seconds)
    ordered = sorted(_dedupe(rows, label="remote training history").values(), key=lambda row: int(row["issue"]))
    _validate_rows(ordered, label="remote training history", require_contiguous=True)
    if expected_last_issue is not None and int(ordered[-1]["issue"]) != expected_last_issue:
        raise HistorySyncError(f"expected latest issue {expected_last_issue}, got {ordered[-1]['issue']}")
    if len(ordered) < keep_rows:
        raise HistorySyncError(f"source yielded {len(ordered)} rows, need {keep_rows}")
    training_rows = ordered[-keep_rows:]
    _validate_rows(training_rows, label="training history", require_contiguous=True)
    destination = Path(destination)
    manifest_path = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in training_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(destination)
    manifest = {
        "source": SOURCE_URL,
        "mode": "source_only_latest_training_window",
        "synced_at": datetime.now().astimezone().isoformat(),
        "fetched_rows": len(ordered),
        "training_rows": len(training_rows),
        "first_issue": training_rows[0]["issue"],
        "last_issue": training_rows[-1]["issue"],
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _normalize_remote(row: dict[str, Any]) -> dict[str, Any]:
    return _normalize(
        {
            "issue": row.get("id"),
            "digits": [row.get("yi"), row.get("er"), row.get("san")],
            "sum": row.get("he"),
            "time": row.get("shijian"),
            "source": "pc28.ai",
        }
    )


def _normalize_local(row: dict[str, Any]) -> dict[str, Any]:
    return _normalize(row)


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    issue = str(row.get("issue", "")).strip()
    digits = [int(value) for value in row.get("digits", row.get("numbers", []))]
    total = int(row.get("sum"))
    timestamp = str(row.get("time", "")).strip()
    if not issue.isdigit() or len(digits) != 3 or any(value < 0 or value > 9 for value in digits):
        raise HistorySyncError(f"invalid record: {row}")
    if sum(digits) != total or not 0 <= total <= 27 or len(timestamp) < 10:
        raise HistorySyncError(f"invalid digits/sum/time for issue {issue}")
    return {"issue": issue, "digits": digits, "sum": total, "time": timestamp}


def _dedupe(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        existing = result.get(row["issue"])
        if existing is not None and _core_fields(existing) != _core_fields(row):
            raise HistorySyncError(f"{label} conflicts within issue {row['issue']}")
        result[row["issue"]] = row
    return result


def _core_fields(row: dict[str, Any]) -> tuple[list[int], int]:
    return list(row["digits"]), int(row["sum"])


def _validate_rows(rows: list[dict[str, Any]], *, label: str, require_contiguous: bool) -> None:
    if not rows:
        raise HistorySyncError(f"{label} is empty")
    prior = None
    for row in rows:
        _normalize(row)
        current = int(row["issue"])
        if prior is not None and (current <= prior or (require_contiguous and current != prior + 1)):
            raise HistorySyncError(f"{label} has a duplicate or gap at issue {row['issue']}")
        prior = current


def main() -> None:
    parser = argparse.ArgumentParser(description="Append verified public PC28 history to a new training dataset.")
    parser.add_argument("--source", default="data/history.jsonl")
    parser.add_argument("--output", default="data/history_combo_synced.jsonl")
    parser.add_argument("--manifest", default="data/history_combo_synced.manifest.json")
    parser.add_argument("--end-day", default=date.today().isoformat())
    parser.add_argument("--pause-seconds", type=float, default=2.1)
    parser.add_argument("--source-only-latest", action="store_true")
    parser.add_argument("--start-day")
    parser.add_argument("--expected-last-issue", type=int)
    parser.add_argument("--keep-rows", type=int, default=10_000)
    args = parser.parse_args()
    if args.source_only_latest:
        if not args.start_day:
            parser.error("--start-day is required with --source-only-latest")
        result = fetch_latest_training_history(
            args.output,
            args.manifest,
            start_day=date.fromisoformat(args.start_day),
            end_day=date.fromisoformat(args.end_day),
            expected_last_issue=args.expected_last_issue,
            keep_rows=args.keep_rows,
            pause_seconds=args.pause_seconds,
        )
    else:
        result = sync_history(args.source, args.output, args.manifest, end_day=date.fromisoformat(args.end_day), pause_seconds=args.pause_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
