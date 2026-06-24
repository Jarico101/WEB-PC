from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .config import DEFAULT_INITIAL_BALANCE
from .file_policy import resolve_history_path


FEATURE_COLUMNS = [
    "current_omission",
    "pressure_score",
    "cond_straight_rate",
    "bot_win_rate",
    "pressure_delta",
    "omission_avg_ratio",
    "window_density_100",
    "relative_capital",
]


FORBIDDEN_FEATURE_KEYS = {
    "outcome",
    "outcome_is_straight",
    "is_straight",
    "straight",
    "digits",
    "numbers",
    "sum",
    "pattern",
    "result",
    "raw",
}


class DataAdapterError(ValueError):
    pass


def load_history(source: str | Path | Iterable[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = resolve_history_path(source)
        if not path.exists():
            raise DataAdapterError(f"history file not found: {path}")
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh))
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]
        if path.suffix.lower() == ".json":
            with path.open(encoding="utf-8") as fh:
                return _rows_from_payload(json.load(fh))
        raise DataAdapterError(f"unsupported history format: {path.suffix}")

    if isinstance(source, dict):
        return _rows_from_payload(source)
    return list(source)


def clean_history(
    source: str | Path | Iterable[dict[str, Any]] | dict[str, Any],
    *,
    initial_capital: float = DEFAULT_INITIAL_BALANCE,
    bot_win_rate: float = 0.945,
    min_rows: int = 10,
) -> list[dict[str, Any]]:
    rows = [_normalize_row(row) for row in load_history(source)]
    rows = [row for row in rows if row.get("outcome_is_straight") is not None]

    if len(rows) < min_rows:
        raise DataAdapterError(f"need at least {min_rows} usable rows, got {len(rows)}")

    cleaned: list[dict[str, Any]] = []
    pre_omissions: list[int] = []
    completed_gaps: list[int] = []
    current_omission = 0
    previous_pressure = 0.0
    straight_count_so_far = 0

    for index, row in enumerate(rows):
        prior_count = index
        prior_base_rate = straight_count_so_far / prior_count if prior_count else 0.055
        avg_gap = mean(completed_gaps) if completed_gaps else max(1.0, (1.0 / max(prior_base_rate, 0.001)) - 1.0)
        pressure_score = min(100.0, 100.0 * (current_omission / max(avg_gap * 2.0, 1.0)))

        features = {
            "current_omission": float(current_omission),
            "pressure_score": _round(pressure_score),
            "cond_straight_rate": _round(
                _conditional_rate(
                    rows,
                    pre_omissions,
                    current_omission,
                    prior_base_rate,
                    current_index=index,
                )
            ),
            "bot_win_rate": _round(bot_win_rate),
            "pressure_delta": _round(pressure_score - previous_pressure),
            "omission_avg_ratio": _round(current_omission / max(avg_gap, 1.0)),
            "window_density_100": _round(_window_density(rows, index, 100, current_index=index)),
            "relative_capital": 1.0,
        }

        cleaned_row = {
            **row,
            **features,
            "features": features,
            "initial_capital": float(initial_capital),
        }
        cleaned.append(cleaned_row)
        pre_omissions.append(current_omission)
        previous_pressure = pressure_score

        if row["outcome_is_straight"]:
            straight_count_so_far += 1
            completed_gaps.append(current_omission)
            current_omission = 0
        else:
            current_omission += 1

    return cleaned


def _rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("rows"), list):
        return payload["rows"]
    if isinstance(payload.get("history"), list):
        return payload["history"]
    if isinstance(payload.get("data"), list):
        return payload["data"]

    feature = payload.get("feature")
    if isinstance(feature, dict):
        if isinstance(feature.get("pressure_series"), list):
            return [
                {
                    "issue": point.get("issue"),
                    "is_straight": point.get("is_straight"),
                    "pressure_score": point.get("risk_score", point.get("value")),
                }
                for point in feature["pressure_series"]
            ]
        if isinstance(feature.get("recent_results"), list):
            return feature["recent_results"]

    raise DataAdapterError("payload does not contain rows/history/data/feature pressure_series")


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    _reject_leaky_features(row)
    digits = _parse_digits(row.get("digits") or row.get("numbers"))
    outcome = _coerce_bool(
        row.get("outcome_is_straight", row.get("is_straight", row.get("straight")))
    )
    if outcome is None and digits:
        outcome = _is_straight_digits(digits)

    pressure_score = _first_float(
        row,
        "pressure_score",
        "risk_score",
        "pressure",
        "pressureIdx",
        "value",
    )
    if pressure_score is not None and pressure_score <= 1.0:
        pressure_score *= 100.0

    return {
        "issue": str(row.get("issue", "")).strip(),
        "time": row.get("time"),
        "digits": digits,
        "sum": _coerce_float(row.get("sum")),
        "outcome_is_straight": outcome,
        "pressure_score": pressure_score,
        "bot_win_rate": _first_float(row, "bot_win_rate", "non_straight_hit_rate"),
        "straight_profit_multiplier": _first_float(row, "straight_profit_multiplier"),
        "non_straight_profit_multiplier": _first_float(row, "non_straight_profit_multiplier"),
        "straight_odds": _first_float(row, "straight_odds"),
        "non_straight_odds": _first_float(row, "non_straight_odds"),
        "raw": row,
    }


def _reject_leaky_features(row: dict[str, Any]) -> None:
    features = row.get("features")
    if not isinstance(features, dict):
        features = row.get("feature")
    if not isinstance(features, dict):
        return
    leaked = sorted(FORBIDDEN_FEATURE_KEYS.intersection(features.keys()))
    if leaked:
        raise DataAdapterError(f"forbidden future/result feature keys: {', '.join(leaked)}")


def _conditional_rate(
    rows: list[dict[str, Any]],
    pre_omissions: list[int],
    current_omission: int,
    fallback: float,
    *,
    current_index: int,
) -> float:
    assert current_index == len(pre_omissions), "pre_omissions must end before the current row"
    assert 0 <= current_index <= len(rows), "current_index out of history bounds"
    matches = [
        rows[past_index]["outcome_is_straight"]
        for index, omission in enumerate(pre_omissions)
        for past_index in (index,)
        if past_index < current_index
        if omission == current_omission
    ]
    if not matches:
        return fallback
    return sum(1 for is_straight in matches if is_straight) / len(matches)


def _window_density(rows: list[dict[str, Any]], index: int, window_size: int, *, current_index: int) -> float:
    assert index == current_index, "window index must equal the current row index"
    assert 0 <= current_index <= len(rows), "current_index out of history bounds"
    start = max(0, current_index - window_size)
    history_indices = [past_index for past_index in range(start, current_index) if past_index < current_index]
    assert all(past_index < current_index for past_index in history_indices), "future row would leak into window"
    sample = [rows[past_index] for past_index in history_indices]
    if not sample:
        return 0.0
    return sum(1 for row in sample if row["outcome_is_straight"]) / len(sample)


def _first_float(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _coerce_float(row.get(key))
        if value is not None:
            return value
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "straight", "顺", "顺子"}:
        return True
    if text in {"false", "0", "no", "n", "non_straight", "非顺"}:
        return False
    return None


def _parse_digits(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    if isinstance(value, str):
        cleaned = value.replace("|", ",").replace(" ", ",")
        return [int(part) for part in cleaned.split(",") if part.strip()]
    return []


def _is_straight_digits(digits: list[int]) -> bool:
    uniq = sorted(set(digits))
    if len(uniq) < 3:
        return False
    for index in range(len(uniq) - 2):
        if uniq[index + 2] - uniq[index] == 2:
            return True
    return False


def _round(value: float) -> float:
    return round(float(value), 12)
