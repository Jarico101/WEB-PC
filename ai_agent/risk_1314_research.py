"""Train-only causal risk research for PC28 sums 13 and 14.

This module is deliberately limited to the first 8,000 Train rows.  It does
not read Validation or the sealed Test split and cannot create a Champion.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data_adapter import clean_combo_history, load_history
from .probability_combo import _expected_calibration_error
from .training_protocol_combo import validate_train_tuning_chunks


EVENTS = ("sum_13", "sum_14", "sum_13_or_14")
THEORETICAL_RISK = {"sum_13": 0.075, "sum_14": 0.075, "sum_13_or_14": 0.15}


@dataclass(frozen=True)
class Risk1314ResearchConfig:
    fit_rows: int = 6_400
    calibration_rows: int = 800
    qualification_rows: int = 800
    research_warmup_rows: int = 400
    research_window_rows: int = 500
    research_folds: int = 12
    frequency_windows: tuple[int, ...] = (50, 100, 200, 400, 800, 1600, 3200)
    shrinkage_candidates: tuple[tuple[int, float], ...] = ((400, 200.0), (800, 400.0), (1600, 800.0), (3200, 1600.0))
    min_group_rows: int = 50

    def __post_init__(self) -> None:
        if self.fit_rows <= 0 or self.calibration_rows <= 0 or self.qualification_rows <= 0:
            raise ValueError("all Train slices must be positive")
        if self.research_warmup_rows <= 0 or self.research_window_rows <= 0 or self.research_folds <= 0:
            raise ValueError("research windows must be positive")
        if self.research_warmup_rows + self.research_window_rows * self.research_folds != self.fit_rows:
            raise ValueError("research folds must exactly cover the Fit slice after warmup")


def run_train_only_risk_research(
    train_chunk_manifest_path: str | Path,
    *,
    config: Risk1314ResearchConfig = Risk1314ResearchConfig(),
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Research, select, calibrate, and independently qualify a 13/14 model."""

    manifest = validate_train_tuning_chunks(train_chunk_manifest_path)
    rows = _load_train_rows(manifest, train_chunk_manifest_path)
    cleaned = clean_combo_history(rows, min_rows=config.fit_rows + config.calibration_rows + config.qualification_rows)
    expected_rows = config.fit_rows + config.calibration_rows + config.qualification_rows
    if len(cleaned) != expected_rows:
        raise ValueError(f"expected exactly {expected_rows} Train rows, got {len(cleaned)}")

    candidates = _candidate_names(config)
    research: dict[str, list[dict[str, Any]]] = {name: [] for name in candidates}
    research_folds: list[dict[str, Any]] = []
    fit_rows = cleaned[: config.fit_rows]
    for fold in range(config.research_folds):
        start = config.research_warmup_rows + fold * config.research_window_rows
        score_rows = fit_rows[start : start + config.research_window_rows]
        targets = _target_matrix(score_rows)
        baseline = _prediction_matrix(score_rows, "rolling_frequency_100", config)
        fold_meta = {
            "fold": fold + 1,
            "training_rows_before_score": start,
            "score_rows": len(score_rows),
            "score_first_issue": score_rows[0]["issue"],
            "score_last_issue": score_rows[-1]["issue"],
        }
        research_folds.append(fold_meta)
        for name in candidates:
            prediction = _prediction_matrix(score_rows, name, config)
            research[name].append(_score_prediction(prediction, baseline, targets, config.min_group_rows))

    aggregate = {name: _aggregate(name, folds) for name, folds in research.items()}
    selected_name = max(aggregate, key=lambda name: (aggregate[name]["mean_brier_skill"], -aggregate[name]["mean_ece"]))

    calibration = cleaned[config.fit_rows : config.fit_rows + config.calibration_rows]
    qualification = cleaned[config.fit_rows + config.calibration_rows :]
    calibration_prediction = _prediction_matrix(calibration, selected_name, config)
    calibration_thresholds = _risk_thresholds(calibration_prediction, _target_matrix(calibration))
    qualification_prediction = _prediction_matrix(qualification, selected_name, config)
    qualification_baseline = _prediction_matrix(qualification, "rolling_frequency_100", config)
    qualification_scores = _score_prediction(
        qualification_prediction,
        qualification_baseline,
        _target_matrix(qualification),
        config.min_group_rows,
        thresholds=calibration_thresholds,
    )
    qualification_passed = _qualification_passed(qualification_scores)
    report = {
        "schema_version": "pc28-train-only-risk-1314-v1",
        "phase": "risk_1314_train_only_research_and_qualification",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classification": "EXPERIMENTAL_NON_CHAMPION",
        "input": {
            "train_chunk_manifest": str(train_chunk_manifest_path),
            "train_rows": len(cleaned),
            "first_issue": cleaned[0]["issue"],
            "last_issue": cleaned[-1]["issue"],
            "validation_accessed": False,
            "test_accessed": False,
        },
        "slices": {
            "fit": _slice_meta(cleaned[: config.fit_rows]),
            "calibration": _slice_meta(calibration),
            "qualification": _slice_meta(qualification),
        },
        "labels": {
            "events": list(EVENTS),
            "definition": "sum equals 13, sum equals 14, or sum equals 13 or 14",
            "theoretical_probability": THEORETICAL_RISK,
        },
        "config": asdict(config),
        "research_folds": research_folds,
        "candidates": aggregate,
        "selection": {
            "selected_candidate": selected_name,
            "selection_source": "12 chronological Fit-only future windows",
            "calibration_thresholds": calibration_thresholds,
        },
        "qualification": {
            "independent": True,
            "model_parameter_sources": {
                "candidate_selection": "Fit-only research folds",
                "risk_thresholds": "Calibration only",
                "qualification": "frozen read-only scoring",
            },
            "scores": qualification_scores,
            "passed": qualification_passed,
            "status": "RISK_MODEL_QUALIFIED" if qualification_passed else "RISK_MODEL_QUALIFICATION_FAIL",
        },
        "result": {
            "status": "RISK_MODEL_QUALIFIED" if qualification_passed else "RISK_MODEL_QUALIFICATION_FAIL",
            "ppo_allowed": qualification_passed,
            "formal_training_started": False,
            "champion_eligible": False,
            "validation_accessed": False,
            "test_accessed": False,
        },
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _candidate_names(config: Risk1314ResearchConfig) -> tuple[str, ...]:
    names = [f"rolling_frequency_{window}" for window in config.frequency_windows]
    names.extend(f"shrunk_frequency_{window}_theory_{int(strength)}" for window, strength in config.shrinkage_candidates)
    return tuple(names)


def _load_train_rows(manifest: Mapping[str, Any], manifest_path: str | Path) -> list[dict[str, Any]]:
    root = Path(manifest_path).parent
    rows: list[dict[str, Any]] = []
    for chunk in manifest["chunks"]:
        raw = Path(chunk["path"])
        path = raw if raw.is_absolute() else root / raw.name
        rows.extend(load_history(path))
    return rows


def _prediction_matrix(rows: Sequence[dict[str, Any]], name: str, config: Risk1314ResearchConfig) -> np.ndarray:
    if name.startswith("rolling_frequency_"):
        window = int(name.removeprefix("rolling_frequency_"))
        strength = 0.0
    else:
        _, _, window_text, _, strength_text = name.split("_")
        window, strength = int(window_text), float(strength_text)
    result = []
    for row in rows:
        features = row["features"]
        values = np.asarray([max(0.0, float(features.get(f"sum_rate_{total:02d}_{window}", 0.0))) for total in range(28)])
        if values.sum() <= 0:
            values = np.full(28, 1.0 / 28.0)
        else:
            values /= values.sum()
        if strength:
            values = (window / (window + strength)) * values + (strength / (window + strength)) * _theoretical_sum()
        result.append((values[13], values[14], values[13] + values[14]))
    return np.asarray(result, dtype=np.float64)


def _theoretical_sum() -> np.ndarray:
    return np.asarray([count / 1000.0 for count in (1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 63, 69, 73, 75, 75, 73, 69, 63, 55, 45, 36, 28, 21, 15, 10, 6, 3, 1)], dtype=np.float64)


def _target_matrix(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    sums = np.asarray([int(row["sum"]) for row in rows])
    return np.column_stack((sums == 13, sums == 14, (sums == 13) | (sums == 14))).astype(np.float64)


def _score_prediction(
    prediction: np.ndarray,
    baseline: np.ndarray,
    targets: np.ndarray,
    min_group_rows: int,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for index, event in enumerate(EVENTS):
        actual = targets[:, index]
        probabilities = prediction[:, index]
        base = baseline[:, index]
        brier = float(np.mean((probabilities - actual) ** 2))
        baseline_brier = float(np.mean((base - actual) ** 2))
        threshold = float(thresholds[event]) if thresholds else float(np.quantile(probabilities, 2 / 3))
        high = probabilities >= threshold
        low = ~high
        scores[event] = {
            "brier_score": _round(brier),
            "baseline_brier_score": _round(baseline_brier),
            "brier_skill": _round(1.0 - brier / max(baseline_brier, 1e-12)),
            "log_loss": _round(float(-np.mean(actual * np.log(np.clip(probabilities, 1e-12, 1.0)) + (1 - actual) * np.log(np.clip(1 - probabilities, 1e-12, 1.0))))),
            "expected_calibration_error": _round(_binary_ece(probabilities, actual)),
            "auc": _round(_auc(actual, probabilities)),
            "threshold": _round(threshold),
            "high_group": {"rows": int(high.sum()), "rate": _round(float(actual[high].mean())) if high.any() else None},
            "low_group": {"rows": int(low.sum()), "rate": _round(float(actual[low].mean())) if low.any() else None},
            "high_low_rate_difference": _round(float(actual[high].mean() - actual[low].mean())) if high.any() and low.any() else None,
            "groups_have_minimum_rows": bool(high.sum() >= min_group_rows and low.sum() >= min_group_rows),
        }
    return scores


def _aggregate(name: str, folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "folds": list(folds)}
    skills = []
    eces = []
    for event in EVENTS:
        event_scores = [fold[event] for fold in folds]
        skill = np.asarray([item["brier_skill"] for item in event_scores], dtype=float)
        ece = np.asarray([item["expected_calibration_error"] for item in event_scores], dtype=float)
        differences = [item["high_low_rate_difference"] for item in event_scores if item["high_low_rate_difference"] is not None]
        result[event] = {
            "mean_brier_skill": _round(float(skill.mean())),
            "median_brier_skill": _round(float(np.median(skill))),
            "mean_ece": _round(float(ece.mean())),
            "positive_skill_fold_rate": _round(float((skill > 0).mean())),
            "positive_high_low_rate_fold_rate": _round(float((np.asarray(differences) > 0).mean())) if differences else 0.0,
        }
        skills.append(float(skill.mean()))
        eces.append(float(ece.mean()))
    result["mean_brier_skill"] = _round(float(np.mean(skills)))
    result["mean_ece"] = _round(float(np.mean(eces)))
    result["stable_direction"] = all(result[event]["positive_high_low_rate_fold_rate"] >= 0.75 for event in EVENTS)
    return result


def _risk_thresholds(prediction: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for index, event in enumerate(EVENTS):
        actual = targets[:, index]
        values = prediction[:, index]
        candidates = np.unique(np.quantile(values, np.linspace(0.5, 0.9, 9)))
        best = max(candidates, key=lambda threshold: _high_low_difference(actual, values >= threshold))
        thresholds[event] = float(best)
    return thresholds


def _qualification_passed(scores: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        float(scores[event]["brier_skill"]) >= 0.01
        and float(scores[event]["expected_calibration_error"]) <= 0.05
        and float(scores[event]["auc"]) >= 0.5
        and bool(scores[event]["groups_have_minimum_rows"])
        and float(scores[event]["high_low_rate_difference"]) > 0
        for event in EVENTS
    )


def _high_low_difference(actual: np.ndarray, high: np.ndarray) -> float:
    low = ~high
    return float(actual[high].mean() - actual[low].mean()) if high.any() and low.any() else -1.0


def _binary_ece(probabilities: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (probabilities >= low) & ((probabilities < high) if index < bins - 1 else (probabilities <= high))
        if mask.any():
            result += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(actual[mask].mean()))
    return result


def _auc(actual: np.ndarray, probabilities: np.ndarray) -> float:
    positives = actual == 1
    negatives = actual == 0
    if not positives.any() or not negatives.any():
        return 0.5
    order = np.argsort(probabilities, kind="stable")
    ranks = np.empty(len(probabilities), dtype=float)
    ranks[order] = np.arange(1, len(probabilities) + 1)
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) / (positives.sum() * negatives.sum()))


def _slice_meta(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {"rows": len(rows), "first_issue": rows[0]["issue"], "last_issue": rows[-1]["issue"]}


def _round(value: float) -> float:
    return round(float(value), 8)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
