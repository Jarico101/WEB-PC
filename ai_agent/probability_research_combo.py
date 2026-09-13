"""Train-only, chronological probability-model comparison for PC28.

This is a diagnostic research runner, not a betting trainer.  It never opens
the Validation or sealed Test split and cannot create a Champion.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .data_adapter import clean_combo_history, load_history
from .probability_combo import ProbabilityTrainingConfig, _expected_calibration_error, fit_combo_probability_model
from .training_protocol_combo import validate_train_tuning_chunks


THEORETICAL_SUM_PROBABILITIES: tuple[float, ...] = tuple(
    float(count) / 1000.0
    for count in (1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 63, 69, 73, 75, 75, 73, 69, 63, 55, 45, 36, 28, 21, 15, 10, 6, 3, 1)
)


@dataclass(frozen=True)
class ProbabilityResearchConfig:
    """Fixed candidate set and chronological 500-issue internal evaluation."""

    first_training_rows: int = 2_000
    evaluation_rows: int = 500
    baseline_window: int = 100
    frequency_windows: tuple[int, ...] = (800, 1600, 3200)
    shrinkage_candidates: tuple[tuple[int, float], ...] = ((1600, 800.0), (3200, 1600.0))
    include_softmax: bool = True
    probability_config: ProbabilityTrainingConfig = ProbabilityTrainingConfig()

    def __post_init__(self) -> None:
        if self.first_training_rows < 100 or self.evaluation_rows <= 0 or self.baseline_window <= 0:
            raise ValueError("research windows must be positive and leave a valid training prefix")
        if any(window <= 0 for window in self.frequency_windows):
            raise ValueError("frequency windows must be positive")
        if any(window <= 0 or strength < 0 for window, strength in self.shrinkage_candidates):
            raise ValueError("shrinkage windows and strengths must be non-negative")


PredictorFactory = Callable[[Sequence[dict[str, Any]]], Callable[[dict[str, Any]], np.ndarray]]


def run_train_internal_probability_research(
    train_chunk_manifest_path: str | Path,
    *,
    config: ProbabilityResearchConfig = ProbabilityResearchConfig(),
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Score predeclared predictors over future Train-only 500-issue windows.

    Each fold trains (when applicable) only on the chronological prefix before
    its score window.  Scores never feed into predictor parameters; the report
    is research evidence, not a hyperparameter selection or qualification pass.
    """

    manifest = validate_train_tuning_chunks(train_chunk_manifest_path)
    rows = _load_train_rows(manifest, train_chunk_manifest_path)
    cleaned = clean_combo_history(rows, min_rows=config.first_training_rows + config.evaluation_rows)
    candidate_factories = _candidate_factories(config)
    fold_starts = range(config.first_training_rows, len(cleaned) - config.evaluation_rows + 1, config.evaluation_rows)
    candidate_results: dict[str, list[dict[str, Any]]] = {name: [] for name in candidate_factories}
    folds: list[dict[str, Any]] = []

    for fold_number, start in enumerate(fold_starts, start=1):
        train_rows = cleaned[:start]
        score_rows = cleaned[start : start + config.evaluation_rows]
        baseline = _feature_probability_matrix(score_rows, config.baseline_window)
        target = _targets(score_rows)
        baseline_brier = _brier(baseline, target)
        folds.append(
            {
                "fold": fold_number,
                "training_rows": len(train_rows),
                "training_first_issue": train_rows[0]["issue"],
                "training_last_issue": train_rows[-1]["issue"],
                "score_rows": len(score_rows),
                "score_first_issue": score_rows[0]["issue"],
                "score_last_issue": score_rows[-1]["issue"],
                "baseline": {"name": f"rolling_frequency_{config.baseline_window}", "brier_score": _round(baseline_brier)},
            }
        )
        for name, factory in candidate_factories.items():
            predictor = factory(train_rows)
            prediction = np.asarray([predictor(row) for row in score_rows], dtype=np.float64)
            brier = _brier(prediction, target)
            candidate_results[name].append(
                {
                    "fold": fold_number,
                    "brier_score": _round(brier),
                    "baseline_brier_score": _round(baseline_brier),
                    "brier_skill": _round(1.0 - brier / baseline_brier),
                    "expected_calibration_error": _round(_expected_calibration_error(prediction, target)),
                }
            )

    candidates = {name: _aggregate_candidate(name, results) for name, results in candidate_results.items()}
    qualified = [item for item in candidates.values() if item["all_folds_passed"]]
    report = {
        "schema_version": "pc28-train-only-probability-research-v1",
        "phase": "train_internal_probability_research",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "diagnostic_only_not_eligible_for_champion_or_test",
        "input": {
            "train_chunk_manifest": str(train_chunk_manifest_path),
            "train_rows": len(cleaned),
            "first_issue": cleaned[0]["issue"],
            "last_issue": cleaned[-1]["issue"],
            "validation_accessed": False,
            "test_accessed": False,
        },
        "config": _config_dict(config),
        "folds": folds,
        "candidates": candidates,
        "result": {
            "status": "RESEARCH_SIGNAL_FOUND" if qualified else "NO_STABLE_PREDICTIVE_SIGNAL",
            "qualified_candidates": [item["name"] for item in qualified],
            "formal_training_started": False,
            "test_accessed": False,
        },
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _candidate_factories(config: ProbabilityResearchConfig) -> dict[str, PredictorFactory]:
    factories: dict[str, PredictorFactory] = {
        "theoretical_three_uniform_digits": lambda _train: lambda _row: np.asarray(THEORETICAL_SUM_PROBABILITIES),
    }
    for window in config.frequency_windows:
        factories[f"rolling_frequency_{window}"] = lambda _train, window=window: lambda row: _feature_probability(row, window)
    for window, strength in config.shrinkage_candidates:
        factories[f"shrunk_frequency_{window}_theory_{int(strength)}"] = (
            lambda _train, window=window, strength=strength: lambda row: _shrunken_feature_probability(row, window, strength)
        )
    if config.include_softmax:
        factories["softmax_long_window"] = lambda train: _softmax_predictor(train, config.probability_config)
    return factories


def _softmax_predictor(train_rows: Sequence[dict[str, Any]], config: ProbabilityTrainingConfig) -> Callable[[dict[str, Any]], np.ndarray]:
    model = fit_combo_probability_model(train_rows, config=config)
    return lambda row: np.asarray(model.predict_sum_probabilities(row["features"]), dtype=np.float64)


def _load_train_rows(manifest: dict[str, Any], manifest_path: str | Path) -> list[dict[str, Any]]:
    root = Path(manifest_path).parent
    rows: list[dict[str, Any]] = []
    for chunk in manifest["chunks"]:
        raw = Path(chunk["path"])
        path = raw if raw.is_absolute() else root / raw.name
        rows.extend(load_history(path))
    return rows


def _feature_probability_matrix(rows: Sequence[dict[str, Any]], window: int) -> np.ndarray:
    return np.asarray([_feature_probability(row, window) for row in rows], dtype=np.float64)


def _feature_probability(row: dict[str, Any], window: int) -> np.ndarray:
    values = np.asarray([max(0.0, float(row["features"].get(f"sum_rate_{total:02d}_{window}", 0.0))) for total in range(28)])
    return values / values.sum() if values.sum() else np.asarray(THEORETICAL_SUM_PROBABILITIES)


def _shrunken_feature_probability(row: dict[str, Any], window: int, strength: float) -> np.ndarray:
    empirical = _feature_probability(row, window)
    weight = window / (window + strength)
    return weight * empirical + (1.0 - weight) * np.asarray(THEORETICAL_SUM_PROBABILITIES)


def _targets(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["sum"]) for row in rows], dtype=np.int64)


def _brier(probabilities: np.ndarray, target: np.ndarray) -> float:
    one_hot = np.eye(28, dtype=np.float64)[target]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _aggregate_candidate(name: str, folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    skill = np.asarray([float(fold["brier_skill"]) for fold in folds])
    ece = np.asarray([float(fold["expected_calibration_error"]) for fold in folds])
    passed = (skill >= 0.01) & (ece <= 0.05)
    return {
        "name": name,
        "folds": list(folds),
        "mean_brier_skill": _round(float(skill.mean())),
        "median_brier_skill": _round(float(np.median(skill))),
        "mean_expected_calibration_error": _round(float(ece.mean())),
        "pass_rate": _round(float(passed.mean())),
        "all_folds_passed": bool(passed.all()),
    }


def _config_dict(config: ProbabilityResearchConfig) -> dict[str, Any]:
    result = asdict(config)
    result["probability_config"] = asdict(config.probability_config)
    return result


def _round(value: float) -> float:
    return round(float(value), 8)
