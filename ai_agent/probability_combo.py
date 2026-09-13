"""Time-ordered, calibrated next-sum probability model for PC28 combos.

This module deliberately predicts only the next outcome distribution.  It does
not choose stakes or orders; the opportunity gate consumes its frozen output in
a separate step.  Base and extreme probabilities are always derived from the
predicted ``sum_00`` through ``sum_27`` distribution.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .data_adapter import COMBO_FEATURE_COLUMNS, PROBABILITY_WINDOWS, clean_combo_history, load_history
from .markets_combo import ExtremeRanges, ZeroPolicy, base_market_for_sum


PROBABILITY_MODEL_VERSION = "pc28-next-sum-softmax-temperature-long-window-v2"
WINDOW_CANDIDATES: tuple[tuple[int, ...], ...] = (
    (100,),
    (200,),
    (400,),
    (800,),
    (1600,),
    (3200,),
    (400, 800),
    (800, 1600),
    (1600, 3200),
    (800, 1600, 3200),
    (3200, 4000),
    (4000, 6400),
    (1600, 3200, 6400),
)


@dataclass(frozen=True)
class ProbabilityTrainingConfig:
    calibration_fraction: float = 0.20
    epochs: int = 300
    learning_rate: float = 0.05
    l2_penalty: float = 0.001

    def __post_init__(self) -> None:
        if not 0.05 <= self.calibration_fraction < 0.5:
            raise ValueError("calibration_fraction must be in [0.05, 0.5)")
        if self.epochs <= 0 or self.learning_rate <= 0 or self.l2_penalty < 0:
            raise ValueError("probability training parameters must be positive")


@dataclass(frozen=True)
class ProbabilityValidationThresholds:
    min_brier_skill: float = 0.01
    max_expected_calibration_error: float = 0.05


@dataclass(frozen=True)
class ComboProbabilityModel:
    feature_columns: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...]
    temperature: float
    training_rows: int
    calibration_rows: int
    training_end_issue: str
    calibration_end_issue: str
    selected_windows: tuple[int, ...] = (100,)
    internal_selection: tuple[dict[str, Any], ...] = ()
    version: str = PROBABILITY_MODEL_VERSION

    def predict_sum_probabilities(self, features: Mapping[str, Any]) -> tuple[float, ...]:
        vector = np.asarray([float(features.get(column, 0.0)) for column in self.feature_columns], dtype=np.float64)
        means = np.asarray(self.feature_means, dtype=np.float64)
        scales = np.asarray(self.feature_scales, dtype=np.float64)
        logits = ((vector - means) / scales) @ np.asarray(self.weights, dtype=np.float64) + np.asarray(self.bias, dtype=np.float64)
        return tuple(float(value) for value in _softmax(logits / self.temperature))

    def predict_markets(
        self,
        features: Mapping[str, Any],
        *,
        zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
        extreme_ranges: ExtremeRanges | None = None,
    ) -> dict[str, float]:
        sums = self.predict_sum_probabilities(features)
        probabilities = {f"sum_{total:02d}": sums[total] for total in range(28)}
        for market in ("small_odd", "small_even", "big_odd", "big_even"):
            probabilities[market] = sum(
                probability
                for total, probability in enumerate(sums)
                if base_market_for_sum(total, zero_policy=zero_policy) is not None
                and base_market_for_sum(total, zero_policy=zero_policy).value == market
            )
        if extreme_ranges is not None:
            probabilities["extreme_small"] = sum(sums[extreme_ranges.small_min : extreme_ranges.small_max + 1])
            probabilities["extreme_big"] = sum(sums[extreme_ranges.big_min : extreme_ranges.big_max + 1])
        return probabilities

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComboProbabilityModel":
        return cls(
            feature_columns=tuple(str(value) for value in payload["feature_columns"]),
            feature_means=tuple(float(value) for value in payload["feature_means"]),
            feature_scales=tuple(float(value) for value in payload["feature_scales"]),
            weights=tuple(tuple(float(value) for value in row) for row in payload["weights"]),
            bias=tuple(float(value) for value in payload["bias"]),
            temperature=float(payload["temperature"]),
            training_rows=int(payload["training_rows"]),
            calibration_rows=int(payload["calibration_rows"]),
            training_end_issue=str(payload["training_end_issue"]),
            calibration_end_issue=str(payload["calibration_end_issue"]),
            selected_windows=tuple(int(value) for value in payload.get("selected_windows", (100,))),
            internal_selection=tuple(dict(item) for item in payload.get("internal_selection", ())),
            version=str(payload.get("version", PROBABILITY_MODEL_VERSION)),
        )


def fit_combo_probability_model(
    history: Sequence[dict[str, Any]] | Any,
    *,
    config: ProbabilityTrainingConfig = ProbabilityTrainingConfig(),
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
) -> ComboProbabilityModel:
    """Fit on the leading Train segment and calibrate on its chronological tail."""

    rows = clean_combo_history(history, zero_policy=zero_policy, min_rows=100)
    calibration_rows = max(1, int(len(rows) * config.calibration_fraction))
    fit_rows = rows[:-calibration_rows]
    calibration = rows[-calibration_rows:]
    if len(fit_rows) < 50:
        raise ValueError("probability fitting requires at least 50 rows before calibration")
    # Window selection must use only the fitting prefix.  Calibration labels are
    # reserved exclusively for temperature selection.
    selected_windows, selection_report = _select_windows(fit_rows, zero_policy=zero_policy)
    features = tuple(
        [*COMBO_FEATURE_COLUMNS,
         *(f"sum_rate_{total:02d}_{window}" for window in selected_windows for total in range(28))]
    )
    x_fit = _feature_matrix(fit_rows, features)
    means = x_fit.mean(axis=0)
    scales = np.maximum(x_fit.std(axis=0), 1e-6)
    x_fit = (x_fit - means) / scales
    target = np.asarray([int(row["sum"]) for row in fit_rows], dtype=np.int64)
    weights = np.zeros((len(features), 28), dtype=np.float64)
    bias = np.zeros(28, dtype=np.float64)
    expected = np.eye(28, dtype=np.float64)[target]
    for _ in range(config.epochs):
        predicted = _softmax(x_fit @ weights + bias)
        error = (predicted - expected) / len(x_fit)
        weights -= config.learning_rate * (x_fit.T @ error + config.l2_penalty * weights)
        bias -= config.learning_rate * error.sum(axis=0)

    calibration_logits = ((_feature_matrix(calibration, features) - means) / scales) @ weights + bias
    calibration_target = np.asarray([int(row["sum"]) for row in calibration], dtype=np.int64)
    temperature = _best_temperature(calibration_logits, calibration_target)
    return ComboProbabilityModel(
        feature_columns=features,
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        weights=tuple(tuple(float(value) for value in row) for row in weights),
        bias=tuple(float(value) for value in bias),
        temperature=temperature,
        training_rows=len(fit_rows),
        calibration_rows=len(calibration),
        training_end_issue=str(fit_rows[-1]["issue"]),
        calibration_end_issue=str(calibration[-1]["issue"]),
        selected_windows=selected_windows,
        internal_selection=tuple(selection_report),
    )


def fit_and_qualify_combo_probability_model(
    history: Sequence[dict[str, Any]] | Any,
    *,
    config: ProbabilityTrainingConfig = ProbabilityTrainingConfig(),
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    thresholds: ProbabilityValidationThresholds = ProbabilityValidationThresholds(),
) -> tuple[ComboProbabilityModel, dict[str, Any]]:
    """Fit, calibrate, then score on three strictly chronological Train slices.

    The qualification tail is never supplied to fitting, window selection, or
    temperature calibration.  Its result is eligibility evidence only; it does
    not update the returned frozen model.
    """

    rows = clean_combo_history(history, zero_policy=zero_policy, min_rows=100)
    fit_count = int(len(rows) * 0.80)
    calibration_count = int(len(rows) * 0.10)
    qualification_count = len(rows) - fit_count - calibration_count
    if fit_count < 50 or calibration_count < 1 or qualification_count < 1:
        raise ValueError("probability qualification requires fit, calibration, and qualification rows")
    fit_rows = rows[:fit_count]
    calibration_rows = rows[fit_count : fit_count + calibration_count]
    qualification_rows = rows[fit_count + calibration_count :]
    development_rows = fit_rows + calibration_rows
    development_config = replace(
        config,
        calibration_fraction=calibration_count / len(development_rows),
    )
    model = fit_combo_probability_model(
        development_rows,
        config=development_config,
        zero_policy=zero_policy,
    )
    report = probability_validation_report(
        model,
        qualification_rows,
        zero_policy=zero_policy,
        thresholds=thresholds,
    )
    report.update(
        {
            "phase": "probability_qualification",
            "independent_qualification": True,
            "slices": {
                "fit": _slice_metadata(fit_rows),
                "calibration": _slice_metadata(calibration_rows),
                "qualification": _slice_metadata(qualification_rows),
            },
            "model_parameter_sources": {
                "weights_and_bias": "fit",
                "feature_scaling": "fit",
                "window_selection": "fit",
                "temperature": "calibration",
                "qualification": "read_only_scoring",
            },
        }
    )
    return model, report


def probability_validation_report(
    model: ComboProbabilityModel,
    history: Sequence[dict[str, Any]] | Any,
    *,
    zero_policy: ZeroPolicy = ZeroPolicy.EXCLUDE,
    thresholds: ProbabilityValidationThresholds = ProbabilityValidationThresholds(),
) -> dict[str, Any]:
    """Evaluate frozen probabilities without updating any model parameter."""

    rows = _feature_rows(history, zero_policy=zero_policy, min_rows=1)
    predicted = np.asarray([model.predict_sum_probabilities(row["features"]) for row in rows], dtype=np.float64)
    baseline = np.asarray([_rolling_frequency_baseline(row["features"]) for row in rows], dtype=np.float64)
    target = np.asarray([int(row["sum"]) for row in rows], dtype=np.int64)
    one_hot = np.eye(28, dtype=np.float64)[target]
    brier = float(np.mean(np.sum((predicted - one_hot) ** 2, axis=1)))
    baseline_brier = float(np.mean(np.sum((baseline - one_hot) ** 2, axis=1)))
    log_loss = float(-np.mean(np.log(np.clip(predicted[np.arange(len(target)), target], 1e-12, 1.0))))
    ece = _expected_calibration_error(predicted, target)
    brier_skill = 1.0 - brier / max(baseline_brier, 1e-12)
    passed = brier_skill >= thresholds.min_brier_skill and ece <= thresholds.max_expected_calibration_error
    return {
        "phase": "probability_validation",
        "model_version": model.version,
        "rows": len(rows),
        "first_issue": rows[0]["issue"],
        "last_issue": rows[-1]["issue"],
        "brier_score": round(brier, 8),
        "baseline_brier_score": round(baseline_brier, 8),
        "brier_skill": round(brier_skill, 8),
        "log_loss": round(log_loss, 8),
        "expected_calibration_error": round(ece, 8),
        "thresholds": asdict(thresholds),
        "passed": passed,
    }


def save_probability_model(model: ComboProbabilityModel, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model.to_dict(), ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return output


def load_probability_model(path: str | Path) -> ComboProbabilityModel:
    return ComboProbabilityModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _feature_matrix(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row["features"].get(column, 0.0)) for column in columns] for row in rows], dtype=np.float64)


def _select_windows(rows: Sequence[dict[str, Any]], *, zero_policy: ZeroPolicy) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    """Choose window features on an earlier chronological slice only."""

    selection_end = min(len(rows), max(50, int(len(rows) * 0.8)))
    tune_start = max(20, int(selection_end * 0.8))
    tune_rows = rows[tune_start:selection_end]
    reports: list[dict[str, Any]] = []
    best: tuple[float, tuple[int, ...]] | None = None
    for windows in WINDOW_CANDIDATES:
        brier = 0.0
        for row in tune_rows:
            probabilities = [float(row["features"].get(f"sum_rate_{total:02d}_{window}", 0.0)) for window in windows for total in range(28)]
            blended = [sum(probabilities[offset + total] for offset in range(0, len(probabilities), 28)) / len(windows) for total in range(28)]
            target = int(row["sum"])
            brier += sum((blended[total] - (1.0 if total == target else 0.0)) ** 2 for total in range(28))
        score = brier / max(len(tune_rows), 1)
        reports.append({"windows": list(windows), "internal_brier_score": round(score, 8), "tune_first_issue": tune_rows[0]["issue"], "tune_last_issue": tune_rows[-1]["issue"]})
        candidate = (score, windows)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[1], reports


def _slice_metadata(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "first_issue": str(rows[0]["issue"]),
        "last_issue": str(rows[-1]["issue"]),
    }


def _feature_rows(
    history: Sequence[dict[str, Any]] | Any,
    *,
    zero_policy: ZeroPolicy,
    min_rows: int,
) -> list[dict[str, Any]]:
    """Preserve precomputed causal features when evaluating a Train tail."""

    rows = load_history(history)
    if rows and all(isinstance(row.get("features"), Mapping) and "sum" in row and "issue" in row for row in rows):
        if len(rows) < min_rows:
            raise ValueError(f"need at least {min_rows} probability rows")
        return list(rows)
    return clean_combo_history(rows, zero_policy=zero_policy, min_rows=min_rows)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(values)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _best_temperature(logits: np.ndarray, target: np.ndarray) -> float:
    candidates = np.linspace(0.5, 3.0, 51)
    losses = [float(-np.mean(np.log(np.clip(_softmax(logits / value)[np.arange(len(target)), target], 1e-12, 1.0)))) for value in candidates]
    return float(candidates[int(np.argmin(losses))])


def _rolling_frequency_baseline(features: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray([max(0.0, float(features.get(f"sum_rate_{total:02d}_100", 0.0))) for total in range(28)], dtype=np.float64)
    return values / values.sum() if values.sum() > 0 else np.full(28, 1.0 / 28.0)


def _expected_calibration_error(predicted: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    confidence = predicted.max(axis=1)
    correct = predicted.argmax(axis=1) == target
    result = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index < bins - 1 else confidence <= high)
        if mask.any():
            result += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return result
