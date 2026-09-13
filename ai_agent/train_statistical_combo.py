"""Train and evaluate the probability-first PC28 statistical candidate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .data_adapter import clean_combo_history
from .markets_combo import CONFIRMED_EXTREME_RANGES, CONFIRMED_ZERO_POLICY, ExtremeRanges, ZeroPolicy
from .probability_combo import (
    ProbabilityTrainingConfig,
    ProbabilityValidationThresholds,
    fit_and_qualify_combo_probability_model,
    save_probability_model,
)
from .statistical_backtest_combo import run_statistical_combo_backtest
from .statistical_combo_policy import OpportunityGateConfig


def train_and_evaluate_statistical_combo(
    train_history: Sequence[dict[str, Any]] | Any,
    validation_history: Sequence[dict[str, Any]] | Any,
    *,
    model_path: str | Path | None = None,
    training_config: ProbabilityTrainingConfig = ProbabilityTrainingConfig(),
    validation_thresholds: ProbabilityValidationThresholds = ProbabilityValidationThresholds(),
    gate_config: OpportunityGateConfig = OpportunityGateConfig(),
    zero_policy: ZeroPolicy = CONFIRMED_ZERO_POLICY,
    extreme_ranges: ExtremeRanges | None = CONFIRMED_EXTREME_RANGES,
) -> dict[str, Any]:
    """Fit/calibrate only Train, then make frozen decisions on Validation."""

    train_rows = clean_combo_history(train_history, zero_policy=zero_policy, min_rows=100)
    validation_rows = clean_combo_history(validation_history, zero_policy=zero_policy, min_rows=1)
    model, qualification = fit_and_qualify_combo_probability_model(
        train_rows,
        config=training_config,
        zero_policy=zero_policy,
        thresholds=validation_thresholds,
    )
    if model_path is not None:
        save_probability_model(model, model_path)
    # Rebuilding features over the concatenated history preserves Train as the
    # only pre-Validation context while each Validation feature uses prior rows.
    evaluation = run_statistical_combo_backtest(
        train_rows + validation_rows,
        model=model,
        probability_calibration=qualification,
        initial_balance=10_000.0,
        start_step=len(train_rows),
        zero_policy=zero_policy,
        extreme_ranges=extreme_ranges,
        gate_config=gate_config,
    )
    return {
        "phase": "statistical_validation",
        "model": model.to_dict(),
        "probability_qualification": qualification,
        "probability_calibration": qualification,
        "validation": evaluation,
        "test_accessed": False,
    }
