"""Validation-only Champion selection and sealed-test confirmation for combo models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .evaluation_combo import evaluate_combo_model
from .markets_combo import CONFIRMED_EXTREME_RANGES, CONFIRMED_ZERO_POLICY
from .training_protocol_combo import validate_combo_split_manifest


class ChampionSelectionError(ValueError):
    pass


def evaluate_candidate_on_validation(
    model_path: str | Path,
    *,
    algorithm: str,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Evaluate exactly on Validation; Test is intentionally not read here."""

    manifest = validate_combo_split_manifest(manifest_path, include_test=False)
    train = _split_path(manifest, manifest_path, "train")
    validation = _split_path(manifest, manifest_path, "validation")
    return {
        "phase": "validation",
        "model_path": str(model_path),
        "algorithm": algorithm.lower(),
        "evaluation": evaluate_combo_model(
            _concat_jsonl(train, validation),
            model_path=model_path,
            algorithm=algorithm,
            window_size=200,
            windows=5,
            zero_policy=CONFIRMED_ZERO_POLICY,
            extreme_ranges=CONFIRMED_EXTREME_RANGES,
        ),
    }


def select_validation_champion(candidate_reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Rank only eligible Validation reports. Test results cannot enter this function."""

    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for report in candidate_reports:
        if report.get("phase") != "validation":
            raise ChampionSelectionError("Champion selection accepts Validation reports only")
        evaluation = report.get("evaluation", {})
        if evaluation.get("verdict", {}).get("status") != "PASS":
            rejected.append({"model_path": report.get("model_path"), "reason": "validation_gate_failed"})
            continue
        eligible.append({**report, "score": validation_score(evaluation)})
    eligible.sort(key=lambda item: item["score"], reverse=True)
    if not eligible:
        return {"status": "NO_CHAMPION", "phase": "validation", "selected": None, "rejected": rejected}
    selected = eligible[0]
    return {
        "status": "SELECTED_PENDING_TEST",
        "phase": "validation",
        "selected": {key: selected[key] for key in ("model_path", "algorithm", "score")},
        "ranked_eligible": [{key: item[key] for key in ("model_path", "algorithm", "score")} for item in eligible],
        "rejected": rejected,
    }


def confirm_selected_champion_on_test(
    selection: dict[str, Any],
    *,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Perform the single sealed Test confirmation after Validation selection."""

    if selection.get("status") != "SELECTED_PENDING_TEST" or not isinstance(selection.get("selected"), dict):
        raise ChampionSelectionError("a Validation-selected candidate is required before Test confirmation")
    manifest = validate_combo_split_manifest(manifest_path)
    train = _split_path(manifest, manifest_path, "train")
    validation = _split_path(manifest, manifest_path, "validation")
    test = _split_path(manifest, manifest_path, "test")
    candidate = selection["selected"]
    evaluation = evaluate_combo_model(
        _concat_jsonl(train, validation, test),
        model_path=candidate["model_path"],
        algorithm=candidate["algorithm"],
        window_size=200,
        windows=5,
        zero_policy=CONFIRMED_ZERO_POLICY,
        extreme_ranges=CONFIRMED_EXTREME_RANGES,
    )
    return {
        "status": "CONFIRMED_CHAMPION" if evaluation["verdict"]["status"] == "PASS" else "NO_CHAMPION",
        "phase": "test",
        "automatic_runtime_activation": False,
        "selected": candidate,
        "evaluation": evaluation,
    }


def validation_score(evaluation: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return the deterministic Validation ranking score used for Champion selection."""

    windows = evaluation.get("walk_forward", [])
    relative_profit = sum(
        float(window["agent"].get("net_profit", 0.0))
        - max((float(metrics.get("net_profit", 0.0)) for metrics in window.get("baselines", {}).values()), default=0.0)
        for window in windows
    )
    max_drawdown = max((float(window["agent"].get("max_drawdown", 1.0)) for window in windows), default=1.0)
    max_exposure = max((float(window["agent"].get("max_exposure_ratio", 1.0)) for window in windows), default=1.0)
    extension_share = max((float(window["agent"].get("extension_positive_profit_share", 1.0)) for window in windows), default=1.0)
    return (round(relative_profit, 6), -round(max_drawdown, 6), -round(max_exposure, 6), -round(extension_share, 6))


def _split_path(manifest: dict[str, Any], manifest_path: str | Path, name: str) -> Path:
    raw = Path(manifest["splits"][name]["path"])
    return raw if raw.is_absolute() else Path(manifest_path).parent / raw.name


def _concat_jsonl(*paths: Path) -> list[dict[str, Any]]:
    import json

    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows
