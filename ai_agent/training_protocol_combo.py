"""Leak-resistant 8:1:1 data protocol for PC28 combo candidates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .data_adapter import clean_combo_history


class ComboTrainingProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ComboSplitConfig:
    total_rows: int = 10_000
    train_rows: int = 8_000
    validation_rows: int = 1_000
    test_rows: int = 1_000

    def __post_init__(self) -> None:
        if min(self.total_rows, self.train_rows, self.validation_rows, self.test_rows) <= 0:
            raise ComboTrainingProtocolError("split row counts must be positive")
        if self.train_rows + self.validation_rows + self.test_rows != self.total_rows:
            raise ComboTrainingProtocolError("train + validation + test must equal total_rows")


@dataclass(frozen=True)
class TrainChunkConfig:
    """Chronological micro-experiment schedule inside the sealed Train split."""

    train_rows: int = 8_000
    chunk_rows: int = 500
    min_training_chunks: int = 2

    def __post_init__(self) -> None:
        if self.train_rows <= 0 or self.chunk_rows <= 0 or self.train_rows % self.chunk_rows:
            raise ComboTrainingProtocolError("train_rows must be divisible by a positive chunk_rows")
        if not 1 <= self.min_training_chunks < self.chunk_count:
            raise ComboTrainingProtocolError("min_training_chunks must leave at least one evaluation chunk")

    @property
    def chunk_count(self) -> int:
        return self.train_rows // self.chunk_rows


def create_combo_splits(
    source: str | Path | Iterable[dict[str, Any]] | dict[str, Any],
    *,
    output_dir: str | Path = "data",
    config: ComboSplitConfig = ComboSplitConfig(),
) -> dict[str, Any]:
    """Create chronological split files without shuffling or overlap."""

    rows = clean_combo_history(source, min_rows=config.total_rows)
    rows = rows[-config.total_rows :]
    _assert_contiguous(rows)
    train = rows[: config.train_rows]
    validation = rows[config.train_rows : config.train_rows + config.validation_rows]
    test = rows[config.train_rows + config.validation_rows :]
    if (
        int(train[-1]["issue"]) >= int(validation[0]["issue"])
        or int(validation[-1]["issue"]) >= int(test[0]["issue"])
    ):
        raise ComboTrainingProtocolError("split issue order is invalid")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": root / "history_combo_train_8000.jsonl",
        "validation": root / "history_combo_validation_1000.jsonl",
        "test": root / "history_combo_test_1000.jsonl",
    }
    metadata = {
        "protocol": "pc28-combo-chronological-8-1-1-v1",
        "source": str(source),
        "config": asdict(config),
        "splits": {},
        "test_sealed": True,
    }
    for name, split_rows in (("train", train), ("validation", validation), ("test", test)):
        _write_jsonl(paths[name], split_rows)
        metadata["splits"][name] = _split_metadata(paths[name], split_rows)
    manifest_path = root / "history_combo_8_1_1.manifest.json"
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {**metadata, "manifest_path": str(manifest_path), "paths": {name: str(path) for name, path in paths.items()}}


def create_train_tuning_chunks(
    manifest_path: str | Path,
    *,
    output_dir: str | Path = "data/train_chunks_500",
    config: TrainChunkConfig = TrainChunkConfig(),
) -> dict[str, Any]:
    """Split the Train segment into chronological chunks for gradual tuning.

    The returned folds use only chunks before their evaluation chunk as model
    input.  Validation and Test are not read by this function.
    """

    manifest = validate_combo_split_manifest(manifest_path, include_test=False)
    train_metadata = manifest["splits"]["train"]
    if int(train_metadata["rows"]) != config.train_rows:
        raise ComboTrainingProtocolError("Train row count does not match the chunk protocol")
    source = Path(train_metadata["path"])
    if not source.is_absolute():
        source = Path(manifest_path).parent / source.name
    rows = clean_combo_history(source, min_rows=config.train_rows)
    _assert_contiguous(rows)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    chunks: list[dict[str, Any]] = []
    for index in range(config.chunk_count):
        start = index * config.chunk_rows
        end = start + config.chunk_rows
        chunk_rows = rows[start:end]
        path = root / f"train_{index + 1:02d}_{config.chunk_rows}.jsonl"
        _write_jsonl(path, chunk_rows)
        chunks.append({"index": index + 1, **_split_metadata(path, chunk_rows)})

    folds = [
        {
            "fold": evaluation_index - config.min_training_chunks,
            "training_chunk_indexes": list(range(1, evaluation_index)),
            "training_rows": (evaluation_index - 1) * config.chunk_rows,
            "evaluation_chunk_index": evaluation_index,
            "evaluation_rows": config.chunk_rows,
        }
        for evaluation_index in range(config.min_training_chunks + 1, config.chunk_count + 1)
    ]
    output = {
        "protocol": "pc28-combo-train-chunks-chronological-v1",
        "source_train": {
            "rows": int(train_metadata["rows"]),
            "first_issue": train_metadata["first_issue"],
            "last_issue": train_metadata["last_issue"],
            "sha256": train_metadata["sha256"],
        },
        "config": asdict(config),
        "chunks": chunks,
        "folds": folds,
        "validation_accessed": False,
        "test_accessed": False,
    }
    chunk_manifest = root / "train_chunks.manifest.json"
    chunk_manifest.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {**output, "manifest_path": str(chunk_manifest)}


def validate_train_tuning_chunks(manifest_path: str | Path) -> dict[str, Any]:
    """Validate train-only chunk hashes, chronology and progressive folds."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "pc28-combo-train-chunks-chronological-v1":
        raise ComboTrainingProtocolError("unknown train chunk protocol")
    if manifest.get("validation_accessed") or manifest.get("test_accessed"):
        raise ComboTrainingProtocolError("train chunk manifest must not access Validation or Test")
    config = TrainChunkConfig(**manifest["config"])
    chunks = manifest.get("chunks", [])
    if len(chunks) != config.chunk_count:
        raise ComboTrainingProtocolError("unexpected number of Train chunks")
    previous_issue: int | None = None
    for index, metadata in enumerate(chunks, start=1):
        if int(metadata.get("index", 0)) != index:
            raise ComboTrainingProtocolError("Train chunk indexes are not consecutive")
        raw = Path(metadata["path"])
        chunk_path = raw if raw.is_absolute() else path.parent / raw.name
        if _sha256(chunk_path) != metadata["sha256"]:
            raise ComboTrainingProtocolError("Train chunk hash does not match manifest")
        rows = clean_combo_history(chunk_path, min_rows=config.chunk_rows)
        if len(rows) != config.chunk_rows:
            raise ComboTrainingProtocolError("Train chunk row count does not match manifest")
        _assert_contiguous(rows)
        if previous_issue is not None and int(rows[0]["issue"]) != previous_issue + 1:
            raise ComboTrainingProtocolError("Train chunks are not chronologically adjacent")
        previous_issue = int(rows[-1]["issue"])
    return manifest


def validate_combo_split_manifest(manifest_path: str | Path, *, include_test: bool = True) -> dict[str, Any]:
    """Verify hashes, sizes, chronology and non-overlap before candidate work."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = ComboSplitConfig(**manifest["config"])
    loaded: dict[str, list[dict[str, Any]]] = {}
    required = {"train", "validation", "test"} if include_test else {"train", "validation"}
    if not required.issubset(manifest.get("splits", {})):
        raise ComboTrainingProtocolError("manifest is missing a required split")
    for name in required:
        metadata = manifest["splits"][name]
        split_path = Path(metadata["path"])
        if not split_path.is_absolute():
            split_path = path.parent / split_path.name
        if _sha256(split_path) != metadata["sha256"]:
            raise ComboTrainingProtocolError(f"{name} split hash does not match manifest")
        rows = clean_combo_history(split_path, min_rows=int(metadata["rows"]))
        if len(rows) != int(metadata["rows"]):
            raise ComboTrainingProtocolError(f"{name} split row count does not match manifest")
        _assert_contiguous(rows)
        loaded[name] = rows
    expected_sizes = [config.train_rows, config.validation_rows]
    actual_sizes = [len(loaded[name]) for name in ("train", "validation")]
    if include_test:
        expected_sizes.append(config.test_rows)
        actual_sizes.append(len(loaded["test"]))
    if actual_sizes != expected_sizes:
        raise ComboTrainingProtocolError("manifest split sizes do not match 8:1:1 configuration")
    if not int(loaded["train"][-1]["issue"]) < int(loaded["validation"][0]["issue"]):
        raise ComboTrainingProtocolError("splits overlap or are out of chronological order")
    if include_test and not int(loaded["validation"][-1]["issue"]) < int(loaded["test"][0]["issue"]):
        raise ComboTrainingProtocolError("splits overlap or are out of chronological order")
    return manifest


def _split_metadata(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path),
        "rows": len(rows),
        "first_issue": rows[0]["issue"],
        "last_issue": rows[-1]["issue"],
        "sha256": _sha256(path),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {key: row[key] for key in ("issue", "time", "digits", "sum")}
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _assert_contiguous(rows: list[dict[str, Any]]) -> None:
    if any(int(current["issue"]) != int(previous["issue"]) + 1 for previous, current in zip(rows, rows[1:])):
        raise ComboTrainingProtocolError("history contains a non-contiguous issue range")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
