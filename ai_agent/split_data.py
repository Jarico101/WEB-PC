from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import DATA_DIR
from .data_adapter import load_history
from .file_policy import resolve_history_path


@dataclass(frozen=True)
class SplitManifest:
    source: str
    rows: int
    train_rows: int
    valid_rows: int
    test_rows: int
    walk_forward_segments: int
    outputs: dict[str, str]


def split_history_file(
    source: str | Path,
    *,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    walk_forward_segments: int = 5,
) -> SplitManifest:
    rows = load_history(source)
    if len(rows) < 3:
        raise ValueError("need at least 3 rows to create train/valid/test splits")
    if not (0.0 < train_ratio < 1.0 and 0.0 <= valid_ratio < 1.0 and train_ratio + valid_ratio < 1.0):
        raise ValueError("split ratios must leave a non-empty test set")

    train_end = max(1, int(round(len(rows) * train_ratio)))
    valid_rows = max(1, int(round(len(rows) * valid_ratio)))
    valid_end = min(max(train_end + valid_rows, train_end + 1), len(rows) - 1)

    splits = {
        "train": rows[:train_end],
        "valid": rows[train_end:valid_end],
        "test": rows[valid_end:],
    }
    output_paths = {
        "train": DATA_DIR / "train.jsonl",
        "valid": DATA_DIR / "valid.jsonl",
        "test": DATA_DIR / "test.jsonl",
    }
    for name, split_rows in splits.items():
        _write_jsonl(output_paths[name], split_rows)

    walk_paths = _write_walk_forward(splits["test"], walk_forward_segments)
    manifest_path = DATA_DIR / "split_manifest.json"
    outputs = {name: str(path) for name, path in output_paths.items()}
    outputs.update({f"walk_forward_{index + 1}": str(path) for index, path in enumerate(walk_paths)})
    outputs["manifest"] = str(manifest_path)

    manifest = SplitManifest(
        source=str(resolve_history_path(source)),
        rows=len(rows),
        train_rows=len(splits["train"]),
        valid_rows=len(splits["valid"]),
        test_rows=len(splits["test"]),
        walk_forward_segments=len(walk_paths),
        outputs=outputs,
    )
    manifest_path.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _write_walk_forward(rows: list[dict[str, Any]], segment_count: int) -> list[Path]:
    if segment_count <= 0:
        return []
    segment_count = min(segment_count, len(rows))
    paths: list[Path] = []
    for index in range(segment_count):
        start = int(len(rows) * index / segment_count)
        end = int(len(rows) * (index + 1) / segment_count)
        segment_rows = rows[start:end]
        path = DATA_DIR / f"walk_forward_{index + 1}.jsonl"
        _write_jsonl(path, segment_rows)
        paths.append(path)
    return paths


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = resolve_history_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed train/valid/test splits without reordering rows.")
    parser.add_argument("history", help="Source history file under data/")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--walk-forward-segments", type=int, default=5)
    args = parser.parse_args()

    manifest = split_history_file(
        args.history,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        walk_forward_segments=args.walk_forward_segments,
    )
    print(json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
