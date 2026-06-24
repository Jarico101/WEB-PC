from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from .file_policy import resolve_model_path


def resolve_ppo_model_path(path: str | Path) -> Path:
    model_path = resolve_model_path(path)
    if _looks_like_dqn_model(model_path):
        raise ValueError(f"PPO entrypoint refuses DQN model: {model_path}. Use ai_agent/train_dqn.py for DQN.")
    return model_path


def _looks_like_dqn_model(model_path: Path) -> bool:
    if "dqn" in model_path.stem.lower():
        return True
    metadata_path = model_path.with_name(f"{model_path.stem}.metadata.json")
    if metadata_path.exists() and _metadata_looks_like_dqn(metadata_path):
        return True
    return _zip_looks_like_dqn(model_path)


def _metadata_looks_like_dqn(metadata_path: Path) -> bool:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        str(payload.get("action_space", "")).startswith("Discrete(")
        or "actions" in payload
        or "dqn" in str(payload.get("model_path", "")).lower()
    )


def _zip_looks_like_dqn(model_path: Path) -> bool:
    if not model_path.exists():
        return False
    try:
        with zipfile.ZipFile(model_path) as archive:
            data = _read_zip_json(archive, "data")
    except (OSError, zipfile.BadZipFile, KeyError):
        return False
    return "stable_baselines3.dqn" in json.dumps(data)


def _read_zip_json(archive: zipfile.ZipFile, name: str) -> Any:
    with archive.open(name) as file:
        return json.loads(file.read().decode("utf-8"))
