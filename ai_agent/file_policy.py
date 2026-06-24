from __future__ import annotations

from pathlib import Path

from .config import DATA_DIR, LOG_DIR, MODEL_DIR, PROJECT_ROOT


class FileAccessError(PermissionError):
    pass


def resolve_history_path(path: str | Path) -> Path:
    return _resolve_allowed(path, DATA_DIR, "history")


def resolve_model_path(path: str | Path) -> Path:
    resolved = _resolve_allowed(path, MODEL_DIR, "model")
    if resolved.suffix != ".zip":
        raise FileAccessError(f"model path must be a .zip file under {MODEL_DIR}: {resolved}")
    return resolved


def resolve_log_path(path: str | Path) -> Path:
    return _resolve_allowed(path, LOG_DIR, "log")


def _resolve_allowed(path: str | Path, root: Path, label: str) -> Path:
    root_resolved = root.resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve(strict=False)
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise FileAccessError(f"{label} file access denied outside {root_resolved}: {resolved}")
    return resolved
