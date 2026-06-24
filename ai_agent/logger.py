from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import LOG_DIR
from .file_policy import resolve_log_path


def append_jsonl(event: dict[str, Any], *, log_path: str | Path | None = None) -> Path:
    path = resolve_log_path(log_path) if log_path else LOG_DIR / "decisions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return path
