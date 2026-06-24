from __future__ import annotations

import torch


def select_torch_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(requested: str = "auto") -> str:
    if requested == "auto":
        return select_torch_device()
    if requested not in {"mps", "cuda", "cpu"}:
        raise ValueError("device must be one of: auto, mps, cuda, cpu")
    return requested
