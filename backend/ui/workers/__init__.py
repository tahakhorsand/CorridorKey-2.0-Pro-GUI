"""Background workers for GPU processing and monitoring (lazy-loaded)."""
from __future__ import annotations

from typing import Any

__all__ = ["SAM3LivePreviewWorker"]


def __getattr__(name: str) -> Any:
    if name == "SAM3LivePreviewWorker":
        from ui.workers.sam3_preview_worker import SAM3LivePreviewWorker
        return SAM3LivePreviewWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
