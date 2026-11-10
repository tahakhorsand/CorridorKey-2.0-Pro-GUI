"""Background worker for live SAM 3.1 interactive segmentation preview."""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from backend.frame_io import imread_unicode
from sam3_tracker import PromptFrame

logger = logging.getLogger(__name__)


class SAM3LivePreviewWorker(QThread):
    """Computes a live SAM 3.1 segmentation mask for the current frame on a background thread.

    Strict VRAM lifecycle: SAM 3.1 inference is performed under service._gpu_lock,
    and weights are immediately offloaded back to CPU in finally:.
    """

    preview_ready = Signal(int, object, object)  # (stem_idx, frame_rgb, mask)

    def __init__(
        self,
        input_path: str,
        prompt: PromptFrame,
        stem_idx: int,
        service: Any,
        is_linear: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.input_path = input_path
        self.prompt = prompt
        self.stem_idx = stem_idx
        self.service = service
        self.is_linear = is_linear

    def run(self) -> None:
        try:
            # 1. Read input frame
            is_exr = self.input_path.lower().endswith(".exr")
            if is_exr:
                frame_bgr = imread_unicode(self.input_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            else:
                frame_bgr = imread_unicode(self.input_path, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                logger.warning("SAM3LivePreviewWorker: failed to read %s", self.input_path)
                return

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_rgb.dtype != np.uint8:
                if self.is_linear:
                    frame_rgb = np.power(np.clip(frame_rgb, 0.0, 1.0), 1.0 / 2.2)
                frame_rgb = (np.clip(frame_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

            # 2. Run SAM 3.1 single-frame interactive preview under GPU lock
            with self.service._gpu_lock:
                tracker = self.service._get_sam3_tracker()
                mask = tracker.preview_prompt(frame_rgb, self.prompt)

            if mask is None:
                return

            self.preview_ready.emit(self.stem_idx, frame_rgb, mask)
        except Exception as e:
            logger.exception("Error in SAM3LivePreviewWorker: %s", e)
