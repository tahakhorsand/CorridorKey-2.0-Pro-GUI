"""CorridorKeyService — clean backend API for the UI and CLI.

Model Residency & VRAM Policy:
    Only ONE heavy model is loaded at a time, and immediately after any GPU job
    completes, GPU memory is released (`release_vram()`) so idle VRAM returns to ~0 MB.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, asdict
from typing import Optional

from ..clip_state import ClipEntry, ClipState, scan_clips_dir
from ..job_queue import GPUJobQueue
from .model_manager import ModelManagerMixin, _ActiveModel
from .frame_ops import FrameOpsMixin
from .inference import InferenceMixin
from .pipelines import PipelinesMixin

logger = logging.getLogger(__name__)

# CorridorKey 2.0 Defaults
_D = {
    "use_hint": True,
    "hint_source": "auto",
    "despill_strength": 0.5,
    "auto_despeckle": False,
    "despeckle_size": 20,
    "despeckle_dilation": 0,
    "despeckle_blur": 0,
    "refiner_scale": 1.0,
    "source_passthrough": False,
    "edge_erode_px": 0,
    "edge_blur_px": 0,
    "screen_color": "green",
    "garbage_matte_px": 0,
}


@dataclass
class InferenceParams:
    """Parameters for CorridorKey 2.0 inference."""
    # Native CorridorKey 2.0 parameters
    use_hint: bool = True
    hint_source: str = "auto"  # "auto", "birefnet", "sam"
    hint_mode: str = "auto"    # "auto", "one_frame", "full_sequence"
    input_is_linear: bool = False
    frame_dir_linear: bool = False
    cutout_linear: bool = False
    despill_strength: float = 0.5
    screen_color: str = "green"
    auto_vram: bool = True
    low_vram: bool = False
    hann_chunk: int = 80
    hann_stride: int = 40
    stage2_batch: int = 1
    carry_hint: bool = True
    hint_quality: float = 0.95
    carry_hint_quality: float = 0.95
    model_scale: float = 1.0
    linear_brightness: float = 1.0
    linear_contrast: float = 1.0
    linear_contrast_pivot: float = 0.18
    output_format: str = "video"
    exr_dtype: str = "half"
    crf: int = 12
    preset: str = "medium"
    pix_fmt: str = "yuv444p10le"
    # Post-processing / Cleanup fields
    auto_despeckle: bool = False
    despeckle_size: int = 20
    despeckle_dilation: int = 0
    despeckle_blur: int = 0
    refiner_scale: float = 1.0
    source_passthrough: bool = False
    edge_erode_px: int = 0
    edge_blur_px: int = 0
    garbage_matte_px: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InferenceParams":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class OutputConfig:
    """Which output types to produce and their format."""
    fg_enabled: bool = True
    fg_format: str = "png"
    matte_enabled: bool = True
    matte_format: str = "png"
    comp_enabled: bool = True
    comp_format: str = "png"
    processed_enabled: bool = True
    processed_format: str = "png"
    cutout_enabled: bool = True
    cutout_format: str = "mov"
    output_format: str = "video"  # "video" or "exr"
    exr_dtype: str = "half"       # "half" or "float"
    exr_compression: str = "dwab"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OutputConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def enabled_outputs(self) -> list[str]:
        """Return list of enabled output names for manifest."""
        out = []
        if self.fg_enabled:
            out.append("fg")
        if self.matte_enabled:
            out.append("matte")
        if self.comp_enabled:
            out.append("comp")
        if self.processed_enabled:
            out.append("processed")
        if self.cutout_enabled:
            out.append("cutout")
        return out


@dataclass
class FrameResult:
    """Result summary for a single processed frame."""
    frame_index: int
    input_stem: str
    success: bool
    warning: Optional[str] = None


class CorridorKeyService(
    ModelManagerMixin,
    FrameOpsMixin,
    InferenceMixin,
    PipelinesMixin,
):
    """Main backend service — scan, validate, process, write."""

    VALID_MODEL_RESOLUTIONS = (1024, 2048)

    def __init__(self):
        self._engine_pool: list = []
        self._pool_size: int = 1
        self._engine_screen_color: str = "green"
        self._model_resolution: int = 2048
        self._inference_backend: str = "auto"
        self._corridorkey_v2_model = None
        self._sam3_tracker = None
        self._matanyone2_processor = None
        self._birefnet_processor = None
        self._active_model = _ActiveModel.NONE
        self._device: str = "cpu"
        self._job_queue: Optional[GPUJobQueue] = None
        self._sam3_model_id: str = "sam3.1-INT8-ConvRot"
        self._gpu_lock = threading.Lock()
        self._inference_active: int = 0
        self._switch_pending: bool = False
        self._gate_lock = threading.Lock()
        self._inference_idle = threading.Event()
        self._inference_idle.set()

    @property
    def job_queue(self) -> GPUJobQueue:
        """Lazy-init GPU job queue."""
        if self._job_queue is None:
            self._job_queue = GPUJobQueue()
        return self._job_queue

    @property
    def sam3_model_id(self) -> str:
        """Current SAM 3.1 checkpoint preference."""
        return self._sam3_model_id

    def set_sam3_model(self, model_id: str) -> None:
        """Update the SAM 3.1 checkpoint used for future tracking jobs."""
        if not model_id or model_id == self._sam3_model_id:
            return
        logger.info("SAM 3.1 model preference changed: %s -> %s", self._sam3_model_id, model_id)
        self._sam3_model_id = model_id
        if self._sam3_tracker is not None:
            self._safe_offload(self._sam3_tracker)
            self._sam3_tracker = None
            if self._active_model == _ActiveModel.SAM3:
                self._active_model = _ActiveModel.NONE
            self.release_vram(keep_corridorkey_cpu_cache=True)

    @property
    def model_resolution(self) -> int:
        return self._model_resolution

    def set_model_resolution(self, res: int) -> None:
        if res not in self.VALID_MODEL_RESOLUTIONS:
            return
        if res != self._model_resolution:
            self._model_resolution = res
            self.release_vram(keep_corridorkey_cpu_cache=True)

    def set_inference_backend(self, backend: str) -> None:
        backend = (backend or "auto").lower()
        if backend not in ("auto", "mlx", "torch"):
            return
        if backend == self._inference_backend:
            return
        self._inference_backend = backend
        self.release_vram(keep_corridorkey_cpu_cache=True)

    def set_pool_size(self, n: int) -> None:
        self._pool_size = max(1, n)

    def _begin_inference(self) -> None:
        with self._gate_lock:
            self._inference_active += 1
            self._inference_idle.clear()
        while self._switch_pending:
            time.sleep(0.05)

    def _end_inference(self) -> None:
        with self._gate_lock:
            self._inference_active -= 1
            if self._inference_active <= 0:
                self._inference_active = 0
                self._inference_idle.set()

    def detect_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        except ImportError:
            self._device = "cpu"
        logger.info(f"Compute device: {self._device}")
        return self._device

    def get_vram_info(self) -> dict[str, float]:
        try:
            import torch
            if not torch.cuda.is_available():
                return {}
            props = torch.cuda.get_device_properties(0)
            total_bytes = getattr(props, "total_memory", getattr(props, "total_mem", 0))
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            return {
                "total": total_bytes / (1024**3),
                "reserved": reserved / (1024**3),
                "allocated": allocated / (1024**3),
                "free": (total_bytes - reserved) / (1024**3),
                "name": torch.cuda.get_device_name(0),
            }
        except Exception as e:
            logger.debug(f"VRAM query failed: {e}")
            return {}

    def scan_clips(
        self, clips_dir: str, allow_standalone_videos: bool = True,
    ) -> list[ClipEntry]:
        return scan_clips_dir(clips_dir, allow_standalone_videos=allow_standalone_videos)

    def get_clips_by_state(
        self,
        clips: list[ClipEntry],
        state: ClipState,
    ) -> list[ClipEntry]:
        return [c for c in clips if c.state == state]
