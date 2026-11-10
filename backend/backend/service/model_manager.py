"""Model loading, residency management, and VRAM lifecycle.



Implements the strict zero-idle-VRAM and single-model-in-VRAM policy:

- Only one heavy model stays on GPU during an active job.

- Immediately after any job finishes (or switches), GPU tensors are moved to CPU

  or dereferenced, Python GC runs, and torch.cuda.empty_cache() frees all VRAM.

"""

from __future__ import annotations



import gc

import logging

import os

import sys

import time

from enum import Enum

from typing import Callable, Optional



from .helpers import BASE_DIR



logger = logging.getLogger(__name__)





class _ActiveModel(Enum):

    """Tracks which heavy model is currently loaded in VRAM."""

    NONE = "none"

    INFERENCE = "inference"

    SAM3 = "sam3"

    MATANYONE2 = "matanyone2"

    BIREFNET = "birefnet"





class ModelManagerMixin:

    """Mixin providing model loading, residency, and strict VRAM management."""



    @staticmethod

    def _vram_allocated_mb() -> float:

        """Return current VRAM allocated in MB, or 0 if unavailable."""

        try:

            import torch

            if torch.cuda.is_available():

                return torch.cuda.memory_allocated(0) / (1024 ** 2)

        except Exception:

            pass

        return 0.0



    @staticmethod

    def _vram_reserved_mb() -> float:

        """Return current VRAM reserved in MB, or 0 if unavailable."""

        try:

            import torch

            if torch.cuda.is_available():

                return torch.cuda.memory_reserved(0) / (1024 ** 2)

        except Exception:

            pass

        return 0.0



    @staticmethod

    def _safe_offload(obj: object) -> None:

        """Move a model's GPU tensors to CPU before dropping the reference."""

        if obj is None:

            return

        name = type(obj).__name__

        logger.debug(f"Offloading model: {name}")

        try:

            if hasattr(obj, "unload"):

                obj.unload()

            elif hasattr(obj, "foundations") and hasattr(obj, "stage1") and hasattr(obj, "stage2"):

                # CorridorKeyV3 runtime model

                obj.foundations.cpu()

                obj.stage1.cpu()

                obj.stage2.cpu()

                obj.cpu()

            elif hasattr(obj, "to"):

                obj.to("cpu")

            elif hasattr(obj, "cpu"):

                obj.cpu()

            else:

                logger.debug(f"Model {name} has no .to()/.cpu()/.unload()")

        except Exception as e:

            logger.warning(f"Model offload failed for {name}: {e}")



    def release_vram(self, keep_corridorkey_cpu_cache: bool = True) -> None:

        """Aggressively offload all GPU models and empty CUDA cache so VRAM drops to ~0 MB."""

        vram_before = self._vram_reserved_mb()

        for eng in getattr(self, "_engine_pool", []):

            self._safe_offload(eng)

        if hasattr(self, "_engine_pool"):

            self._engine_pool.clear()



        ck2 = getattr(self, "_corridorkey_v2_model", None)

        if ck2 is not None:

            self._safe_offload(ck2)

            if not keep_corridorkey_cpu_cache:

                self._corridorkey_v2_model = None



        for attr in ("_sam3_tracker", "_matanyone2_processor", "_birefnet_processor"):

            proc = getattr(self, attr, None)

            if proc is not None:

                self._safe_offload(proc)

                setattr(self, attr, None)



        self._active_model = _ActiveModel.NONE

        gc.collect()

        gc.collect()

        try:

            import torch

            if torch.cuda.is_available():

                torch.cuda.synchronize()

                torch.cuda.empty_cache()

                try:

                    torch.cuda.ipc_collect()

                except Exception:

                    pass

                torch.cuda.reset_peak_memory_stats()

        except Exception:

            pass

        vram_after = self._vram_reserved_mb()

        logger.info(

            f"VRAM released: reserved {vram_before:.0f}MB -> {vram_after:.0f}MB "

            f"(allocated {self._vram_allocated_mb():.0f}MB)"

        )



    def _ensure_model(self, needed: _ActiveModel, on_status=None) -> None:

        """Model residency manager — unload current model if switching types."""

        if self._active_model == needed:

            return



        if self._active_model == _ActiveModel.INFERENCE and needed != _ActiveModel.INFERENCE:

            with self._gate_lock:

                self._switch_pending = True

            if not self._inference_idle.wait(timeout=300):

                logger.warning("Timed out waiting for inference sessions to drain")

            with self._gate_lock:

                self._switch_pending = False



        if self._active_model != _ActiveModel.NONE:

            if on_status:

                on_status(f"Switching {self._active_model.value} -> {needed.value}...")

            self.release_vram(keep_corridorkey_cpu_cache=True)



        self._active_model = needed



    def _get_corridorkey_v2_model(self, on_status=None):

        """Lazy-load the CorridorKey v2 foundation model (kept on CPU between jobs, moved to CUDA on demand)."""

        self._ensure_model(_ActiveModel.INFERENCE, on_status=on_status)

        if getattr(self, "_corridorkey_v2_model", None) is None:

            if on_status:

                on_status("Loading CorridorKey 2.0 foundation model...")

            from pathlib import Path

            from corridorkey.runtime import RuntimePaths, load_model

            package_root = Path(__file__).resolve().parent.parent.parent

            paths = RuntimePaths.from_package(package_root)

            paths.validate()

            self._corridorkey_v2_model = load_model(paths, device="cpu")

        return self._corridorkey_v2_model



    def _get_engine_pool(self, on_status=None, screen_color: str = "green") -> list:

        """Fallback engine pool getter."""

        self._ensure_model(_ActiveModel.INFERENCE, on_status=on_status)

        return [self._get_corridorkey_v2_model(on_status=on_status)]



    def _get_sam3_tracker(

        self,

        *,

        on_progress: Callable[[int, int], None] | None = None,

        on_status: Callable[[str], None] | None = None,

    ):

        """Lazy-load the SAM 3.1 tracker."""

        self._ensure_model(_ActiveModel.SAM3, on_status=on_status)



        if self._sam3_tracker is not None:

            return self._sam3_tracker



        from sam3_tracker import SAM3Tracker



        logger.info("Loading SAM 3.1 tracker...")

        t0 = time.monotonic()

        self._sam3_tracker = SAM3Tracker(

            model_id=self._sam3_model_id,

            device=self._device,

            vos_optimized=False,

            offload_video_to_cpu=self._device.startswith("cuda"),

            offload_state_to_cpu=False,

        )

        if self._sam3_tracker is not None:

            self._sam3_tracker.prepare(on_progress=on_progress, on_status=on_status)

        logger.info(f"SAM 3.1 tracker ready in {time.monotonic() - t0:.1f}s")

        return self._sam3_tracker



    def _get_matanyone2(self):

        """Placeholder after MatAnyone 2 removal."""

        raise NotImplementedError("MatAnyone2 has been removed. Use BiRefNet or native alpha.")



    def _get_birefnet(self, usage: str = "Matting", on_status=None):

        """Lazy-load the BiRefNet processor for a given model variant from local HF cache."""

        self._ensure_model(_ActiveModel.BIREFNET, on_status=on_status)



        if (self._birefnet_processor is not None

                and self._birefnet_processor._usage == usage):

            return self._birefnet_processor



        if self._birefnet_processor is not None:

            self._safe_offload(self._birefnet_processor)

            self._birefnet_processor = None



        from modules.BiRefNetModule.wrapper import BiRefNetProcessor

        logger.info(f"Loading BiRefNet ({usage})...")

        t0 = time.monotonic()

        self._birefnet_processor = BiRefNetProcessor(

            device=self._device, usage=usage,

        )

        self._birefnet_processor._ensure_loaded(on_status=on_status)

        logger.info(f"BiRefNet ({usage}) loaded in {time.monotonic() - t0:.1f}s")

        return self._birefnet_processor



    def unload_engines(self) -> None:

        """Free GPU memory by unloading all engines including CPU cache."""

        self.release_vram(keep_corridorkey_cpu_cache=False)

        logger.info("All engines unloaded, VRAM freed")



    def is_engine_loaded(self) -> bool:

        """True if the inference engine is currently active."""

        return (

            self._active_model == _ActiveModel.INFERENCE

            and getattr(self, "_corridorkey_v2_model", None) is not None

        )
