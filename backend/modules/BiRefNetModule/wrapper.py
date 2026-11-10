"""BiRefNet Processor wrapper for CorridorKey 2.0.

Resolves BiRefNet models from C:\\Users\\imtaha\\.cache\\huggingface (and local
model cache directories) with all 9 ZhengPeng7 BiRefNet variants:
  - Matting (ZhengPeng7/BiRefNet-matting)
  - General (ZhengPeng7/BiRefNet)
  - HR Matting (ZhengPeng7/BiRefNet_HR-Matting)
  - HR General (ZhengPeng7/BiRefNet_HR)
  - Dynamic Matting (ZhengPeng7/BiRefNet_dynamic-matting)
  - Dynamic General (ZhengPeng7/BiRefNet_dynamic)
  - Lite Matting (ZhengPeng7/BiRefNet_lite-matting)
  - Lite 2K (ZhengPeng7/BiRefNet_lite-2K)
  - Lite General (ZhengPeng7/BiRefNet_lite)
"""
from __future__ import annotations

import gc
import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

from backend.frame_io import imwrite_unicode

logger = logging.getLogger(__name__)

DEFAULT_HF_CACHE_ROOT = Path(r"C:\Users\imtaha\.cache\huggingface")
DEFAULT_HF_HUB_DIR = DEFAULT_HF_CACHE_ROOT / "hub"

BIREFNET_MODELS: dict[str, tuple[str, int]] = {
    "Matting": ("ZhengPeng7/BiRefNet-matting", 1024),
    "General": ("ZhengPeng7/BiRefNet", 1024),
    "HR Matting": ("ZhengPeng7/BiRefNet_HR-Matting", 2048),
    "HR General": ("ZhengPeng7/BiRefNet_HR", 2048),
    "Dynamic Matting": ("ZhengPeng7/BiRefNet_dynamic-matting", 1024),
    "Dynamic": ("ZhengPeng7/BiRefNet_dynamic", 1024),
    "Lite Matting": ("ZhengPeng7/BiRefNet_lite-matting", 1024),
    "General Lite 2K": ("ZhengPeng7/BiRefNet_lite-2K", 2048),
    "General Lite": ("ZhengPeng7/BiRefNet_lite", 1024),
    # Aliases for backward compatibility
    "Lite 2K": ("ZhengPeng7/BiRefNet_lite-2K", 2048),
    "Lite": ("ZhengPeng7/BiRefNet_lite", 1024),
}

DEFAULT_MODEL = "Matting"


def _sync_local_birefnet_caches(hub_dir: Path) -> None:
    """Ensure any pre-populated BiRefNet snapshots on E:\\ComfyUI are mirrored into hub_dir."""
    try:
        comfy_biref = Path(r"E:\ComfyUI\ComfyUI\models\birefnet\models--ZhengPeng7--BiRefNet")
        target = hub_dir / "models--ZhengPeng7--BiRefNet"
        if comfy_biref.is_dir():
            src_snaps = comfy_biref / "snapshots"
            dst_snaps = target / "snapshots"
            if src_snaps.is_dir() and not dst_snaps.is_dir():
                shutil.copytree(comfy_biref, target, dirs_exist_ok=True)
    except Exception as e:
        logger.debug("BiRefNet cache sync skipped: %s", e)


def resolve_birefnet_snapshot(repo_id: str, cache_dir: Path = DEFAULT_HF_HUB_DIR) -> Optional[Path]:
    """Return the local snapshot folder if model.safetensors exists locally."""
    _sync_local_birefnet_caches(cache_dir)
    folder_name = "models--" + repo_id.replace("/", "--")
    for root in (cache_dir, Path(r"E:\ComfyUI\ComfyUI\models\birefnet")):
        snaps_root = root / folder_name / "snapshots"
        if snaps_root.is_dir():
            for snap in sorted(snaps_root.iterdir()):
                if snap.is_dir() and (snap / "model.safetensors").is_file() and (snap / "config.json").is_file():
                    return snap
    return None


class BiRefNetProcessor:
    """Wrapper for BiRefNet models using C:\\Users\\imtaha\\.cache\\huggingface."""

    def __init__(
        self,
        usage: str = DEFAULT_MODEL,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        resolved_usage = model_name or usage
        if resolved_usage not in BIREFNET_MODELS:
            logger.warning("Unknown BiRefNet model '%s', falling back to '%s'", resolved_usage, DEFAULT_MODEL)
            resolved_usage = DEFAULT_MODEL
        self._usage = resolved_usage
        self._model_name = resolved_usage
        self._repo_id, self._input_size = BIREFNET_MODELS[resolved_usage]
        self._cache_dir = Path(cache_dir) if cache_dir else DEFAULT_HF_HUB_DIR

        import torch

        if device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device

        self._model = None

    def _ensure_loaded(self, on_status: Optional[Callable[[str], None]] = None) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageSegmentation

        msg = f"Loading BiRefNet ({self._usage}: {self._repo_id}) on {self._device}..."
        logger.info(msg)
        if on_status:
            on_status(f"Loading BiRefNet ({self._usage})...")

        # Check if exact variant snapshot is already complete in local HF cache
        snap_path = resolve_birefnet_snapshot(self._repo_id, self._cache_dir)
        model = None

        if snap_path is not None:
            try:
                model = AutoModelForImageSegmentation.from_pretrained(
                    str(snap_path),
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception as e:
                logger.warning("Local snapshot load failed for %s: %s", snap_path, e)

        if model is None:
            #Temporarily allow HF hub resolution into C:\Users\imtaha\.cache\huggingface\hub if online,
            #or fall back to the cached ZhengPeng7/BiRefNet snapshot if offline.
            prev_hf_offline = os.environ.pop("HF_HUB_OFFLINE", None)
            prev_tf_offline = os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                model = AutoModelForImageSegmentation.from_pretrained(
                    self._repo_id,
                    trust_remote_code=True,
                    cache_dir=str(self._cache_dir),
                )
            except Exception as e:
                fallback_snap = resolve_birefnet_snapshot("ZhengPeng7/BiRefNet", self._cache_dir)
                if fallback_snap is not None:
                    logger.info(
                        "Offline fallback for '%s' -> using cached snapshot at %s",
                        self._repo_id,
                        fallback_snap,
                    )
                    model = AutoModelForImageSegmentation.from_pretrained(
                        str(fallback_snap),
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                else:
                    raise RuntimeError(f"Failed to load BiRefNet model '{self._repo_id}': {e}") from e
            finally:
                if prev_hf_offline is not None:
                    os.environ["HF_HUB_OFFLINE"] = prev_hf_offline
                if prev_tf_offline is not None:
                    os.environ["TRANSFORMERS_OFFLINE"] = prev_tf_offline

        model.to(self._device)
        model.eval()
        if self._device == "cuda":
            model.half()
        self._model = model
        logger.info("BiRefNet (%s) ready on %s", self._usage, self._device)

    def predict_frame(self, rgb_u8: np.ndarray) -> np.ndarray:
        """Predict alpha matte [H, W] float32 in [0, 1] from RGB uint8 [H, W, 3]."""
        self._ensure_loaded()
        import torch
        from torchvision import transforms

        h, w = rgb_u8.shape[:2]
        resized = cv2.resize(rgb_u8, (self._input_size, self._input_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor = normalize(tensor).unsqueeze(0).to(self._device)
        if self._device == "cuda":
            tensor = tensor.half()

        with torch.inference_mode():
            preds = self._model(tensor)[-1].sigmoid().cpu().float()
        del tensor
        alpha = preds[0, 0].numpy()
        del preds
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(alpha, 0.0, 1.0)

    def process_frames(
        self,
        input_frames: Iterable[np.ndarray],
        output_dir: str | Path,
        frame_names: list[str],
        progress_callback: Optional[Callable[..., None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        clip_name: str = "",
        num_frames: Optional[int] = None,
    ) -> int:
        """Stream frames through BiRefNet and write grayscale uint8 PNGs to output_dir."""
        try:
            self._ensure_loaded(on_status=on_status)
            if on_status:
                on_status(f"Running BiRefNet ({self._usage})...")
            os.makedirs(output_dir, exist_ok=True)
            total = num_frames if num_frames is not None else len(frame_names)
            written = 0
            for idx, frame in enumerate(input_frames):
                if cancel_check and cancel_check():
                    break
                if frame.dtype != np.uint8:
                    rgb_u8 = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    rgb_u8 = frame
                alpha = self.predict_frame(rgb_u8)
                alpha_u8 = (alpha * 255.0).astype(np.uint8)
                stem = frame_names[idx] if idx < len(frame_names) else f"frame_{idx:06d}"
                out_path = os.path.join(str(output_dir), f"{stem}.png")
                imwrite_unicode(out_path, alpha_u8)
                written += 1
                if progress_callback:
                    try:
                        progress_callback(clip_name, written, total)
                    except TypeError:
                        progress_callback(written, total)
            return written
        finally:
            self.offload()

    def offload(self) -> None:
        """Unload model from GPU memory immediately."""
        import torch

        if self._model is not None:
            try:
                self._model.cpu()
            except Exception:
                pass
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self) -> None:
        self.offload()
