"""Core CorridorKey inference pipeline — dispatcher, sequential, and finalization."""
from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Callable, Optional

import cv2
import numpy as np

from ..clip_state import ClipEntry, ClipState
from ..errors import (
    CorridorKeyError,
    FrameReadError,
    JobCancelledError,
    WriteFailureError,
)
from ..frame_io import open_video, read_video_frame_at, read_video_mask_at
from ..job_queue import GPUJob
from ..validators import ensure_output_dirs, validate_frame_counts
from .inference_parallel import ParallelInferenceMixin

logger = logging.getLogger(__name__)


class InferenceMixin(ParallelInferenceMixin):
    """Mixin providing core inference pipeline for CorridorKeyService."""

    def _resolve_screen_color(self, clip: ClipEntry) -> str:
        """Detect green vs blue from the clip's middle frame.

        Reads a single frame at the midpoint and checks the dominant
        border hue. Caches the result on the clip entry.
        """
        cached = getattr(clip, '_screen_color_cache', None)
        if cached:
            return cached

        from corridorkey.color_utils import detect_screen_color
        from . import read_image_frame

        try:
            if clip.input_asset is None:
                return "green"
            files = clip.input_asset.get_frame_files()
            if not files:
                return "green"
            mid_idx = len(files) // 2
            fpath = os.path.join(clip.input_asset.path, files[mid_idx])
            frame = read_image_frame(fpath)
            if frame is None:
                return "green"
            # detect_screen_color expects uint8; convert from float32 [0,1]
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
            result = detect_screen_color(frame)
            clip._screen_color_cache = result
            logger.info("Auto-detected screen color for '%s': %s (frame %d/%d)",
                        clip.name, result, mid_idx, len(files))
            return result
        except Exception as e:
            logger.warning("Screen color detection failed for '%s': %s", clip.name, e)
            return "green"

    def run_inference(
        self,
        clip: ClipEntry,
        params,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[..., None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        skip_stems: Optional[set[str]] = None,
        output_config=None,
        frame_range: Optional[tuple[int, int]] = None,
    ) -> list:
        """Run CorridorKey inference on a single clip.

        Uses gated model switch protocol. If pool_size > 1, dispatches
        to _run_inference_parallel with N engines processing N frames
        concurrently. Otherwise uses _run_inference_sequential.
        """
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset")

        package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        v2_weights = os.path.join(package_root, "weights", "corridorkey_v2.safetensors")
        if os.path.isfile(v2_weights):
            return self._run_inference_v2(
                clip, params, job=job,
                on_progress=on_progress, on_warning=on_warning,
                on_status=on_status, skip_stems=skip_stems,
                output_config=output_config, frame_range=frame_range,
            )

        if clip.alpha_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing alpha asset for v1 inference")

        # Resolve "auto" screen color from the clip's middle frame
        if params.screen_color == "auto":
            from .core import InferenceParams
            resolved = self._resolve_screen_color(clip)
            params = InferenceParams(**{**params.to_dict(), "screen_color": resolved})

        self._begin_inference()
        try:
            if on_status:
                on_status("Loading model...")
            logger.info("run_inference: waiting for _gpu_lock")
            with self._gpu_lock:
                logger.info("run_inference: acquired _gpu_lock")
                engines = self._get_engine_pool(on_status=on_status,
                                                screen_color=params.screen_color)

            if len(engines) == 1:
                return self._run_inference_sequential(
                    clip, params, engines[0], job=job,
                    on_progress=on_progress, on_warning=on_warning,
                    on_status=on_status, skip_stems=skip_stems,
                    output_config=output_config, frame_range=frame_range,
                )
            return self._run_inference_parallel(
                clip, params, engines, job=job,
                on_progress=on_progress, on_warning=on_warning,
                on_status=on_status, skip_stems=skip_stems,
                output_config=output_config, frame_range=frame_range,
            )
        finally:
            self._end_inference()

    @staticmethod
    def _postprocess_alpha_v2(
        alpha_f32: np.ndarray,
        params,
        hint_mask_u8: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply EZ-CorridorKey despeckle and garbage matte parameters to [H, W] float32 alpha."""
        alpha = np.clip(alpha_f32.astype(np.float32), 0.0, 1.0)
        auto_despeckle = bool(getattr(params, "auto_despeckle", True))
        despeckle_size = int(getattr(params, "despeckle_size", 400))
        if auto_despeckle and despeckle_size > 0:
            binary = (alpha > 0.05).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if num_labels > 1:
                keep = np.ones(alpha.shape[:2], dtype=np.float32)
                for lbl in range(1, num_labels):
                    if stats[lbl, cv2.CC_STAT_AREA] < despeckle_size:
                        keep[labels == lbl] = 0.0
                alpha = alpha * keep

        garbage_px = int(getattr(params, "garbage_matte_px", 0))
        if garbage_px > 0 and hint_mask_u8 is not None:
            hm = hint_mask_u8
            if hm.ndim == 3:
                hm = hm[:, :, 0]
            if hm.shape[:2] != alpha.shape[:2]:
                hm = cv2.resize(hm, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_NEAREST)
            ksize = max(3, garbage_px * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            dilated = cv2.dilate((hm > 10).astype(np.uint8), kernel)
            feather = max(3, (garbage_px // 2) | 1)
            soft = cv2.GaussianBlur(dilated.astype(np.float32), (feather, feather), 0)
            alpha = alpha * np.clip(soft, 0.0, 1.0)

        return np.clip(alpha, 0.0, 1.0)

    def _run_corridorkey_v2_alpha_hint(
        self,
        clip: ClipEntry,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[..., None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        mask_hint_dir: Optional[str] = None,
    ) -> int:
        """Generate per-frame AlphaHint/*.png using CorridorKey 2.0."""
        import shutil
        import tempfile
        from pathlib import Path
        from ..clip_state import ClipAsset
        from ..frame_io import imwrite_unicode
        from corridorkey.runtime import InferenceConfig, WriterConfig, run_inference as ck2_run_inference

        if on_status:
            on_status("Loading CorridorKey 2.0 model for alpha generation...")
        with self._gpu_lock:
            model = self._get_corridorkey_v2_model(on_status=on_status)

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        selected_files = self._selected_sequence_files(clip)
        if not selected_files and clip.input_asset.asset_type == "sequence":
            selected_files = clip.input_asset.get_frame_files()
        stems = [os.path.splitext(f)[0] for f in selected_files]

        staging_root = tempfile.mkdtemp(prefix="ck2_alpha_stage_")
        try:
            if clip.input_asset.asset_type == "sequence":
                stage_in = os.path.join(staging_root, "input")
                os.makedirs(stage_in, exist_ok=True)
                for fname in selected_files:
                    src = os.path.join(clip.input_asset.path, fname)
                    dst = os.path.join(stage_in, fname)
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                input_path = stage_in
            else:
                input_path = clip.input_asset.path

            hint_video_path = None
            initial_hint_path = None
            initial_hint_frame_index = 0
            if mask_hint_dir and os.path.isdir(mask_hint_dir):
                mfiles = sorted(
                    f for f in os.listdir(mask_hint_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".exr"))
                )
                if len(mfiles) == 1:
                    initial_hint_path = os.path.join(mask_hint_dir, mfiles[0])
                    m_stem = os.path.splitext(mfiles[0])[0]
                    initial_hint_frame_index = stems.index(m_stem) if m_stem in stems else 0
                elif len(mfiles) > 1:
                    stage_mask = os.path.join(staging_root, "mask")
                    os.makedirs(stage_mask, exist_ok=True)
                    m_lookup = {os.path.splitext(f)[0]: f for f in mfiles}
                    for i, stem in enumerate(stems):
                        mf = m_lookup.get(stem) or (mfiles[i] if i < len(mfiles) else mfiles[-1])
                        src = os.path.join(mask_hint_dir, mf)
                        dst = os.path.join(stage_mask, f"{stem}{os.path.splitext(mf)[1]}")
                        try:
                            os.link(src, dst)
                        except OSError:
                            shutil.copy2(src, dst)
                    hint_video_path = stage_mask

            package_root = Path(__file__).resolve().parent.parent.parent
            local_ffmpeg = package_root / "ffmpeg" / "ffmpeg.exe"
            ffmpeg_bin = str(local_ffmpeg) if local_ffmpeg.is_file() else "ffmpeg"
            writer_cfg = WriterConfig(ffmpeg_bin=ffmpeg_bin, codec="libx264", crf=18, preset="ultrafast")
            infer_cfg = InferenceConfig(
                auto_vram=True,
                carry_hint=True,
                outputs={"alpha"},
                output_format="video",
            )

            written = 0

            def on_frame_rendered(frame_k: int, f_dict: dict[str, np.ndarray]):
                nonlocal written
                if job and job.is_cancelled:
                    raise JobCancelledError(clip.name, frame_k)
                if "alpha" in f_dict:
                    alpha_raw = f_dict["alpha"]
                    alpha_plane = alpha_raw[:, :, 0] if alpha_raw.ndim == 3 else alpha_raw
                    alpha_u8 = (
                        (alpha_plane >> 8).astype(np.uint8)
                        if alpha_plane.dtype == np.uint16
                        else (np.clip(alpha_plane, 0, 1) * 255).astype(np.uint8)
                    )
                    stem = stems[frame_k] if frame_k < len(stems) else f"frame_{frame_k:06d}"
                    imwrite_unicode(os.path.join(alpha_dir, f"{stem}.png"), alpha_u8)
                    written += 1

            def on_prog(done_count: int, total_count: int):
                if job and job.is_cancelled:
                    raise JobCancelledError(clip.name, done_count)
                if on_progress:
                    try:
                        on_progress(clip.name, done_count, total_count)
                    except TypeError:
                        pass

            with self._gpu_lock:
                ck2_run_inference(
                    model,
                    input_path,
                    os.path.join(staging_root, "out"),
                    infer_cfg,
                    writer_cfg,
                    initial_hint_path=initial_hint_path,
                    hint_video_path=hint_video_path,
                    initial_hint_frame_index=initial_hint_frame_index,
                    on_frame_rendered=on_frame_rendered,
                    on_progress=on_prog,
                    on_status=on_status,
                    is_cancelled=lambda: bool(job and job.is_cancelled),
                )

            clip.alpha_asset = ClipAsset(alpha_dir, "sequence")
            try:
                clip.transition_to(ClipState.READY)
            except Exception:
                clip.state = ClipState.READY
            return written
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
            self.release_vram(keep_corridorkey_cpu_cache=True)

    def _resolve_hint_asset(self, clip: ClipEntry, hint_source: str = "auto"):
        """Resolve which hint/mask asset to use for inference based on hint_source.

        Args:
            clip: Target ClipEntry.
            hint_source: "auto", "birefnet", or "sam".
        """
        from ..clip_state import ClipAsset

        sam_dirs = [
            os.path.join(clip.root_path, "SamMaskHint"),
            os.path.join(clip.root_path, "VideoMamaMaskHint"),
        ]
        birefnet_dirs = [
            os.path.join(clip.root_path, "BiRefNetHint"),
            os.path.join(clip.root_path, "AlphaHint"),
        ]

        sam_asset = None
        for d in sam_dirs:
            if os.path.isdir(d) and os.listdir(d):
                sam_asset = ClipAsset(d, "sequence")
                break
        if sam_asset is None and clip.mask_asset is not None:
            sam_asset = clip.mask_asset

        birefnet_asset = None
        for d in birefnet_dirs:
            if os.path.isdir(d) and os.listdir(d):
                birefnet_asset = ClipAsset(d, "sequence")
                break
        if birefnet_asset is None and clip.alpha_asset is not None:
            birefnet_asset = clip.alpha_asset

        source_mode = str(hint_source or "auto").lower()

        if source_mode == "sam":
            if sam_asset is not None:
                return sam_asset
            logger.warning(
                "Clip '%s': SAM mask hint requested but no SAM masks found on disk. Falling back to available hint.",
                clip.name,
            )
            return birefnet_asset or clip.alpha_asset or clip.mask_asset

        elif source_mode == "birefnet":
            if birefnet_asset is not None:
                return birefnet_asset
            logger.warning(
                "Clip '%s': BiRefNet alpha hint requested but no BiRefNet alpha found on disk. Falling back to available hint.",
                clip.name,
            )
            return sam_asset or clip.alpha_asset or clip.mask_asset

        else:  # "auto"
            # In auto mode: if SAM mask track exists (interactive user-drawn), prioritize SAM; otherwise BiRefNet / AlphaHint
            return sam_asset or birefnet_asset or clip.alpha_asset or clip.mask_asset

    def _run_inference_v2(
        self,
        clip: ClipEntry,
        params,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[..., None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        skip_stems: Optional[set[str]] = None,
        output_config=None,
        frame_range: Optional[tuple[int, int]] = None,
    ) -> list:
        import shutil
        import tempfile
        from pathlib import Path
        from .core import FrameResult, OutputConfig
        from ..frame_io import imread_unicode
        from corridorkey import color_utils as cu
        from corridorkey.runtime import InferenceConfig, WriterConfig, run_inference as ck2_run_inference

        t_start = time.monotonic()
        dirs = ensure_output_dirs(clip.output_dir)
        cfg = output_config or OutputConfig()
        self._write_manifest(dirs['root'], cfg, params)

        staging_dir: Optional[str] = None
        self._begin_inference()
        try:
            if on_status:
                on_status("Loading CorridorKey 2.0 model...")
            with self._gpu_lock:
                model = self._get_corridorkey_v2_model(on_status=on_status)

            input_path = clip.input_asset.path
            is_seq = clip.input_asset.asset_type == 'sequence'
            all_files = clip.input_asset.get_frame_files() if is_seq else []
            total_clip_frames = len(all_files) if is_seq else clip.input_asset.frame_count

            if is_seq and frame_range is not None:
                lo = max(0, frame_range[0])
                hi = min(len(all_files) - 1, frame_range[1])
                files = all_files[lo:hi + 1]
            else:
                lo = 0
                files = all_files

            input_stems = [os.path.splitext(f)[0] for f in files]

            use_hint = bool(getattr(params, "use_hint", True))
            hint_source = str(getattr(params, "hint_source", "auto") or "auto").lower()
            hint_mode = str(getattr(params, "hint_mode", "auto") or "auto").lower()
            initial_hint_path = None
            hint_video_path = None
            initial_hint_frame_index = 0
            hint_lookup: dict[str, str] = {}
            if use_hint:
                hint_asset = self._resolve_hint_asset(clip, hint_source)
                if hint_asset is not None:
                    if hint_asset.asset_type == 'video':
                        hint_video_path = hint_asset.path
                    else:
                        afiles = hint_asset.get_frame_files()
                        hint_lookup = {os.path.splitext(f)[0]: os.path.join(hint_asset.path, f) for f in afiles}
                        if len(afiles) == 1 or hint_mode == "one_frame":
                            chosen_file = afiles[0] if afiles else None
                            ofh_path = os.path.join(clip.root_path, "one_frame_hint.json")
                            target_stem = None
                            if os.path.isfile(ofh_path):
                                try:
                                    import json as _json
                                    with open(ofh_path, "r", encoding="utf-8") as f:
                                        ofh_data = _json.load(f)
                                        target_stem = ofh_data.get("stem")
                                except Exception:
                                    target_stem = None

                            if target_stem:
                                for f in afiles:
                                    if os.path.splitext(f)[0] == target_stem:
                                        chosen_file = f
                                        break

                            if chosen_file:
                                initial_hint_path = os.path.join(hint_asset.path, chosen_file)
                                hint_stem = os.path.splitext(chosen_file)[0]
                                if hint_stem in input_stems:
                                    initial_hint_frame_index = input_stems.index(hint_stem)
                                else:
                                    initial_hint_frame_index = 0
                                logger.info(
                                    "[%s] One Frame Hint applied on frame index %d (stem '%s'): %s",
                                    clip.name, initial_hint_frame_index, hint_stem, initial_hint_path
                                )
                        elif len(afiles) > 1:
                            hint_video_path = hint_asset.path
                    logger.info(
                        "[%s] Hint system '%s' (mode: %s) selected -> Using %s (%d frame(s)).",
                        clip.name, hint_source, hint_mode, hint_asset.path, len(hint_lookup) or 1
                    )
                else:
                    logger.warning("[%s] 'Use Hint' is enabled (system: %s) but no hint asset was found on disk.", clip.name, hint_source)
            else:
                logger.info(f"[{clip.name}] 'Use Hint' is disabled — running CorridorKey 2.0 standalone without hints.")

            # If frame_range is active on a sequence (or hint count mismatches), stage the exact slice
            if is_seq and (
                frame_range is not None
                or (hint_video_path and os.path.isdir(hint_video_path) and len(hint_lookup) != len(files))
            ):
                staging_dir = tempfile.mkdtemp(prefix="ck2_infer_stage_")
                stage_in = os.path.join(staging_dir, "input")
                os.makedirs(stage_in, exist_ok=True)
                for fname in files:
                    src = os.path.join(clip.input_asset.path, fname)
                    dst = os.path.join(stage_in, fname)
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)
                input_path = stage_in

                if hint_video_path and os.path.isdir(hint_video_path) and hint_lookup:
                    stage_hint = os.path.join(staging_dir, "hint")
                    os.makedirs(stage_hint, exist_ok=True)
                    fallback_hint = next(iter(hint_lookup.values()))
                    for stem in input_stems:
                        hsrc = hint_lookup.get(stem, fallback_hint)
                        hdst = os.path.join(stage_hint, f"{stem}{os.path.splitext(hsrc)[1]}")
                        try:
                            os.link(hsrc, hdst)
                        except OSError:
                            shutil.copy2(hsrc, hdst)
                    hint_video_path = stage_hint

            package_root = Path(__file__).resolve().parent.parent.parent
            local_ffmpeg = package_root / "ffmpeg" / "ffmpeg.exe"
            ffmpeg_bin = str(local_ffmpeg) if local_ffmpeg.is_file() else "ffmpeg"

            writer_cfg = WriterConfig(
                ffmpeg_bin=ffmpeg_bin,
                codec="libx264",
                crf=int(getattr(params, "crf", 12)),
                preset=str(getattr(params, "preset", "medium")),
                pix_fmt=str(getattr(params, "pix_fmt", "yuv444p10le")),
                threads=2,
            )

            despill = float(getattr(params, "despill_strength", 0.5))
            frame_dir_linear = bool(getattr(params, "input_is_linear", False))
            cutout_linear = bool(getattr(params, "cutout_linear", False))
            auto_vram = bool(getattr(params, "auto_vram", True))
            low_vram = bool(getattr(params, "low_vram", False))
            hann_chunk = int(getattr(params, "hann_chunk", 80))
            hann_stride = int(getattr(params, "hann_stride", 40))
            stage2_batch = int(getattr(params, "stage2_batch", 1))
            carry_hint = bool(getattr(params, "carry_hint", True))
            hint_quality = float(getattr(params, "hint_quality", 0.95))
            carry_hint_quality = float(getattr(params, "carry_hint_quality", 0.95))
            model_scale = float(getattr(params, "model_scale", 1.0))
            linear_brightness = float(getattr(params, "linear_brightness", 1.0))
            linear_contrast = float(getattr(params, "linear_contrast", 1.0))
            linear_contrast_pivot = float(getattr(params, "linear_contrast_pivot", 0.18))
            output_format = str(getattr(cfg, "output_format", getattr(params, "output_format", "video")))
            if output_format not in ("video", "exr"):
                output_format = "video"
            exr_dtype = str(getattr(cfg, "exr_dtype", getattr(params, "exr_dtype", "half")))

            infer_cfg = InferenceConfig(
                hann_chunk=hann_chunk,
                hann_stride=hann_stride,
                carry_hint=carry_hint,
                hint_quality=hint_quality,
                carry_hint_quality=carry_hint_quality,
                model_scale=model_scale,
                low_vram=low_vram,
                auto_vram=auto_vram,
                stage2_batch=stage2_batch,
                linear_brightness=linear_brightness,
                linear_contrast=linear_contrast,
                linear_contrast_pivot=linear_contrast_pivot,
                despill_strength=despill,
                frame_dir_linear=frame_dir_linear,
                cutout_linear=cutout_linear,
                outputs={"alpha", "fg", "checker", "cutout"},
                output_format=output_format,
                exr_dtype=exr_dtype,
            )
            if not is_seq and frame_range is not None:
                infer_cfg.num_frames = frame_range[1] - frame_range[0] + 1

            results: list[FrameResult] = []
            skipped: list[int] = []
            need_post = (
                bool(getattr(params, "auto_despeckle", True))
                or int(getattr(params, "garbage_matte_px", 0)) > 0
                or cfg.fg_format == "exr"
                or cfg.matte_format == "exr"
                or cfg.comp_format == "exr"
                or cfg.processed_enabled
            )

            def on_frame_rendered(frame_k: int, f_dict: dict[str, np.ndarray]):
                if job and job.is_cancelled:
                    raise JobCancelledError(clip.name, frame_k)

                abs_idx = lo + frame_k
                stem = input_stems[frame_k] if (input_stems and frame_k < len(input_stems)) else f"frame_{abs_idx:06d}"

                if skip_stems and stem in skip_stems:
                    results.append(FrameResult(abs_idx, stem, True, "resumed (skipped)"))
                    return

                if not need_post:
                    if "checker" in f_dict and cfg.comp_enabled:
                        comp_u16 = f_dict["checker"]
                        comp_u8 = (comp_u16 >> 8).astype(np.uint8) if comp_u16.dtype == np.uint16 else (np.clip(comp_u16, 0, 1) * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(dirs['comp'], f"{stem}.png"), comp_u8)

                    if "alpha" in f_dict and cfg.matte_enabled:
                        alpha_raw = f_dict["alpha"]
                        alpha_plane = alpha_raw[:, :, 0] if alpha_raw.ndim == 3 else alpha_raw
                        alpha_u8 = (alpha_plane >> 8).astype(np.uint8) if alpha_plane.dtype == np.uint16 else (np.clip(alpha_plane, 0, 1) * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(dirs['matte'], f"{stem}.png"), alpha_u8)

                    if "fg" in f_dict and cfg.fg_enabled:
                        fg_u16 = f_dict["fg"]
                        fg_u8 = (fg_u16 >> 8).astype(np.uint8) if fg_u16.dtype == np.uint16 else (np.clip(fg_u16, 0, 1) * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(dirs['fg'], f"{stem}.png"), fg_u8)
                else:
                    # Convert rendered BGR/alpha to float32 [0, 1], apply despeckle & garbage matte, and use _write_outputs
                    alpha_raw = f_dict.get("alpha")
                    if alpha_raw is not None:
                        alpha_plane = alpha_raw[:, :, 0] if alpha_raw.ndim == 3 else alpha_raw
                        alpha_f = (
                            alpha_plane.astype(np.float32) / 65535.0
                            if alpha_plane.dtype == np.uint16
                            else np.clip(alpha_plane.astype(np.float32), 0.0, 1.0)
                        )
                    else:
                        alpha_f = np.ones((4, 4), dtype=np.float32)

                    hint_mask = None
                    if use_hint and int(getattr(params, "garbage_matte_px", 0)) > 0 and stem in hint_lookup:
                        hint_mask = imread_unicode(hint_lookup[stem], cv2.IMREAD_GRAYSCALE)
                    alpha_f = self._postprocess_alpha_v2(alpha_f, params, hint_mask_u8=hint_mask)

                    fg_raw = f_dict.get("fg")
                    if fg_raw is not None:
                        fg_bgr = (
                            fg_raw.astype(np.float32) / 65535.0
                            if fg_raw.dtype == np.uint16
                            else np.clip(fg_raw.astype(np.float32), 0.0, 1.0)
                        )
                        fg_rgb = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        fg_rgb = np.zeros((*alpha_f.shape[:2], 3), dtype=np.float32)

                    fg_despilled = cu.despill(
                        fg_rgb,
                        strength=despill,
                        screen_color=getattr(params, "screen_color", "green"),
                    )
                    checker = cu.make_checkerboard(alpha_f.shape[0], alpha_f.shape[1])
                    a3 = alpha_f[:, :, np.newaxis]
                    comp_rgb = fg_despilled * a3 + checker * (1.0 - a3)
                    fg_lin = cu.srgb_to_linear(np.clip(fg_despilled, 0.0, 1.0))
                    proc_rgba = np.concatenate([fg_lin * a3, a3], axis=-1)

                    res_dict = {
                        "fg": fg_despilled,
                        "alpha": alpha_f,
                        "comp": comp_rgb,
                        "processed": proc_rgba,
                    }
                    self._write_outputs(res_dict, dirs, stem, clip.name, abs_idx, cfg)

                results.append(FrameResult(abs_idx, stem, True))

            def on_prog(done_count: int, total_count: int):
                if job and job.is_cancelled:
                    raise JobCancelledError(clip.name, done_count)
                if on_progress:
                    elapsed = time.monotonic() - t_start
                    fps = done_count / elapsed if elapsed > 0 else 0.0
                    rem = total_count - done_count
                    eta = rem / fps if fps > 0 else 0.0
                    on_progress(clip.name, done_count, total_count, fps=fps, elapsed=elapsed, eta_seconds=eta)

            def is_cancelled_check():
                return bool(job and job.is_cancelled)

            if on_status:
                on_status("Running CorridorKey 2.0...")

            with self._gpu_lock:
                ck2_run_inference(
                    model,
                    input_path,
                    dirs['root'],
                    infer_cfg,
                    writer_cfg,
                    initial_hint_path=initial_hint_path,
                    hint_video_path=hint_video_path,
                    initial_hint_frame_index=initial_hint_frame_index,
                    on_frame_rendered=on_frame_rendered,
                    on_progress=on_prog,
                    on_status=on_status,
                    is_cancelled=is_cancelled_check,
                )

            processed_count = len(results)
            return self._finalize_inference(
                clip, results, skipped, processed_count,
                t_start, frame_range, total_clip_frames or processed_count,
                on_warning=on_warning,
            )
        finally:
            if staging_dir:
                shutil.rmtree(staging_dir, ignore_errors=True)
            self._end_inference()
            self.release_vram(keep_corridorkey_cpu_cache=True)

    def _run_inference_sequential(
        self,
        clip: ClipEntry,
        params,
        engine,
        job: Optional[GPUJob] = None,
        on_progress: Optional[Callable[..., None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        skip_stems: Optional[set[str]] = None,
        output_config=None,
        frame_range: Optional[tuple[int, int]] = None,
    ) -> list:
        """Sequential frame loop — exact extraction of original run_inference."""
        from .core import FrameResult, OutputConfig
        t_start = time.monotonic()
        dirs = ensure_output_dirs(clip.output_dir)
        cfg = output_config or OutputConfig()

        self._write_manifest(dirs['root'], cfg, params)

        if clip.input_asset.asset_type == 'sequence' and clip.alpha_asset.asset_type == 'sequence':
            num_frames = clip.input_asset.frame_count
            if clip.input_asset.frame_count != clip.alpha_asset.frame_count:
                logger.warning(
                    "Clip '%s': sequence alpha count mismatch — input has %d, alpha has %d. "
                    "Using stem-matched alpha reads across the selected range.",
                    clip.name,
                    clip.input_asset.frame_count,
                    clip.alpha_asset.frame_count,
                )
        else:
            num_frames = validate_frame_counts(
                clip.name,
                clip.input_asset.frame_count,
                clip.alpha_asset.frame_count,
            )

        input_cap = None
        alpha_cap = None
        input_files: list[str] = []
        alpha_files: list[str] = []

        if clip.input_asset.asset_type == 'video':
            input_cap = open_video(clip.input_asset.path)
        else:
            input_files = clip.input_asset.get_frame_files()

        if clip.alpha_asset.asset_type == 'video':
            alpha_cap = open_video(clip.alpha_asset.path)
        else:
            alpha_files = clip.alpha_asset.get_frame_files()
        alpha_stem_lookup = (
            {os.path.splitext(fname)[0]: fname for fname in alpha_files}
            if alpha_files else None
        )

        results: list[FrameResult] = []
        skipped: list[int] = []
        skip_stems = skip_stems or set()
        frame_times: deque[float] = deque(maxlen=10)
        processed_count = 0

        if frame_range is not None:
            range_start = max(0, frame_range[0])
            range_end = min(num_frames - 1, frame_range[1])
            frame_indices = range(range_start, range_end + 1)
            range_count = range_end - range_start + 1
        else:
            frame_indices = range(num_frames)
            range_count = num_frames

        _warmup_done = False

        try:
            for progress_i, i in enumerate(frame_indices):
                if job and job.is_cancelled:
                    raise JobCancelledError(clip.name, i)

                if not _warmup_done and on_status:
                    on_status("Compiling (first frame may take a minute)...")

                if on_progress:
                    timing_kwargs: dict[str, float] = {}
                    elapsed = time.monotonic() - t_start
                    timing_kwargs["elapsed"] = elapsed
                    if frame_times:
                        avg_time = sum(frame_times) / len(frame_times)
                        fps = 1.0 / avg_time if avg_time > 0 else 0.0
                        remaining = range_count - progress_i
                        timing_kwargs["fps"] = fps
                        timing_kwargs["eta_seconds"] = remaining * avg_time
                    on_progress(clip.name, progress_i, range_count, **timing_kwargs)

                try:
                    img, input_stem, is_linear = self._read_input_frame(
                        clip, i, input_files, input_cap, params.input_is_linear,
                    )
                    if img is None:
                        skipped.append(i)
                        results.append(FrameResult(i, f"{i:05d}", False, "video read failed"))
                        continue

                    if input_stem in skip_stems:
                        results.append(FrameResult(i, input_stem, True, "resumed (skipped)"))
                        continue

                    mask = self._read_alpha_frame(
                        clip, i, alpha_files, alpha_cap,
                        input_stem=input_stem,
                        alpha_stem_lookup=alpha_stem_lookup,
                    )
                    if mask is None:
                        skipped.append(i)
                        results.append(FrameResult(i, input_stem, False, "alpha read failed"))
                        continue

                    if mask.shape[:2] != img.shape[:2]:
                        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

                    t_frame = time.monotonic()
                    with self._gpu_lock:
                        res = engine.process_frame(
                            img, mask,
                            input_is_linear=is_linear,
                            fg_is_straight=True,
                            despill_strength=params.despill_strength,
                            auto_despeckle=params.auto_despeckle,
                            despeckle_size=params.despeckle_size,
                            despeckle_dilation=params.despeckle_dilation,
                            despeckle_blur=params.despeckle_blur,
                            refiner_scale=params.refiner_scale,
                            source_passthrough=params.source_passthrough,
                            edge_erode_px=params.edge_erode_px,
                            edge_blur_px=params.edge_blur_px,
                            screen_color=params.screen_color,
                            garbage_matte_px=params.garbage_matte_px,
                        )
                    dt = time.monotonic() - t_frame
                    frame_times.append(dt)
                    processed_count += 1

                    if not _warmup_done:
                        _warmup_done = True
                        if on_status:
                            on_status("")
                    total_t = sum(frame_times)
                    avg_fps = len(frame_times) / total_t if total_t > 0 else 0.0
                    logger.debug(
                        f"Frame {i}: {dt * 1000:.0f}ms ({avg_fps:.1f} fps avg)"
                    )

                    self._write_outputs(res, dirs, input_stem, clip.name, i, cfg)
                    results.append(FrameResult(i, input_stem, True))

                except FrameReadError as e:
                    logger.warning(str(e))
                    skipped.append(i)
                    results.append(FrameResult(i, f"{i:05d}", False, str(e)))
                    if on_warning:
                        on_warning(str(e))

                except WriteFailureError as e:
                    logger.error(str(e))
                    results.append(FrameResult(i, f"{i:05d}", False, str(e)))
                    if on_warning:
                        on_warning(str(e))

            if on_progress:
                final_elapsed = time.monotonic() - t_start
                final_kwargs: dict[str, float] = {"elapsed": final_elapsed, "eta_seconds": 0.0}
                total_t = sum(frame_times)
                if total_t > 0:
                    final_kwargs["fps"] = len(frame_times) / total_t
                on_progress(clip.name, range_count, range_count, **final_kwargs)

        finally:
            if input_cap:
                input_cap.release()
            if alpha_cap:
                alpha_cap.release()

        return self._finalize_inference(
            clip, results, skipped, processed_count,
            t_start, frame_range, num_frames,
            on_warning=on_warning,
        )

    def _finalize_inference(
        self,
        clip: ClipEntry,
        results: list,
        skipped: list[int],
        processed_count: int,
        t_start: float,
        frame_range: Optional[tuple[int, int]],
        num_frames: int,
        on_warning: Optional[Callable[[str], None]] = None,
    ) -> list:
        """Shared post-processing for sequential and parallel inference."""
        range_count = len(results) if results else 0
        processed = sum(1 for r in results if r.success)
        if skipped:
            msg = (
                f"Clip '{clip.name}': {len(skipped)} frame(s) skipped: "
                f"{skipped[:20]}{'...' if len(skipped) > 20 else ''}"
            )
            logger.warning(msg)
            if on_warning:
                on_warning(msg)

        t_total = time.monotonic() - t_start
        avg_fps = processed_count / t_total if t_total > 0 and processed_count > 0 else 0.0
        range_label = f" (range {frame_range[0]}-{frame_range[1]})" if frame_range else ""
        logger.info(
            f"Clip '{clip.name}': inference complete{range_label}. {processed}/{range_count} frames "
            f"in {t_total:.1f}s ({t_total / max(processed, 1):.2f}s/frame, {avg_fps:.1f} fps avg)"
        )

        is_full_clip = (frame_range is None or
                        (frame_range[0] == 0 and frame_range[1] >= num_frames - 1))
        if processed == range_count and is_full_clip:
            try:
                clip.transition_to(ClipState.COMPLETE)
            except Exception as e:
                logger.warning(f"Clip '{clip.name}': state transition to COMPLETE failed: {e}")

        return results

    def reprocess_single_frame(
        self,
        clip: ClipEntry,
        params,
        frame_index: int,
        job: Optional[GPUJob] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Optional[dict]:
        """Reprocess a single frame with current params.

        Returns the result dict (fg, alpha, comp, processed) or None.
        Does NOT write to disk — returns in-memory results for preview.
        """
        from . import read_image_frame

        # Resolve "auto" screen color
        if params.screen_color == "auto":
            from .core import InferenceParams
            resolved = self._resolve_screen_color(clip)
            params = InferenceParams(**{**params.to_dict(), "screen_color": resolved})

        t_start = time.monotonic()
        if clip.input_asset is None:
            return None

        # Try fast disk-cached preview with updated despill
        input_stem = f"{frame_index:05d}"
        if clip.input_asset.asset_type == 'sequence':
            input_files = clip.input_asset.get_frame_files()
            if frame_index < len(input_files):
                input_stem = os.path.splitext(input_files[frame_index])[0]

        comp_path = os.path.join(clip.output_dir, "Comp", f"{input_stem}.png")
        if not os.path.isfile(comp_path):
            comp_path = os.path.join(clip.output_dir, "Comp", f"frame_{frame_index:06d}.png")
        fg_path = os.path.join(clip.output_dir, "FG", f"{input_stem}.png")
        if not os.path.isfile(fg_path):
            fg_path = os.path.join(clip.output_dir, "FG", f"frame_{frame_index:06d}.png")
        matte_path = os.path.join(clip.output_dir, "Matte", f"{input_stem}.png")
        if not os.path.isfile(matte_path):
            matte_path = os.path.join(clip.output_dir, "Matte", f"frame_{frame_index:06d}.png")

        for ext in ("png", "exr"):
            cand_fg = os.path.join(clip.output_dir, "FG", f"{input_stem}.{ext}")
            cand_matte = os.path.join(clip.output_dir, "Matte", f"{input_stem}.{ext}")
            if os.path.isfile(cand_fg):
                fg_path = cand_fg
            if os.path.isfile(cand_matte):
                matte_path = cand_matte

        if os.path.isfile(fg_path) and os.path.isfile(matte_path):
            from backend.frame_io import imread_unicode
            fg_img = imread_unicode(fg_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            alpha_img = imread_unicode(matte_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if fg_img is not None and alpha_img is not None:
                fg_f = fg_img.astype(np.float32) / (255.0 if fg_img.dtype == np.uint8 else 1.0)
                fg_f = cv2.cvtColor(fg_f, cv2.COLOR_BGR2RGB)
                if fg_path.lower().endswith(".exr"):
                    from corridorkey import color_utils as _cu
                    fg_f = np.clip(_cu.linear_to_srgb(np.maximum(fg_f, 0.0)), 0.0, 1.0)
                alpha_2d = alpha_img.astype(np.float32) / (255.0 if alpha_img.dtype == np.uint8 else 1.0)
                if alpha_2d.ndim == 3:
                    alpha_2d = alpha_2d[:, :, 0]
                hint_mask = None
                if int(getattr(params, "garbage_matte_px", 0)) > 0 and clip.alpha_asset is not None and clip.alpha_asset.asset_type == "sequence":
                    for af in clip.alpha_asset.get_frame_files():
                        if os.path.splitext(af)[0] == input_stem:
                            hint_mask = imread_unicode(os.path.join(clip.alpha_asset.path, af), cv2.IMREAD_GRAYSCALE)
                            break
                alpha_2d = self._postprocess_alpha_v2(alpha_2d, params, hint_mask_u8=hint_mask)
                alpha_f = alpha_2d[:, :, np.newaxis]
                from corridorkey import color_utils as cu
                fg_despill = cu.despill(fg_f, strength=params.despill_strength, screen_color=params.screen_color)
                checker = cu.make_checkerboard(fg_f.shape[0], fg_f.shape[1])
                comp = fg_despill * alpha_f + checker * (1.0 - alpha_f)
                fg_lin = cu.srgb_to_linear(np.clip(fg_despill, 0.0, 1.0))
                processed_rgba = np.concatenate([fg_lin * alpha_f, alpha_f], axis=-1)
                return {
                    'fg': fg_despill,
                    'alpha': alpha_f,
                    'comp': comp,
                    'processed': processed_rgba,
                }

        if clip.alpha_asset is None:
            return None

        if job and job.is_cancelled:
            return None

        with self._gpu_lock:
            engines = self._get_engine_pool(on_status=on_status,
                                            screen_color=params.screen_color)
            engine = engines[0]

        # Read the specific input frame
        is_linear = params.input_is_linear
        input_stem = f"{frame_index:05d}"
        if clip.input_asset.asset_type == 'video':
            img = read_video_frame_at(clip.input_asset.path, frame_index)
        else:
            input_files = clip.input_asset.get_frame_files()
            if frame_index >= len(input_files):
                return None
            fpath = os.path.join(clip.input_asset.path, input_files[frame_index])
            input_stem = os.path.splitext(input_files[frame_index])[0]
            img = read_image_frame(fpath)
        if img is None:
            return None

        # Read the specific alpha frame
        if clip.alpha_asset.asset_type == 'video':
            mask = read_video_mask_at(clip.alpha_asset.path, frame_index)
        else:
            alpha_files = clip.alpha_asset.get_frame_files()
            alpha_stem_lookup = {os.path.splitext(fname)[0]: fname for fname in alpha_files}
            mask = self._read_alpha_frame(
                clip,
                frame_index,
                alpha_files,
                None,
                input_stem=input_stem,
                alpha_stem_lookup=alpha_stem_lookup,
            )
        if mask is None:
            return None

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        if job and job.is_cancelled:
            return None

        with self._gpu_lock:
            res = engine.process_frame(
                img, mask,
                input_is_linear=is_linear,
                fg_is_straight=True,
                despill_strength=params.despill_strength,
                auto_despeckle=params.auto_despeckle,
                despeckle_size=params.despeckle_size,
                despeckle_dilation=params.despeckle_dilation,
                despeckle_blur=params.despeckle_blur,
                refiner_scale=params.refiner_scale,
                source_passthrough=params.source_passthrough,
                edge_erode_px=params.edge_erode_px,
                edge_blur_px=params.edge_blur_px,
                screen_color=params.screen_color,
                garbage_matte_px=params.garbage_matte_px,
            )
        logger.debug(f"Clip '{clip.name}' frame {frame_index}: reprocess {time.monotonic() - t_start:.3f}s")
        return res
