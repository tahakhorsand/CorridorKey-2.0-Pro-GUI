"""Core CorridorKey inference pipeline — dispatcher, sequential, and finalization."""
from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict, deque
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
            frame = None
            mid_idx = 0
            total_n = 0
            if clip.input_asset.asset_type == "video":
                total_n = max(1, int(clip.input_asset.frame_count or 1))
                mid_idx = total_n // 2
                frame = read_video_frame_at(clip.input_asset.path, mid_idx)
            else:
                files = clip.input_asset.get_frame_files()
                if not files:
                    return "green"
                total_n = len(files)
                mid_idx = total_n // 2
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
                        clip.name, result, mid_idx, total_n)
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

        # Resolve "auto" screen color from the clip's middle frame BEFORE v2 or v1 dispatch
        if getattr(params, "screen_color", "green") == "auto":
            from .core import InferenceParams
            resolved = self._resolve_screen_color(clip)
            params = InferenceParams(**{**params.to_dict(), "screen_color": resolved})

        package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        v2_safetensors = os.path.join(package_root, "weights", "corridorkey_v2.safetensors")
        v2_pt = os.path.join(package_root, "weights", "corridorkey_v2.pt")
        if os.path.isfile(v2_safetensors) or os.path.isfile(v2_pt):
            return self._run_inference_v2(
                clip, params, job=job,
                on_progress=on_progress, on_warning=on_warning,
                on_status=on_status, skip_stems=skip_stems,
                output_config=output_config, frame_range=frame_range,
            )

        if clip.alpha_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing alpha asset for v1 inference")

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
    def _load_clean_plate_rgb(params) -> Optional[np.ndarray]:
        if not bool(getattr(params, "clean_plate_enabled", False)):
            return None
        cpath = str(getattr(params, "clean_plate_path", "") or "").strip()
        if not cpath or not os.path.isfile(cpath):
            return None
        from ..frame_io import imread_unicode
        img = imread_unicode(cpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] >= 4:
            bgr = img[:, :, :3]
        else:
            bgr = img
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.dtype == np.uint8:
            return rgb.astype(np.float32) * (1.0 / 255.0)
        if rgb.dtype == np.uint16:
            return rgb.astype(np.float32) * (1.0 / 65535.0)
        return np.clip(rgb.astype(np.float32), 0.0, 1.0)

    @staticmethod
    def _postprocess_alpha_v2(
        alpha_f32: np.ndarray,
        params,
        hint_mask_u8: Optional[np.ndarray] = None,
        *,
        polygon_shapes=None,
        frame_index: int = 0,
        clean_plate_rgb: Optional[np.ndarray] = None,
        frame_rgb_f32: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply Alpha Levels, clean plate diff, despeckle, garbage matte, edge choke/grow, feather, and polygons."""
        alpha = np.clip(alpha_f32.astype(np.float32), 0.0, 1.0)

        if (
            bool(getattr(params, "clean_plate_enabled", False))
            and clean_plate_rgb is not None
            and frame_rgb_f32 is not None
        ):
            from backend.keying_post import apply_clean_plate_difference
            alpha = apply_clean_plate_difference(
                alpha,
                frame_rgb_f32,
                clean_plate_rgb,
                strength=float(getattr(params, "clean_plate_strength", 0.5) or 0.5),
                screen_color=str(getattr(params, "screen_color", "green") or "green"),
            )

        clip_black = float(getattr(params, "alpha_clip_black", 0.0) or 0.0)
        clip_white_raw = getattr(params, "alpha_clip_white", 1.0)
        clip_white = float(1.0 if clip_white_raw is None else clip_white_raw)
        gamma = float(getattr(params, "alpha_gamma", 1.0) or 1.0)
        clip_black = max(0.0, min(0.98, clip_black))
        clip_white = max(clip_black + 0.01, min(1.0, clip_white))
        gamma = max(0.1, min(4.0, gamma))
        if clip_black > 0.0 or clip_white < 1.0 or abs(gamma - 1.0) > 1e-4:
            alpha = np.clip((alpha - clip_black) / max(clip_white - clip_black, 1e-4), 0.0, 1.0)
            if abs(gamma - 1.0) > 1e-4:
                alpha = np.power(alpha, 1.0 / gamma)

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

        erode_px = int(getattr(params, "edge_erode_px", 0) or 0)
        if erode_px != 0:
            ek = abs(erode_px) * 2 + 1
            ekernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
            if erode_px < 0:
                alpha = cv2.erode(alpha, ekernel)
            else:
                alpha = cv2.dilate(alpha, ekernel)

        blur_px = int(getattr(params, "edge_blur_px", 0) or 0)
        if blur_px > 0:
            bk = max(3, (blur_px * 2 + 1) | 1)
            alpha = cv2.GaussianBlur(alpha, (bk, bk), 0)

        if polygon_shapes:
            from backend.matte_shapes import apply_polygon_mattes
            alpha = apply_polygon_mattes(alpha, polygon_shapes, int(frame_index))

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
        from ..frame_io import imread_unicode, imwrite_unicode
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
            elif not is_seq and frame_range is not None:
                lo = max(0, frame_range[0])
                files = all_files
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

            export_codec = str(getattr(params, "export_codec", "h264") or "h264").lower()
            if export_codec == "prores_4444":
                writer_cfg = WriterConfig(
                    ffmpeg_bin=ffmpeg_bin,
                    codec="prores_ks",
                    crf=0,
                    preset="4",
                    pix_fmt="yuva444p10le",
                    threads=2,
                )
            elif export_codec == "prores_422hq":
                writer_cfg = WriterConfig(
                    ffmpeg_bin=ffmpeg_bin,
                    codec="prores_ks",
                    crf=0,
                    preset="3",
                    pix_fmt="yuv422p10le",
                    threads=2,
                )
            else:
                writer_cfg = WriterConfig(
                    ffmpeg_bin=ffmpeg_bin,
                    codec="libx264",
                    crf=int(getattr(params, "crf", 12)),
                    preset=str(getattr(params, "preset", "medium")),
                    pix_fmt=str(getattr(params, "pix_fmt", "yuv444p10le")),
                    threads=2,
                )

            despill = (
                float(getattr(params, "despill_strength", 0.5))
                if bool(getattr(params, "despill_enabled", True))
                else 0.0
            )
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

            screen_color_resolved = str(getattr(params, "screen_color", "green") or "green").lower()
            if screen_color_resolved == "auto":
                screen_color_resolved = self._resolve_screen_color(clip)

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
                screen_color=screen_color_resolved,
                frame_dir_linear=frame_dir_linear,
                cutout_linear=cutout_linear,
                outputs={"alpha", "fg", "checker", "cutout"},
                output_format=output_format,
                exr_dtype=exr_dtype,
            )
            if not is_seq and frame_range is not None:
                infer_cfg.start_frame = lo
                infer_cfg.num_frames = max(1, frame_range[1] - lo + 1)

            from backend.matte_shapes import PolygonMatteModel
            polygon_shapes = PolygonMatteModel.load_from_clip(clip.root_path).shapes
            has_levels_adj = (
                float(getattr(params, "alpha_clip_black", 0.0) or 0.0) > 0.0
                or float(getattr(params, "alpha_clip_white", 1.0) if getattr(params, "alpha_clip_white", None) is not None else 1.0) < 1.0
                or abs(float(getattr(params, "alpha_gamma", 1.0) or 1.0) - 1.0) > 1e-4
            )

            clean_plate_rgb = self._load_clean_plate_rgb(params)
            adv_despill_enabled = bool(getattr(params, "advanced_despill_enabled", False))
            clean_plate_enabled = clean_plate_rgb is not None

            results: list[FrameResult] = []
            skipped: list[int] = []
            need_post = (
                bool(getattr(params, "auto_despeckle", True))
                or int(getattr(params, "garbage_matte_px", 0)) > 0
                or int(getattr(params, "edge_erode_px", 0)) != 0
                or int(getattr(params, "edge_blur_px", 0)) > 0
                or has_levels_adj
                or bool(polygon_shapes)
                or adv_despill_enabled
                or clean_plate_enabled
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
                        imwrite_unicode(os.path.join(dirs['comp'], f"{stem}.png"), comp_u8)

                    if "alpha" in f_dict and cfg.matte_enabled:
                        alpha_raw = f_dict["alpha"]
                        alpha_plane = alpha_raw[:, :, 0] if alpha_raw.ndim == 3 else alpha_raw
                        alpha_u8 = (alpha_plane >> 8).astype(np.uint8) if alpha_plane.dtype == np.uint16 else (np.clip(alpha_plane, 0, 1) * 255).astype(np.uint8)
                        imwrite_unicode(os.path.join(dirs['matte'], f"{stem}.png"), alpha_u8)

                    if "fg" in f_dict and cfg.fg_enabled:
                        fg_u16 = f_dict["fg"]
                        fg_u8 = (fg_u16 >> 8).astype(np.uint8) if fg_u16.dtype == np.uint16 else (np.clip(fg_u16, 0, 1) * 255).astype(np.uint8)
                        imwrite_unicode(os.path.join(dirs['fg'], f"{stem}.png"), fg_u8)
                else:
                    # Convert rendered BGR/alpha to float32, apply despeckle, edge refine, & garbage matte
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
                    frame_rgb_for_cp = None
                    if clean_plate_rgb is not None and is_seq and frame_k < len(files):
                        from ..frame_io import read_image_frame as _read_img_frame
                        cp_in = _read_img_frame(os.path.join(clip.input_asset.path, files[frame_k]))
                        if cp_in is not None:
                            frame_rgb_for_cp = (
                                cp_in.astype(np.float32) / 255.0
                                if cp_in.dtype == np.uint8
                                else np.clip(cp_in.astype(np.float32), 0.0, 1.0)
                            )
                    alpha_f = self._postprocess_alpha_v2(
                        alpha_f,
                        params,
                        hint_mask_u8=hint_mask,
                        polygon_shapes=polygon_shapes,
                        frame_index=abs_idx,
                        clean_plate_rgb=clean_plate_rgb,
                        frame_rgb_f32=frame_rgb_for_cp,
                    )

                    fg_raw = f_dict.get("fg")
                    is_exr_linear = (output_format == "exr" and fg_raw is not None and fg_raw.dtype != np.uint16)
                    checker = cu.make_checkerboard(alpha_f.shape[0], alpha_f.shape[1])
                    a3 = alpha_f[:, :, np.newaxis]

                    if fg_raw is None:
                        fg_despilled = np.zeros((*alpha_f.shape[:2], 3), dtype=np.float32)
                        comp_rgb = fg_despilled * a3 + checker * (1.0 - a3)
                        fg_lin = fg_despilled
                        proc_rgba = np.concatenate([fg_lin * a3, a3], axis=-1)
                        res_dict = {
                            "fg": fg_despilled,
                            "fg_is_linear": False,
                            "alpha": alpha_f,
                            "comp": comp_rgb,
                            "processed": proc_rgba,
                        }
                    elif is_exr_linear:
                        # Unclamped scene-linear float32 from runtime EXR renderer
                        fg_lin_rgb = cv2.cvtColor(fg_raw.astype(np.float32), cv2.COLOR_BGR2RGB)
                        fg_lin_despilled = fg_lin_rgb.copy()
                        if despill > 0.0:
                            if screen_color_resolved == "blue":
                                lim = (fg_lin_despilled[..., 0] + fg_lin_despilled[..., 1]) * 0.5
                                excess = np.maximum(0.0, fg_lin_despilled[..., 2] - lim)
                                fg_lin_despilled[..., 2] -= excess * despill
                            else:
                                lim = (fg_lin_despilled[..., 0] + fg_lin_despilled[..., 2]) * 0.5
                                excess = np.maximum(0.0, fg_lin_despilled[..., 1] - lim)
                                fg_lin_despilled[..., 1] -= excess * despill
                        fg_srgb = np.clip(cu.linear_to_srgb(np.maximum(fg_lin_despilled, 0.0)), 0.0, 1.0)
                        comp_rgb = fg_srgb * a3 + checker * (1.0 - a3)
                        proc_rgba = np.concatenate([fg_lin_despilled * a3, a3], axis=-1)
                        res_dict = {
                            "fg": fg_lin_despilled if cfg.fg_format == "exr" else fg_srgb,
                            "fg_is_linear": (cfg.fg_format == "exr"),
                            "alpha": alpha_f,
                            "comp": comp_rgb,
                            "processed": proc_rgba,
                        }
                    else:
                        fg_bgr = (
                            fg_raw.astype(np.float32) / 65535.0
                            if fg_raw.dtype == np.uint16
                            else np.clip(fg_raw.astype(np.float32), 0.0, 1.0)
                        )
                        fg_rgb = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2RGB)
                        if adv_despill_enabled:
                            from backend.keying_post import advanced_despill, apply_edge_extend, apply_light_wrap
                            fg_despilled = advanced_despill(
                                fg_rgb,
                                strength=despill,
                                screen_color=screen_color_resolved,
                                mode=str(getattr(params, "despill_mode", "average") or "average"),
                                luma_restore=float(getattr(params, "despill_luma_restore", 0.0) or 0.0),
                            )
                            fg_despilled = apply_edge_extend(
                                fg_despilled, alpha_f, int(getattr(params, "edge_extend_px", 0) or 0)
                            )
                            fg_despilled = apply_light_wrap(
                                fg_despilled,
                                checker,
                                alpha_f,
                                strength=float(getattr(params, "light_wrap_strength", 0.0) or 0.0),
                                radius_px=int(getattr(params, "light_wrap_radius", 12) or 12),
                            )
                        else:
                            fg_despilled = cu.despill(
                                fg_rgb,
                                strength=despill,
                                screen_color=screen_color_resolved,
                            )
                        comp_rgb = fg_despilled * a3 + checker * (1.0 - a3)
                        fg_lin = cu.srgb_to_linear(np.clip(fg_despilled, 0.0, 1.0))
                        proc_rgba = np.concatenate([fg_lin * a3, a3], axis=-1)
                        res_dict = {
                            "fg": fg_despilled,
                            "fg_is_linear": False,
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

            if bool(getattr(params, "temporal_deflicker_enabled", False)) and cfg.matte_enabled:
                from backend.keying_post import apply_temporal_alpha_deflicker
                from backend.natural_sort import natsorted
                matte_dir = dirs["matte"]
                if os.path.isdir(matte_dir):
                    mfiles = natsorted(f for f in os.listdir(matte_dir) if f.lower().endswith(".png"))
                    if len(mfiles) >= 3:
                        raw_alphas = []
                        for mf in mfiles:
                            am = imread_unicode(os.path.join(matte_dir, mf), cv2.IMREAD_GRAYSCALE)
                            if am is not None:
                                raw_alphas.append(am.astype(np.float32) * (1.0 / 255.0))
                        if len(raw_alphas) == len(mfiles):
                            smoothed = apply_temporal_alpha_deflicker(
                                raw_alphas,
                                strength=float(getattr(params, "temporal_deflicker_strength", 0.35) or 0.35),
                                radius=int(getattr(params, "temporal_deflicker_radius", 1) or 1),
                            )
                            fg_dir = dirs.get("fg")
                            comp_dir = dirs.get("comp")
                            for mf, a_sm in zip(mfiles, smoothed):
                                a_u8 = (np.clip(a_sm, 0.0, 1.0) * 255.0).astype(np.uint8)
                                imwrite_unicode(os.path.join(matte_dir, mf), a_u8)
                                if cfg.comp_enabled and fg_dir and comp_dir:
                                    fg_p = os.path.join(fg_dir, mf)
                                    if os.path.isfile(fg_p):
                                        fg_im = imread_unicode(fg_p, cv2.IMREAD_COLOR)
                                        if fg_im is not None:
                                            fg_f = fg_im[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)
                                            chk = cu.make_checkerboard(a_sm.shape[0], a_sm.shape[1])
                                            a3_sm = a_sm[:, :, np.newaxis]
                                            cmp_rgb = fg_f * a3_sm + chk * (1.0 - a3_sm)
                                            imwrite_unicode(
                                                os.path.join(comp_dir, mf),
                                                (np.clip(cmp_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1],
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
                    if not hasattr(engine, "process_frame"):
                        raise CorridorKeyError(
                            "Legacy frame-by-frame engine is unavailable; CorridorKey 2.0 weights "
                            "(corridorkey_v2.safetensors) are required."
                        )
                    with self._gpu_lock:
                        res = engine.process_frame(
                            img, mask,
                            input_is_linear=is_linear,
                            fg_is_straight=True,
                            despill_strength=(
                                params.despill_strength
                                if getattr(params, "despill_enabled", True)
                                else 0.0
                            ),
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
        """Run or re-postprocess a single frame in memory for Live Render preview.

        Never writes any outputs to disk. Uses a 2-tier cache:
          1. Caches raw CorridorKey 2.0 GPU predictions (`alpha_raw`, `fg_raw`, `fg_linear`)
             keyed by clip/frame/hint/linear/model_scale settings in `self._live_raw_cache`
             so pure despill / despeckle / erode / blur slider tweaks update on CPU in ~15ms.
          2. Falls cleanly into `infer_single_frame()` on GPU/CPU for RAW/unrendered clips.
        """
        from corridorkey import color_utils as cu
        from . import read_image_frame

        if params.screen_color == "auto":
            from .core import InferenceParams
            resolved = self._resolve_screen_color(clip)
            params = InferenceParams(**{**params.to_dict(), "screen_color": resolved})

        if clip.input_asset is None:
            return None
        if job and job.is_cancelled:
            return None

        # Locate exact input stem and resolve optional hint mask (alpha_asset -> mask_asset -> standalone)
        input_stem = f"frame_{frame_index:06d}"
        input_path = None
        if clip.input_asset.asset_type == "sequence":
            input_files = clip.input_asset.get_frame_files()
            if frame_index < 0 or frame_index >= len(input_files):
                return None
            input_stem = os.path.splitext(input_files[frame_index])[0]
            input_path = os.path.join(clip.input_asset.path, input_files[frame_index])

        hint_mask = None
        hint_sig: tuple = ("none",)
        use_hint = bool(getattr(params, "use_hint", True))
        hint_source = str(getattr(params, "hint_source", "auto") or "auto")
        if use_hint:
            hint_asset = None
            if hint_source == "sam" and clip.mask_asset is not None:
                hint_asset = clip.mask_asset
            elif hint_source in ("birefnet", "gvm", "matanyone2", "chroma", "imported", "applevision") and clip.alpha_asset is not None:
                hint_asset = clip.alpha_asset
            else:
                hint_asset = clip.alpha_asset or clip.mask_asset

            if hint_asset is not None:
                if hint_asset.asset_type == "video":
                    hint_mask = read_video_mask_at(hint_asset.path, frame_index)
                    try:
                        st = os.stat(hint_asset.path)
                        hint_sig = ("video", hint_asset.path, int(st.st_mtime_ns), frame_index)
                    except OSError:
                        hint_sig = ("video", hint_asset.path, frame_index)
                else:
                    hfiles = hint_asset.get_frame_files()
                    if hfiles:
                        lookup = {os.path.splitext(fn)[0]: fn for fn in hfiles}
                        chosen = lookup.get(input_stem)
                        if chosen is None and len(hfiles) == 1:
                            chosen = hfiles[0]
                        elif chosen is None and frame_index < len(hfiles):
                            chosen = hfiles[frame_index]
                        if chosen is not None:
                            hpath = os.path.join(hint_asset.path, chosen)
                            hint_mask = self._read_alpha_frame(
                                clip,
                                frame_index,
                                hfiles,
                                None,
                                input_stem=input_stem,
                                alpha_stem_lookup=lookup,
                                alpha_dir=hint_asset.path,
                            )
                            try:
                                st = os.stat(hpath)
                                hint_sig = ("seq", hpath, int(st.st_mtime_ns))
                            except OSError:
                                hint_sig = ("seq", hpath)

        model_cache_key = (
            clip.root_path,
            int(frame_index),
            use_hint and (hint_mask is not None),
            hint_source,
            hint_sig,
            bool(getattr(params, "input_is_linear", False)),
            round(float(getattr(params, "model_scale", 1.0)), 3),
            round(float(getattr(params, "hint_quality", 0.95)), 3),
            round(float(getattr(params, "linear_brightness", 1.0)), 3),
            round(float(getattr(params, "linear_contrast", 1.0)), 3),
            round(float(getattr(params, "linear_contrast_pivot", 0.18)), 3),
            bool(getattr(params, "auto_roi_crop", False)),
            int(getattr(params, "auto_roi_pad_px", 96)),
            bool(getattr(params, "color_mgmt_enabled", False)),
            str(getattr(params, "input_log_profile", "none") or "none"),
        )

        cache = getattr(self, "_live_raw_cache", None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            self._live_raw_cache = cache

        cached = cache.get(model_cache_key)
        if cached is not None:
            cache.move_to_end(model_cache_key)
            alpha_raw, fg_raw, fg_linear = cached
        else:
            if clip.input_asset.asset_type == "video":
                frame_rgb = read_video_frame_at(clip.input_asset.path, frame_index)
            else:
                if input_path is None:
                    return None
                frame_rgb = read_image_frame(input_path)
            if frame_rgb is None:
                return None

            is_lin_param = bool(getattr(params, "input_is_linear", False))
            if (
                bool(getattr(params, "color_mgmt_enabled", False))
                and str(getattr(params, "input_log_profile", "none") or "none") != "none"
            ):
                from backend.color_management import log_to_linear
                f_norm = (
                    frame_rgb.astype(np.float32) * (1.0 / 255.0)
                    if frame_rgb.dtype == np.uint8
                    else frame_rgb.astype(np.float32, copy=False)
                )
                frame_rgb = log_to_linear(f_norm, params.input_log_profile)
                is_lin_param = True

            if hint_mask is not None and hint_mask.shape[:2] != frame_rgb.shape[:2]:
                hint_mask = cv2.resize(
                    hint_mask, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_LINEAR
                )

            if job and job.is_cancelled:
                return None

            with self._gpu_lock:
                if job and job.is_cancelled:
                    return None
                engines = self._get_engine_pool(on_status=on_status, screen_color=params.screen_color)
                model = engines[0]

                from corridorkey.runtime import infer_single_frame
                if on_status:
                    on_status(f"Live Render: frame {frame_index + 1}...")
                raw_out = infer_single_frame(
                    model,
                    frame_rgb,
                    is_linear=is_lin_param,
                    hint_mask=hint_mask,
                    hint_quality=float(getattr(params, "hint_quality", 0.95)),
                    model_scale=float(getattr(params, "model_scale", 1.0)),
                    linear_brightness=float(getattr(params, "linear_brightness", 1.0)),
                    linear_contrast=float(getattr(params, "linear_contrast", 1.0)),
                    linear_contrast_pivot=float(getattr(params, "linear_contrast_pivot", 0.18)),
                    low_vram=bool(getattr(params, "low_vram", False)),
                    auto_roi_crop=bool(getattr(params, "auto_roi_crop", False)),
                    auto_roi_pad_px=int(getattr(params, "auto_roi_pad_px", 96)),
                )

            alpha_raw = raw_out["alpha_raw"]
            fg_raw = raw_out["fg_raw"]
            fg_linear = raw_out["fg_linear"]

            cache[model_cache_key] = (alpha_raw, fg_raw, fg_linear)
            cache.move_to_end(model_cache_key)
            while len(cache) > 4:
                cache.popitem(last=False)

        if job and job.is_cancelled:
            return None

        # Apply lightweight CPU alpha cleanup & despill on copies (never mutating cache)
        hint_mask_u8 = None
        if hint_mask is not None:
            hint_mask_u8 = (
                hint_mask
                if hint_mask.dtype == np.uint8
                else np.clip(hint_mask * 255.0, 0, 255).astype(np.uint8)
            )
        from backend.matte_shapes import PolygonMatteModel
        polygon_shapes = PolygonMatteModel.load_from_clip(clip.root_path).shapes
        clean_plate_rgb = self._load_clean_plate_rgb(params)
        frame_rgb_cp = None
        if clean_plate_rgb is not None:
            cp_raw = (
                read_video_frame_at(clip.input_asset.path, frame_index)
                if clip.input_asset.asset_type == "video"
                else (read_image_frame(input_path) if input_path else None)
            )
            if cp_raw is not None:
                frame_rgb_cp = (
                    cp_raw.astype(np.float32) * (1.0 / 255.0)
                    if cp_raw.dtype == np.uint8
                    else np.clip(cp_raw.astype(np.float32), 0.0, 1.0)
                )
        alpha_2d = self._postprocess_alpha_v2(
            alpha_raw.copy(),
            params,
            hint_mask_u8=hint_mask_u8,
            polygon_shapes=polygon_shapes,
            frame_index=frame_index,
            clean_plate_rgb=clean_plate_rgb,
            frame_rgb_f32=frame_rgb_cp,
        )
        alpha_f = alpha_2d[:, :, np.newaxis]

        checker = cu.make_checkerboard(alpha_2d.shape[0], alpha_2d.shape[1])
        eff_despill = (
            float(getattr(params, "despill_strength", 0.5))
            if bool(getattr(params, "despill_enabled", True))
            else 0.0
        )
        if bool(getattr(params, "advanced_despill_enabled", False)):
            from backend.keying_post import advanced_despill, apply_edge_extend, apply_light_wrap
            if eff_despill > 0.0:
                fg_despill = advanced_despill(
                    fg_raw.copy(),
                    strength=eff_despill,
                    screen_color=params.screen_color,
                    mode=str(getattr(params, "despill_mode", "average") or "average"),
                    luma_restore=float(getattr(params, "despill_luma_restore", 0.0) or 0.0),
                )
            else:
                fg_despill = fg_raw.copy()
            fg_despill = apply_edge_extend(
                fg_despill, alpha_2d, int(getattr(params, "edge_extend_px", 0) or 0)
            )
            fg_despill = apply_light_wrap(
                fg_despill,
                checker,
                alpha_2d,
                strength=float(getattr(params, "light_wrap_strength", 0.0) or 0.0),
                radius_px=int(getattr(params, "light_wrap_radius", 12) or 12),
            )
        elif eff_despill > 0.0:
            fg_despill = cu.despill(
                fg_raw.copy(),
                strength=eff_despill,
                screen_color=params.screen_color,
            )
        else:
            fg_despill = fg_raw.copy()
        comp = fg_despill * alpha_f + checker * (1.0 - alpha_f)
        fg_lin = cu.srgb_to_linear(np.clip(fg_despill, 0.0, 1.0))
        processed_rgba = np.concatenate([fg_lin * alpha_f, alpha_f], axis=-1)
        return {
            "fg": fg_despill,
            "alpha": alpha_f,
            "comp": comp,
            "processed": processed_rgba,
            "clip_root": clip.root_path,
            "frame_index": int(frame_index),
        }
