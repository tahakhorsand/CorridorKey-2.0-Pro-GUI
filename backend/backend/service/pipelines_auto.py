"""Auto-alpha pipelines — BiRefNet (local HF cache), Chroma Key, Apple Vision."""
from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Callable, Optional

from ..clip_state import ClipAsset, ClipEntry, ClipState
from ..errors import (
    CorridorKeyError,
    GPURequiredError,
    JobCancelledError,
)
from .model_manager import _ActiveModel

logger = logging.getLogger(__name__)


class AutoPipelinesMixin:
    """Mixin providing auto-alpha pipelines (BiRefNet, Chroma Key) for CorridorKeyService."""

    def run_birefnet(
        self,
        clip: ClipEntry,
        usage: str = "Matting",
        frame_index: Optional[int] = None,
        job=None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Optional[dict[str, object]]:
        """Run BiRefNet automatic alpha generation for a clip using local cached weights.

        If frame_index is provided, generates a single-frame alpha hint (One Frame Hint).
        Otherwise streams the full sequence.
        Transitions clip: RAW -> READY (creates AlphaHint directory) and frees VRAM upon completion.
        """
        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for BiRefNet")

        if self._device == "cpu":
            raise GPURequiredError("BiRefNet")

        def _status(msg: str) -> None:
            logger.info(f"BiRefNet [{clip.name}]: {msg}")
            if on_status:
                on_status(msg)

        def _check_cancel() -> bool:
            return bool(job and job.is_cancelled)

        t_start = time.monotonic()

        try:
            _status(f"Loading BiRefNet ({usage}) from local cache...")
            with self._gpu_lock:
                processor = self._get_birefnet(usage=usage, on_status=on_status)

            if _check_cancel():
                raise JobCancelledError(clip.name, 0)

            selected_input_names = self._selected_sequence_files(clip)
            if not selected_input_names:
                raise CorridorKeyError(f"Clip '{clip.name}' has no input frames for BiRefNet")

            # --- ONE FRAME HINT MODE ---
            if frame_index is not None:
                import numpy as np
                target_idx = max(0, min(int(frame_index), len(selected_input_names) - 1))
                target_file = selected_input_names[target_idx]
                stem = os.path.splitext(target_file)[0]
                _status(f"Running BiRefNet One Frame Hint on frame {target_idx + 1} ({stem})...")

                named_frames = self._load_named_sequence_frames(
                    clip.input_asset,
                    [target_file],
                    clip.name,
                    gamma_correct_exr=False,
                    job=job,
                    on_status=None,
                )
                if not named_frames:
                    raise CorridorKeyError(f"Could not load frame '{target_file}' for BiRefNet")

                frame_rgb = named_frames[0][1]
                if frame_rgb.dtype != np.uint8:
                    rgb_u8 = (np.clip(frame_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
                else:
                    rgb_u8 = frame_rgb

                alpha = processor.predict_frame(rgb_u8)
                self.save_one_frame_hint(clip, target_idx, alpha, stem=stem, source="birefnet")

                if on_progress:
                    on_progress(clip.name, 1, 1)

                logger.info(
                    f"BiRefNet One Frame Hint complete for '{clip.name}' [frame {target_idx + 1}] in "
                    f"{time.monotonic() - t_start:.2f}s"
                )
                return {
                    "kind": "one_frame_hint",
                    "source": "BiRefNet",
                    "clip_name": clip.name,
                    "frame_index": target_idx,
                    "stem": stem,
                    "alpha": alpha,
                }

            # --- FULL SEQUENCE MODE ---
            _status("Streaming frames...")
            frame_stems = [os.path.splitext(fname)[0] for fname in selected_input_names]
            num_frames = len(selected_input_names)

            frame_iter = (
                frame
                for _, frame in self._iter_named_sequence_frames(
                    clip.input_asset,
                    selected_input_names,
                    clip.name,
                    job=job,
                    on_status=None,
                )
            )

            if _check_cancel():
                raise JobCancelledError(clip.name, 0)

            alpha_dir = os.path.join(clip.root_path, "AlphaHint")
            frames_written = processor.process_frames(
                input_frames=frame_iter,
                output_dir=alpha_dir,
                frame_names=frame_stems,
                progress_callback=on_progress,
                on_status=on_status,
                cancel_check=_check_cancel,
                clip_name=clip.name,
                num_frames=num_frames,
            )

            # Preserve in dedicated BiRefNetHint directory so tracking SAM does not overwrite BiRefNet
            birefnet_dir = os.path.join(clip.root_path, "BiRefNetHint")
            os.makedirs(birefnet_dir, exist_ok=True)
            for fname in os.listdir(alpha_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    shutil.copy2(os.path.join(alpha_dir, fname), os.path.join(birefnet_dir, fname))

            # Full track is authoritative, remove one_frame_hint manifest if it existed
            ofh_path = os.path.join(clip.root_path, "one_frame_hint.json")
            if os.path.isfile(ofh_path):
                try:
                    os.remove(ofh_path)
                except OSError:
                    pass

            clip.alpha_asset = ClipAsset(alpha_dir, "sequence")
            clip.find_assets()
            try:
                clip.transition_to(ClipState.READY)
            except Exception as e:
                if on_warning:
                    on_warning(f"State transition after BiRefNet: {e}")

            logger.info(
                f"BiRefNet complete for '{clip.name}': "
                f"{frames_written} alpha frames in {time.monotonic() - t_start:.1f}s"
            )
            return None
        finally:
            self.release_vram(keep_corridorkey_cpu_cache=True)

    def run_chroma_key(
        self,
        clip: ClipEntry,
        chroma_params: dict,
        job=None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Generate alpha hints via Chroma Key (Rust ck2_native / CPU multi-threaded)."""
        import cv2 as _cv2
        import numpy as _np
        from concurrent.futures import ThreadPoolExecutor, Future
        from corridorkey.chroma_key import chroma_key_matte
        from ..frame_io import imread_unicode as _imread, imwrite_unicode as _imwrite

        if clip.input_asset is None:
            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for chroma key")

        t_start = time.monotonic()

        selected_input_names = self._selected_sequence_files(clip)
        if not selected_input_names:
            raise CorridorKeyError(f"Clip '{clip.name}' has no input frames for chroma key")

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        os.makedirs(alpha_dir, exist_ok=True)

        num_frames = len(selected_input_names)
        if on_progress:
            on_progress(clip.name, 0, num_frames)

        input_dir = clip.input_asset.path
        ck_params = {
            "screen_color": chroma_params.get("screen_color"),
            "screen_type": chroma_params.get("screen_type", "green"),
            "strength": chroma_params.get("strength", 1.0),
            "clip_black": chroma_params.get("clip_black", 0.0),
            "clip_white": chroma_params.get("clip_white", 1.0),
            "shrink_grow": chroma_params.get("shrink_grow", 0),
            "edge_blur": chroma_params.get("edge_blur", 0),
        }

        from ui.widgets.annotation_overlay import AnnotationModel
        _holdout_model = AnnotationModel(filename="holdout_strokes.json")
        _holdout_model.load(clip.root_path)
        _holdout_needs_raster = _holdout_model.has_annotations(0)
        _holdout_mask = None

        def _read_frame(fname: str) -> tuple[str, _np.ndarray | None]:
            fpath = os.path.join(input_dir, fname)
            is_exr = fname.lower().endswith(".exr")
            if is_exr:
                bgr = _imread(fpath, _cv2.IMREAD_ANYCOLOR | _cv2.IMREAD_ANYDEPTH)
            else:
                bgr = _imread(fpath, _cv2.IMREAD_COLOR)
            if bgr is None:
                logger.warning(f"Chroma key: could not read {fpath}, skipping")
                return fname, None
            rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
            if rgb.dtype != _np.uint8:
                rgb = (_np.clip(rgb, 0.0, 1.0) * 255.0).astype(_np.uint8)
            return fname, rgb

        def _write_matte(out_path: str, matte: _np.ndarray) -> None:
            _imwrite(out_path, matte)

        _PREFETCH = 4
        _WRITE_WORKERS = 4

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ck-read") as read_pool, \
             ThreadPoolExecutor(max_workers=_WRITE_WORKERS, thread_name_prefix="ck-write") as write_pool:

            read_futures: list[Future] = []
            for fname in selected_input_names[:_PREFETCH]:
                read_futures.append(read_pool.submit(_read_frame, fname))

            write_futures: list[Future] = []
            next_read_idx = _PREFETCH

            for i in range(num_frames):
                if job and job.is_cancelled:
                    raise JobCancelledError(clip.name, i)

                fname, frame_rgb = read_futures[i].result()

                if next_read_idx < num_frames:
                    read_futures.append(
                        read_pool.submit(_read_frame, selected_input_names[next_read_idx])
                    )
                    next_read_idx += 1

                if frame_rgb is None:
                    continue

                if _holdout_needs_raster and _holdout_mask is None:
                    h, w = frame_rgb.shape[:2]
                    _holdout_mask = AnnotationModel.rasterize_holdout_mask(
                        _holdout_model.get_strokes(0), w, h
                    )
                    _holdout_needs_raster = False

                matte = chroma_key_matte(frame_rgb, holdout_mask=_holdout_mask, **ck_params)

                stem = os.path.splitext(fname)[0]
                out_path = os.path.join(alpha_dir, f"{stem}.png")
                write_futures.append(write_pool.submit(_write_matte, out_path, matte))

                if len(write_futures) > _WRITE_WORKERS * 2:
                    write_futures = [f for f in write_futures if not f.done()]

                if on_progress:
                    elapsed = time.monotonic() - t_start
                    fps = (i + 1) / elapsed if elapsed > 0 else 0
                    on_progress(clip.name, i + 1, num_frames, fps=fps)

            for f in write_futures:
                f.result()

        clip.alpha_asset = ClipAsset(alpha_dir, "sequence")

        try:
            clip.transition_to(ClipState.READY)
        except Exception as e:
            if on_warning:
                on_warning(f"State transition after chroma key: {e}")

        logger.info(
            f"Chroma key complete for '{clip.name}': "
            f"{num_frames} alpha frames in {time.monotonic() - t_start:.1f}s"
        )

    def run_applevision(
        self,
        clip: ClipEntry,
        job=None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Apple Vision hint stub on Windows."""
        import sys as _sys
        if _sys.platform != "darwin":
            raise CorridorKeyError("Apple Vision hint requires macOS")
