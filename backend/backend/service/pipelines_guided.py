"""Guided pipelines — SAM 3.1 tracking and MatAnyone2 guided matting."""

from __future__ import annotations



import logging

import os

import time

from typing import Any, Callable, Optional



import cv2

import numpy as np



from ..clip_state import (

    ClipAsset,

    ClipEntry,

    ClipState,

)

from ..errors import (

    CorridorKeyError,

    GPURequiredError,

    JobCancelledError,

    WriteFailureError,

)

from ..frame_io import imread_unicode, imwrite_unicode

from .model_manager import _ActiveModel



logger = logging.getLogger(__name__)





class GuidedPipelinesMixin:

    """Mixin providing guided pipelines (SAM 3.1, MatAnyone2) for CorridorKeyService."""



    def run_sam3_track(

        self,

        clip: ClipEntry,

        input_is_linear: bool | None = None,

        job=None,

        on_progress: Optional[Callable[[str, int, int], None]] = None,

        on_warning: Optional[Callable[[str], None]] = None,

        on_status: Optional[Callable[[str], None]] = None,

    ) -> None:

        """Turn sparse point/box annotations into dense masks & alpha hints using SAM 3.1."""

        if clip.input_asset is None:

            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for SAM 3.1 tracking")

        if clip.input_asset.asset_type != "sequence":

            raise CorridorKeyError("SAM 3.1 tracking requires an extracted image sequence")



        selected_files = self._selected_sequence_files(clip)

        if not selected_files:

            raise CorridorKeyError(f"Clip '{clip.name}' has no input frames for SAM 3.1 tracking")



        def _status(message: str) -> None:

            logger.info(f"SAM 3.1 [{clip.name}]: {message}")

            if on_status:

                on_status(message)



        def _check_cancel() -> None:

            if job and job.is_cancelled:

                raise JobCancelledError(clip.name, 0)



        try:

            _status("Loading SAM 3.1 model...")

            with self._gpu_lock:

                tracker = self._get_sam3_tracker(

                    on_progress=(

                        None

                        if on_progress is None

                        else lambda current, total: on_progress(clip.name, current, total)

                    ),

                    on_status=on_status,

                )

            _check_cancel()



            _status("Loading frames...")

            sequence_is_linear = self._resolve_sequence_input_is_linear(clip, input_is_linear)

            named_frames = self._load_named_sequence_frames(

                clip.input_asset,

                selected_files,

                clip.name,

                gamma_correct_exr=sequence_is_linear,

                job=job,

                on_status=on_status,

            )

            _check_cancel()

            if not named_frames:

                raise CorridorKeyError(f"Clip '{clip.name}' has no readable input frames for SAM 3.1 tracking")



            start_index = clip.in_out_range.in_point if clip.in_out_range is not None else 0

            allowed_indices = list(range(start_index, start_index + len(selected_files)))

            from . import load_annotation_prompt_frames

            prompt_frames = load_annotation_prompt_frames(

                clip.root_path,

                allowed_indices=allowed_indices,

            )

            if not prompt_frames:

                raise CorridorKeyError(

                    f"Clip '{clip.name}' has no usable annotations for SAM 3.1 tracking"

                )



            from sam3_tracker import PromptFrame, SAM3NotInstalledError as SAM3NotInstalledError



            local_prompts = [
                PromptFrame(
                    frame_index=prompt.frame_index - start_index,
                    positive_points=prompt.positive_points,
                    negative_points=prompt.negative_points,
                    box=prompt.box,
                    hair_points=getattr(prompt, "hair_points", []),
                    negative_strokes=getattr(prompt, "negative_strokes", []),
                )
                for prompt in prompt_frames
            ]

            if not any(

                prompt.positive_points or prompt.box is not None

                for prompt in local_prompts

            ):

                message = "SAM 3.1 tracking requires at least one foreground point (1) or bounding box (R)"

                if on_warning:

                    on_warning(message)

                raise CorridorKeyError(message)



            _status("Running SAM 3.1 video tracker...")

            try:

                masks = tracker.track_video(

                    [frame for _, frame in named_frames],

                    local_prompts,

                    on_progress=(

                        None

                        if on_progress is None

                        else lambda current, total: on_progress(clip.name, current, total)

                    ),

                    on_status=on_status,

                    check_cancel=_check_cancel,

                )

            except SAM3NotInstalledError as exc:

                raise CorridorKeyError(str(exc)) from exc

            except ValueError as exc:

                message = str(exc)

                if on_warning:

                    on_warning(message)

                raise CorridorKeyError(message) from exc

            _check_cancel()



            mask_dir = os.path.join(clip.root_path, "VideoMamaMaskHint")
            sam_dir = os.path.join(clip.root_path, "SamMaskHint")
            alpha_dir = os.path.join(clip.root_path, "AlphaHint")

            os.makedirs(mask_dir, exist_ok=True)
            os.makedirs(sam_dir, exist_ok=True)
            os.makedirs(alpha_dir, exist_ok=True)

            for d in (mask_dir, sam_dir, alpha_dir):
                for fname in os.listdir(d):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        os.remove(os.path.join(d, fname))

            stems: list[str] = []
            for (fname, _), mask in zip(named_frames, masks):
                stem = os.path.splitext(fname)[0]
                stems.append(stem)
                out_mask_path = os.path.join(mask_dir, f"{stem}.png")
                out_sam_path = os.path.join(sam_dir, f"{stem}.png")
                out_alpha_path = os.path.join(alpha_dir, f"{stem}.png")
                if not imwrite_unicode(out_mask_path, mask):
                    raise WriteFailureError(clip.name, len(stems) - 1, out_mask_path)
                imwrite_unicode(out_sam_path, mask)
                imwrite_unicode(out_alpha_path, mask)



            self._write_mask_track_manifest(

                clip,

                source="sam3",

                frame_stems=stems,

                model_id=getattr(tracker, "model_id", "sam3.1-INT8-ConvRot"),

            )

            clip.mask_asset = ClipAsset(mask_dir, "sequence")
            clip.alpha_asset = ClipAsset(alpha_dir, "sequence")
            clip.find_assets()
            clip.state = ClipState.READY

            # Clear one_frame_hint manifest now that full tracking is authoritative
            ofh_path = os.path.join(clip.root_path, "one_frame_hint.json")
            if os.path.isfile(ofh_path):
                try:
                    os.remove(ofh_path)
                except OSError:
                    pass



            logger.info(

                "SAM 3.1 tracking complete for '%s': %d dense masks written to AlphaHint & VideoMamaMaskHint",

                clip.name,

                len(stems),

            )

        finally:

            self.release_vram(keep_corridorkey_cpu_cache=True)



    def preview_sam3_prompt(

        self,

        clip: ClipEntry,

        *,

        preferred_frame_index: int | None = None,

        input_is_linear: bool | None = None,

        job=None,

        on_progress: Optional[Callable[[str, int, int], None]] = None,

        on_warning: Optional[Callable[[str], None]] = None,

        on_status: Optional[Callable[[str], None]] = None,

    ) -> Optional[dict[str, Any]]:

        """Run a fast SAM 3.1 preview on one annotated frame without writing to disk."""

        if clip.input_asset is None:

            raise CorridorKeyError(f"Clip '{clip.name}' missing input asset for SAM 3.1 preview")

        if clip.input_asset.asset_type != "sequence":

            raise CorridorKeyError("SAM 3.1 preview requires an extracted image sequence")



        selected_files = self._selected_sequence_files(clip)

        if not selected_files:

            raise CorridorKeyError(f"Clip '{clip.name}' has no input frames for SAM 3.1 preview")



        def _status(message: str) -> None:

            logger.info(f"SAM 3.1 Preview [{clip.name}]: {message}")

            if on_status:

                on_status(message)



        def _check_cancel() -> None:

            if job and job.is_cancelled:

                raise JobCancelledError(clip.name, 0)



        try:

            _status("Loading SAM 3.1 model...")

            with self._gpu_lock:

                tracker = self._get_sam3_tracker(

                    on_progress=(

                        None

                        if on_progress is None

                        else lambda current, total: on_progress(clip.name, current, total)

                    ),

                    on_status=on_status,

                )

            _check_cancel()



            start_index = clip.in_out_range.in_point if clip.in_out_range is not None else 0

            allowed_indices = list(range(start_index, start_index + len(selected_files)))

            from . import load_annotation_prompt_frames

            prompt_frames = load_annotation_prompt_frames(

                clip.root_path,

                allowed_indices=allowed_indices,

            )

            if not prompt_frames:

                raise CorridorKeyError(

                    f"Clip '{clip.name}' has no usable annotations for SAM 3.1 preview"

                )



            prompt = next(

                (item for item in prompt_frames if item.frame_index == preferred_frame_index),

                prompt_frames[0],

            )

            if preferred_frame_index is not None and prompt.frame_index != preferred_frame_index and on_warning:

                on_warning(

                    f"No prompts on frame {preferred_frame_index + 1}; previewing annotated frame {prompt.frame_index + 1} instead."

                )



            local_index = prompt.frame_index - start_index

            if local_index < 0 or local_index >= len(selected_files):

                raise CorridorKeyError("Annotated frame is outside the selected in/out range")



            _status("Loading preview frame...")

            sequence_is_linear = self._resolve_sequence_input_is_linear(clip, input_is_linear)

            named_frames = self._load_named_sequence_frames(

                clip.input_asset,

                [selected_files[local_index]],

                clip.name,

                gamma_correct_exr=sequence_is_linear,

                job=job,

                on_status=on_status,

            )

            _check_cancel()

            if not named_frames:

                raise CorridorKeyError(f"Clip '{clip.name}' has no readable input frames for SAM 3.1 preview")



            from sam3_tracker import PromptFrame, SAM3NotInstalledError as SAM3NotInstalledError



            local_prompt = PromptFrame(
                frame_index=0,
                positive_points=prompt.positive_points,
                negative_points=prompt.negative_points,
                box=prompt.box,
                hair_points=getattr(prompt, "hair_points", []),
                negative_strokes=getattr(prompt, "negative_strokes", []),
            )



            _status("Previewing SAM 3.1 on annotated frame...")

            try:

                masks = tracker.track_video(

                    [named_frames[0][1]],

                    [local_prompt],

                    on_progress=(

                        None

                        if on_progress is None

                        else lambda current, total: on_progress(clip.name, current, total)

                    ),

                    on_status=on_status,

                    check_cancel=_check_cancel,

                )

            except SAM3NotInstalledError as exc:

                raise CorridorKeyError(str(exc)) from exc

            except ValueError as exc:

                message = str(exc)

                if on_warning:

                    on_warning(message)

                raise CorridorKeyError(message) from exc

            _check_cancel()



            mask = masks[0]
            stem = os.path.splitext(named_frames[0][0])[0]

            return {
                "kind": "sam3_preview",
                "clip_name": clip.name,
                "frame_index": prompt.frame_index,
                "frame_name": named_frames[0][0],
                "stem": stem,
                "frame_rgb": named_frames[0][1],
                "mask": mask,
                "fill": float((mask > 0).mean()),
            }

        finally:

            self.release_vram(keep_corridorkey_cpu_cache=True)



    def run_matanyone2(
        self,
        clip: ClipEntry,
        job=None,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
        on_warning: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Native CorridorKey 2.0 guided alpha pipeline."""
        self._run_corridorkey_v2_alpha_hint(
            clip,
            job=job,
            on_progress=on_progress,
            on_status=on_status,
            mask_hint_dir=(clip.mask_asset.path if clip.mask_asset else (clip.alpha_asset.path if clip.alpha_asset else None)),
        )

    def save_one_frame_hint(
        self,
        clip: ClipEntry,
        frame_index: int,
        mask: np.ndarray,
        stem: str,
        source: str = "sam3",
    ) -> None:
        """Save a single-frame mask as authoritative One Frame Hint for the clip."""
        import json
        from ..frame_io import imwrite_unicode

        alpha_dir = os.path.join(clip.root_path, "AlphaHint")
        target_sub = "SamMaskHint" if source == "sam3" else "BiRefNetHint"
        target_dir = os.path.join(clip.root_path, target_sub)
        os.makedirs(alpha_dir, exist_ok=True)
        os.makedirs(target_dir, exist_ok=True)

        # Clear prior hint files so only this single frame hint remains
        for d in (alpha_dir, target_dir):
            for fname in os.listdir(d):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".exr")):
                    try:
                        os.remove(os.path.join(d, fname))
                    except OSError:
                        pass

        mask_u8 = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8) if mask.dtype != np.uint8 else mask
        out_alpha_path = os.path.join(alpha_dir, f"{stem}.png")
        out_target_path = os.path.join(target_dir, f"{stem}.png")
        imwrite_unicode(out_alpha_path, mask_u8)
        imwrite_unicode(out_target_path, mask_u8)

        # Write one_frame_hint.json manifest
        manifest_path = os.path.join(clip.root_path, "one_frame_hint.json")
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({
                    "mode": "one_frame_hint",
                    "frame_index": int(frame_index),
                    "stem": stem,
                    "source": source,
                }, f, indent=2)
        except Exception as e:
            logger.warning("Failed writing one_frame_hint.json: %s", e)

        clip.alpha_asset = ClipAsset(alpha_dir, "sequence")
        if source == "sam3":
            clip.mask_asset = ClipAsset(target_dir, "sequence")
            vmama_dir = os.path.join(clip.root_path, "VideoMamaMaskHint")
            os.makedirs(vmama_dir, exist_ok=True)
            for fname in os.listdir(vmama_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    try:
                        os.remove(os.path.join(vmama_dir, fname))
                    except OSError:
                        pass
            imwrite_unicode(os.path.join(vmama_dir, f"{stem}.png"), mask_u8)

        try:
            clip.find_assets()
        except Exception:
            pass
        try:
            clip.transition_to(ClipState.READY)
        except Exception:
            pass

        logger.info(
            "Saved One Frame Hint (%s) on frame %d ('%s') for clip '%s'",
            source, frame_index + 1, stem, clip.name,
        )
