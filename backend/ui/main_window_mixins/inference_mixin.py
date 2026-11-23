from __future__ import annotations

import logging
import sys

import numpy as np
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Slot, QThread
from PySide6.QtGui import QImage

from . import _tr

from backend import (
    ClipState, JobType,
    PipelineRoute, classify_pipeline_route,
)
from ui.preview.frame_index import ViewMode
from ui.preview.display_transform import composite_over_backdrop, processed_rgba_to_qimage
from ui.workers.gpu_job_worker import create_job_snapshot

logger = logging.getLogger(__name__)


class InferenceMixin:
    """Inference control, parameter changes, and GPU job submission for MainWindow."""

    _last_live_render: tuple[str, int, dict] | None = None
    _live_render_seq: int = 0
    _last_input_is_linear: bool | None = None

    @Slot()
    def _on_params_changed(self) -> None:
        """Handle parameter change — persist per-clip params and debounce live render."""
        self._remember_current_clip_input_color_space()
        params = self._param_panel.get_params()
        if self._current_clip is not None and self._current_clip.root_path:
            try:
                from backend.project import save_inference_params
                save_inference_params(self._current_clip.root_path, params.to_dict())
            except Exception as exc:
                logger.debug("Per-clip inference params save failed: %s", exc)
        is_lin = bool(params.input_is_linear)
        self._dual_viewer.set_input_exr_is_linear(is_lin)
        if self._current_clip is not None and self._last_input_is_linear != is_lin:
            self._last_input_is_linear = is_lin
            self._refresh_input_thumbnail(
                self._current_clip,
                input_is_linear=is_lin,
            )
        if self._param_panel.live_preview_enabled:
            self._reprocess_timer.start()

    def _load_inference_params_for_clip(self, clip) -> None:
        """Restore persisted per-clip InferenceParams onto ParameterPanel."""
        if clip is None or not getattr(clip, "root_path", ""):
            return
        try:
            from backend.project import load_inference_params
            from backend.service.core import InferenceParams
            saved = load_inference_params(clip.root_path)
            if saved:
                restored = InferenceParams.from_dict(saved)
                self._param_panel.set_params(restored)
                self._last_input_is_linear = bool(restored.input_is_linear)
        except Exception as exc:
            logger.debug("Per-clip inference params load failed: %s", exc)

    @Slot(str)
    def _on_output_mode_changed(self, mode_value: str) -> None:
        """Keep export-strip thumbnails aligned and preserve active live previews across tabs."""
        try:
            mode = ViewMode(mode_value)
        except ValueError:
            return

        for clip in self._clip_model.clips:
            if clip.state == ClipState.COMPLETE:
                self._refresh_export_thumbnail(clip)

        iv = self._dual_viewer.input_viewer
        stem_idx = iv.current_stem_index
        clip = getattr(self, "_current_clip", None)

        # Preserve in-memory Live Render result across render tabs (FG / MATTE / COMP / PROC)
        if (
            clip is not None
            and self._param_panel.live_preview_enabled
            and mode in (ViewMode.FG, ViewMode.MATTE, ViewMode.COMP, ViewMode.PROCESSED)
            and self._last_live_render is not None
        ):
            lr_root, lr_stem, lr_res = self._last_live_render
            if lr_root == clip.root_path and lr_stem == stem_idx:
                self._render_live_result_for_current_mode(lr_res, auto_switch_mode=False)
                return

        # Otherwise preserve SAM 3.1 live interactive preview across prompt tabs
        if iv.annotation_model.has_annotations(stem_idx):
            if hasattr(self, "_last_sam3_preview") and self._last_sam3_preview is not None:
                p_stem, frame_rgb, mask = self._last_sam3_preview
                if p_stem == stem_idx:
                    qimg = self._render_sam3_preview_for_mode(frame_rgb, mask)
                    self._dual_viewer.show_reprocess_preview(qimg)
                    return
            if hasattr(self, "_refresh_live_sam3_preview"):
                self._refresh_live_sam3_preview(stem_idx)

    @Slot(str)
    def _on_output_backdrop_changed(self, _backdrop: str) -> None:
        """Refresh active live render or SAM 3.1 preview when backdrop changes."""
        mode = self._dual_viewer.current_output_mode
        if mode not in (ViewMode.COMP, ViewMode.PROCESSED):
            return
        stem_idx = self._dual_viewer.input_viewer.current_stem_index
        clip = getattr(self, "_current_clip", None)
        if (
            clip is not None
            and self._param_panel.live_preview_enabled
            and self._last_live_render is not None
        ):
            lr_root, lr_stem, lr_res = self._last_live_render
            if lr_root == clip.root_path and lr_stem == stem_idx:
                self._render_live_result_for_current_mode(lr_res, auto_switch_mode=False)
                return
        if mode == ViewMode.COMP and hasattr(self, "_last_sam3_preview") and self._last_sam3_preview is not None:
            p_stem, frame_rgb, mask = self._last_sam3_preview
            if stem_idx == p_stem:
                qimg = self._render_sam3_preview_for_mode(frame_rgb, mask)
                self._dual_viewer.show_reprocess_preview(qimg)

    @Slot(str)
    def _on_screen_color_changed(self, color: str) -> None:
        """Update the UI accent color when BG Color changes.

        Reloads the QSS with the appropriate accent palette and
        updates the brand mark text color.
        """
        from ui.theme import load_stylesheet, ACCENT_COLORS

        # Resolve "auto" to detected color for the current clip
        effective = color
        if color == "auto" and self._current_clip is not None:
            effective = getattr(self._current_clip, '_screen_color_cache', 'green')
        if effective not in ("green", "blue"):
            effective = "green"

        # Swap QSS accent colors
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.setStyleSheet(load_stylesheet(screen_color=effective))

        # Update brand mark text — hyphen and KEY carry the accent
        palette = ACCENT_COLORS[effective]
        self._current_accent = palette["accent"]
        self._brand_label.setText(
            f'<span style="color:#FFF203;">CORRIDOR</span>'
            f'<span style="color:{palette["accent"]};">KEY</span>'
            f'<span style="color:#FFF203;font-size:11px;"> 2.0</span>'
        )

    @Slot(bool)
    def _on_live_preview_toggled(self, checked: bool) -> None:
        """Toggle single-frame in-memory Live Render and keep status bar + panel synced."""
        self._status_bar.set_live_render_checked(checked)
        self._param_panel.set_live_preview_checked(checked)
        if checked:
            self._status_bar.set_message(
                _tr("Live Render ON — updates current frame in memory only (nothing written to disk).")
            )
            self._reprocess_timer.start()
        else:
            self._reprocess_timer.stop()
            self._live_render_seq += 1
            self._last_live_render = None
            self._status_bar.set_live_render_busy(False)
            self._dual_viewer.set_busy(False)
            self._dual_viewer.set_live_render_modes(False)
            self._dual_viewer.clear_live_preview()
            stem_idx = self._dual_viewer.current_stem_index
            if hasattr(self, "_refresh_live_sam3_preview"):
                self._refresh_live_sam3_preview(stem_idx)
            elif stem_idx >= 0:
                self._dual_viewer.output_viewer.navigate_to_frame(stem_idx)

    def _do_reprocess(self) -> None:
        """Submit a zero-disk PREVIEW_REPROCESS job through the GPU queue."""
        if not self._param_panel.live_preview_enabled:
            return
        if self._current_clip is None or self._current_clip.input_asset is None:
            return
        clip = self._current_clip
        if clip.state not in (ClipState.RAW, ClipState.MASKED, ClipState.READY, ClipState.COMPLETE):
            return
        if getattr(self._service.job_queue, "has_non_preview_running", False):
            return

        frame_idx = max(0, self._dual_viewer.current_stem_index)
        params = self._param_panel.get_params()
        self._live_render_seq += 1
        job = create_job_snapshot(clip, params, job_type=JobType.PREVIEW_REPROCESS)
        job.params["_frame_index"] = frame_idx
        job.params["_live_seq"] = self._live_render_seq
        job.params["_clip_root"] = clip.root_path

        if self._service.job_queue.submit(job):
            self._status_bar.set_live_render_busy(True)
            self._dual_viewer.set_busy(True, _tr("Live Rendering..."))
            self._start_worker_if_needed(quiet=True)

    def _render_live_result_for_current_mode(
        self,
        result: dict,
        *,
        auto_switch_mode: bool = True,
    ) -> None:
        """Render a single-frame Live Render result dictionary according to current view mode."""
        if "comp" not in result:
            return
        self._dual_viewer.set_live_render_modes(True)
        if "alpha" in result and hasattr(self._dual_viewer, "set_alpha_probe_array"):
            self._dual_viewer.set_alpha_probe_array(result["alpha"])
        mode = self._dual_viewer.current_output_mode
        if mode in (ViewMode.INPUT, ViewMode.MASK, ViewMode.ALPHA):
            if not auto_switch_mode:
                return
            self._dual_viewer.output_viewer.set_view_mode(ViewMode.COMP.value)
            mode = ViewMode.COMP

        if mode == ViewMode.MATTE and "alpha" in result:
            arr = result["alpha"]
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            display = np.power(np.clip(arr, 0.0, 1.0), 0.85)
            gray8 = (display * 255.0).astype(np.uint8)
            rgb = np.stack([gray8, gray8, gray8], axis=2)
        elif mode == ViewMode.FG and "fg" in result:
            rgb = (np.clip(result["fg"], 0.0, 1.0) * 255.0).astype(np.uint8)
        elif mode == ViewMode.PROCESSED and "processed" in result:
            backdrop = getattr(self._dual_viewer, "current_backdrop", "checker")
            qimg = processed_rgba_to_qimage(result["processed"], backdrop=backdrop)
            self._dual_viewer.show_reprocess_preview(qimg)
            return
        else:
            backdrop = getattr(self._dual_viewer, "current_backdrop", "checker")
            f_idx = int(result.get("frame_index", 0) or 0)
            if backdrop != "checker" and "fg" in result and "alpha" in result:
                rgb = composite_over_backdrop(result["fg"], result["alpha"], backdrop=backdrop, frame_index=f_idx)
            else:
                rgb = (np.clip(result["comp"], 0.0, 1.0) * 255.0).astype(np.uint8)

        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._dual_viewer.show_reprocess_preview(qimg)

    @Slot(str, object)
    def _on_reprocess_result(self, job_id: str, result: object) -> None:
        """Handle live reprocess result — display preview matching current view mode."""
        if not isinstance(result, dict):
            return

        if result.get("kind") == "one_frame_hint":
            clip_name = result.get("clip_name")
            if not isinstance(clip_name, str):
                return
            clip = next((c for c in self._clip_model.clips if c.name == clip_name), None)
            if clip is None or self._current_clip is None or self._current_clip.name != clip_name:
                return
            stem_idx = int(result.get("frame_index", 0))
            clip.find_assets()
            self._clip_model.update_clip_state(clip.name, clip.state)
            self._io_tray.refresh()
            self._refresh_button_state()
            self._dual_viewer.output_viewer.refresh_available_assets()
            self._sync_view_mode_buttons()
            if hasattr(self, "_update_annotation_info"):
                self._update_annotation_info()
            self._dual_viewer.output_viewer.set_view_mode(ViewMode.ALPHA.value)
            self._dual_viewer.output_viewer.refresh_current_frame()
            self._dual_viewer.output_viewer.navigate_to_frame(stem_idx)
            src = result.get("source", "BiRefNet")
            self._status_bar.set_message(
                _tr("One Frame Hint ready for frame %d (%s). Click Run to infer or Track to animate across video.") % (stem_idx + 1, src)
            )
            return

        if result.get("kind") == "sam3_preview":
            clip_name = result.get("clip_name")
            if not isinstance(clip_name, str):
                return
            clip = next((c for c in self._clip_model.clips if c.name == clip_name), None)
            if clip is None or self._current_clip is None or self._current_clip.name != clip_name:
                return

            frame_rgb = result.get("frame_rgb")
            mask = result.get("mask")
            stem = result.get("stem")
            if not isinstance(frame_rgb, np.ndarray) or not isinstance(mask, np.ndarray):
                return

            qimg = self._sam3_preview_qimage(frame_rgb, mask)
            self._dual_viewer.show_reprocess_preview(qimg)

            frame_number = int(result.get("frame_index", 0)) + 1
            fill_pct = float(result.get("fill", 0.0)) * 100.0
            actual_stem = stem or f"frame_{frame_number - 1:06d}"

            # Direct One Frame Hint button was clicked
            if getattr(self, "_sam3_one_frame_mode", False):
                self._sam3_one_frame_mode = False
                self._service.save_one_frame_hint(clip, frame_number - 1, mask, stem=actual_stem, source="sam3")
                clip.find_assets()
                self._clip_model.update_clip_state(clip.name, clip.state)
                self._io_tray.refresh()
                self._refresh_button_state()
                self._dual_viewer.output_viewer.refresh_available_assets()
                self._sync_view_mode_buttons()
                if hasattr(self, "_update_annotation_info"):
                    self._update_annotation_info()
                self._dual_viewer.output_viewer.set_view_mode(ViewMode.ALPHA.value)
                self._dual_viewer.output_viewer.refresh_current_frame()
                self._dual_viewer.output_viewer.navigate_to_frame(frame_number - 1)
                self._status_bar.set_message(
                    _tr("Saved frame %d as One Frame Hint (SAM 3.1). Click Run to infer or Track Mask to animate.") % frame_number
                )
                return

            # Opened via TRACK MASK (SAM 3.1): offer Track All vs One Frame Hint vs Keep Editing
            msg = QMessageBox(self)
            msg.setWindowTitle(_tr("SAM 3.1 Mask Preview"))
            msg.setText(
                _tr("SAM 3.1 preview on frame %d covers %.1f%% of the frame.\n\n"
                    "Choose how to proceed:") % (frame_number, fill_pct)
            )
            track_btn = msg.addButton(_tr("Track All Frames (Animate)"), QMessageBox.AcceptRole)
            one_frame_btn = msg.addButton(_tr("Use as One Frame Hint"), QMessageBox.ActionRole)
            cancel_btn = msg.addButton(_tr("Keep Editing / Cancel"), QMessageBox.RejectRole)
            msg.setDefaultButton(track_btn)
            msg.exec()
            clicked = msg.clickedButton()

            if clicked == track_btn:
                self._submit_sam3_track_job(clip)
            elif clicked == one_frame_btn:
                self._service.save_one_frame_hint(clip, frame_number - 1, mask, stem=actual_stem, source="sam3")
                clip.find_assets()
                self._clip_model.update_clip_state(clip.name, clip.state)
                self._io_tray.refresh()
                self._refresh_button_state()
                self._dual_viewer.output_viewer.refresh_available_assets()
                self._sync_view_mode_buttons()
                if hasattr(self, "_update_annotation_info"):
                    self._update_annotation_info()
                self._dual_viewer.output_viewer.set_view_mode(ViewMode.ALPHA.value)
                self._dual_viewer.output_viewer.refresh_current_frame()
                self._dual_viewer.output_viewer.navigate_to_frame(frame_number - 1)
                self._status_bar.set_message(
                    _tr("Saved frame %d as One Frame Hint (SAM 3.1). Click Track Mask anytime to propagate across all frames.") % frame_number
                )
            else:
                self._status_bar.set_message(_tr("Track preview ready. Refine paint strokes and run Track Mask again."))
            return

        if "comp" not in result:
            return

        self._status_bar.set_live_render_busy(False)
        self._dual_viewer.set_busy(False)
        if not self._param_panel.live_preview_enabled:
            return
        res_seq = result.get("_live_seq")
        if res_seq is not None and res_seq != self._live_render_seq:
            return
        clip = getattr(self, "_current_clip", None)
        if clip is None:
            return
        res_root = result.get("clip_root")
        if res_root and res_root != clip.root_path:
            return
        res_idx = result.get("frame_index")
        cur_idx = self._dual_viewer.current_stem_index
        if res_idx is not None and cur_idx >= 0 and int(res_idx) != int(cur_idx):
            return

        stem_key = int(res_idx) if res_idx is not None else max(0, cur_idx)
        self._last_live_render = (clip.root_path, stem_key, result)
        self._render_live_result_for_current_mode(result, auto_switch_mode=True)

    # ── Inference Control ──

    @Slot()
    def _on_run_inference(self) -> None:
        # If multiple clips are selected, dispatch to pipeline
        if self._io_tray.selected_count() > 1:
            self._on_run_pipeline()
            return

        if self._current_clip is None:
            return

        clip = self._current_clip
        if clip.state in (ClipState.EXTRACTING, ClipState.ERROR):
            QMessageBox.warning(
                self, _tr("Not Ready"),
                _tr("Clip '%s' is in %s state.\n"
                    "Wait for extraction to finish before processing.") % (clip.name, clip.state.value),
            )
            return

        # For COMPLETE, RAW, or MASKED clips, ensure state is READY for processing
        if clip.state in (ClipState.COMPLETE, ClipState.RAW, ClipState.MASKED):
            clip.transition_to(ClipState.READY)
            self._clip_model.update_clip_state(clip.name, ClipState.READY)

        params = self._param_panel.get_params()
        job = create_job_snapshot(clip, params, resume=False)

        # Store output config in job params
        output_config = self._param_panel.get_output_config()
        job.params["_output_config"] = output_config

        # Pass in/out frame range (GVM always processes full clip)
        if clip.in_out_range:
            job.params["_frame_range"] = (
                clip.in_out_range.in_point,
                clip.in_out_range.out_point,
            )

        if not self._service.job_queue.submit(job):
            QMessageBox.information(self, _tr("Duplicate"), _tr("'%s' is already queued.") % clip.name)
            return

        self._start_worker_if_needed(job.id)

    @Slot()
    def _on_resume_inference(self) -> None:
        """Resume inference — skip already-processed frames, process full clip."""
        if self._current_clip is None:
            return

        clip = self._current_clip
        if clip.state not in (ClipState.READY, ClipState.COMPLETE):
            return

        # For COMPLETE clips, transition back to READY
        if clip.state == ClipState.COMPLETE:
            clip.transition_to(ClipState.READY)

        params = self._param_panel.get_params()
        job = create_job_snapshot(clip, params, resume=True)

        output_config = self._param_panel.get_output_config()
        job.params["_output_config"] = output_config
        # Resume always processes full clip — no in/out range

        if not self._service.job_queue.submit(job):
            QMessageBox.information(self, _tr("Duplicate"), _tr("'%s' is already queued.") % clip.name)
            return

        self._start_worker_if_needed(job.id)

    def _refresh_button_state(self) -> None:
        """Update run/resume button state based on current clip."""
        clip = self._current_clip
        batch_count = self._io_tray.selected_count()
        if clip is None:
            self._status_bar.update_button_state(
                can_run=False, has_partial=False, has_in_out=False,
            )
            return
        can_run = clip.state in (ClipState.RAW, ClipState.MASKED, ClipState.READY, ClipState.COMPLETE)
        self._status_bar.update_button_state(
            can_run=can_run,
            has_partial=clip.completed_frame_count() > 0,
            has_in_out=clip.in_out_range is not None,
            batch_count=batch_count if batch_count > 1 else 0,
        )

    @Slot()
    def _on_run_all_ready(self) -> None:
        """Queue all READY clips for inference.

        Sorts clips by screen_color so all green clips run first, then
        all blue clips (or vice versa). This minimizes checkpoint reloads
        in mixed-color projects.
        """
        ready_clips = self._clip_model.clips_by_state(ClipState.READY)
        if not ready_clips:
            QMessageBox.information(self, _tr("No Clips"), _tr("No READY clips to process."))
            return

        params = self._param_panel.get_params()
        output_config = self._param_panel.get_output_config()

        # Sort by screen_color to minimize model reloads
        ready_clips = sorted(
            ready_clips,
            key=lambda c: getattr(c, '_screen_color_cache', 'green'),
        )

        queued = 0
        for clip in ready_clips:
            job = create_job_snapshot(clip, params)
            job.params["_output_config"] = output_config
            if clip.in_out_range:
                job.params["_frame_range"] = (
                    clip.in_out_range.in_point,
                    clip.in_out_range.out_point,
                )
            if self._service.job_queue.submit(job):
                queued += 1

        if queued > 0:
            first_job = self._service.job_queue.next_job()
            self._start_worker_if_needed(first_job.id if first_job else None)
            logger.info(f"Batch queued: {queued} clips")

    def _on_run_pipeline(self) -> None:
        """Full pipeline: classify each selected clip and queue appropriate jobs.

        Routes per clip (see PipelineRoute):
        - INFERENCE_ONLY: queue inference directly
        - GVM_PIPELINE: queue GVM, auto-chain inference on completion
        - VIDEOMAMA_PIPELINE: track dense masks, queue VideoMaMa, auto-chain inference
        - VIDEOMAMA_INFERENCE: queue VideoMaMa, auto-chain inference
        - SKIP: skip clips in EXTRACTING/ERROR state
        """
        selected = self._io_tray.get_selected_clips()
        if not selected:
            return
        self._auto_save_annotations()

        params = self._param_panel.get_params()
        output_config = self._param_panel.get_output_config()

        # Classify all clips
        routes: dict[str, PipelineRoute] = {}
        for clip in selected:
            route = classify_pipeline_route(clip)
            if route != PipelineRoute.SKIP:
                routes[clip.name] = route

        if not routes:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self, _tr("Nothing to Process"),
                _tr("No selected clips are in a processable state."),
            )
            return

        self._pipeline_steps.clear()

        queued = 0
        first_job_id = None

        for clip in selected:
            route = routes.get(clip.name)
            if route is None:
                continue
            job = None
            next_steps: list[JobType] = []

            if route == PipelineRoute.BIREFNET_PIPELINE:
                job = create_job_snapshot(clip, job_type=JobType.BIREFNET_ALPHA)
                next_steps = [JobType.INFERENCE]
            elif route == PipelineRoute.SAM3_PIPELINE:
                job = create_job_snapshot(clip, params, job_type=JobType.SAM3_TRACK)
                next_steps = [JobType.INFERENCE]
            elif route == PipelineRoute.MATANYONE2_INFERENCE:
                job = create_job_snapshot(clip, job_type=JobType.MATANYONE2_ALPHA)
                next_steps = [JobType.INFERENCE]
            elif route == PipelineRoute.INFERENCE_ONLY:
                if clip.state == ClipState.COMPLETE:
                    clip.transition_to(ClipState.READY)
                job = create_job_snapshot(clip, params)
                job.params["_output_config"] = output_config
                if clip.in_out_range:
                    job.params["_frame_range"] = (
                        clip.in_out_range.in_point,
                        clip.in_out_range.out_point,
                    )

            if job is None:
                continue
            if self._service.job_queue.submit(job):
                clip.set_processing(True)
                if next_steps:
                    self._pipeline_steps[clip.name] = next_steps
                if first_job_id is None:
                    first_job_id = job.id
                queued += 1

        if queued > 0:
            self._start_worker_if_needed(first_job_id, job_label="Pipeline")
            logger.info("Pipeline submitted %d jobs", queued)

    def _start_worker_if_needed(
        self,
        first_job_id: str | None = None,
        job_label: str = "Inference",
        quiet: bool = False,
    ) -> None:
        """Ensure GPU worker is running and optionally expand queue/reset status bar."""
        if first_job_id and self._active_job_id is None:
            self._active_job_id = first_job_id

        if not self._gpu_worker.isRunning():
            self._gpu_worker.start(QThread.HighPriority)
        else:
            self._gpu_worker.wake()

        if quiet:
            self._queue_panel.refresh()
            return

        self._force_stop_armed = False
        self._status_bar.set_running(True)
        self._status_bar.set_stop_button_mode(force=False)
        self._status_bar.reset_progress()
        self._status_bar.start_job_timer(label=job_label)
        self._queue_panel.refresh()
        if self._queue_panel._collapsed:
            self._queue_panel.toggle_collapsed()  # auto-expand when job starts

    def _sync_view_mode_buttons(self) -> None:
        """Refresh output viewer available assets and view mode buttons."""
        if hasattr(self, "_dual_viewer"):
            ov = getattr(self._dual_viewer, "output_viewer", None)
            if ov is not None and hasattr(ov, "refresh_available_assets"):
                ov.refresh_available_assets()

    # ── 24-Frame In-Memory RAM Preview ─────────────────────────────────────────

    def _on_ram_preview_requested(self) -> None:
        """Render up to 24 frames around playhead in memory without writing to disk."""
        worker = getattr(self, "_ram_worker", None)
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            self._status_bar.set_ram_preview_busy(False)
            self._dual_viewer.set_busy(False)
            self._status_bar.set_message(_tr("RAM Preview cancelled."))
            return

        clip = getattr(self, "_current_clip", None)
        if clip is None or clip.input_asset is None:
            return

        total = max(1, clip.input_asset.frame_count)
        cur = max(0, self._dual_viewer.current_stem_index)
        scrubber = getattr(self._dual_viewer, "_scrubber", None)
        if scrubber is not None and scrubber.has_range:
            lo, hi = scrubber.get_in_out()
            start = max(0, int(lo if lo is not None else cur))
            end = min(total - 1, min(start + 23, int(hi if hi is not None else total - 1)))
        else:
            start = cur
            end = min(total - 1, start + 23)
            if end - start < 23 and start > 0:
                start = max(0, end - 23)

        indices = list(range(start, end + 1))
        if not indices:
            return

        params = self._param_panel.get_params()
        mode = self._dual_viewer.current_output_mode
        backdrop = getattr(self._dual_viewer, "current_backdrop", "checker")

        worker_obj = _create_ram_preview_worker(
            self._service,
            clip,
            params,
            indices,
            mode,
            backdrop,
            parent=self,
        )
        self._ram_worker = worker_obj
        self._ram_frames: dict[int, QImage] = {}
        self._status_bar.set_ram_preview_busy(True, _tr("RAM 0/%d") % len(indices))
        self._dual_viewer.set_busy(True, _tr("RAM Preview 0/%d...") % len(indices), progress=0.0)

        def _on_ram_frame(idx: int, qimg: object) -> None:
            if isinstance(qimg, QImage):
                self._ram_frames[idx] = qimg
            done = len(self._ram_frames)
            tot = max(1, len(indices))
            self._status_bar.set_ram_preview_busy(
                True, _tr("RAM %d/%d") % (done, len(indices))
            )
            self._dual_viewer.set_busy(
                True,
                _tr("RAM Preview %d/%d...") % (done, len(indices)),
                progress=float(done) / float(tot),
            )

        def _on_ram_done() -> None:
            self._status_bar.set_ram_preview_busy(False)
            self._dual_viewer.set_busy(False)
            w_old = self._ram_worker
            self._ram_worker = None
            if w_old is not None:
                w_old.deleteLater()
            if self._ram_frames:
                self._play_ram_preview_buffer(sorted(self._ram_frames.keys()))

        worker_obj.frame_ready.connect(_on_ram_frame)
        worker_obj.finished.connect(_on_ram_done)
        worker_obj.start()

    def _play_ram_preview_buffer(self, frame_indices: list[int]) -> None:
        from PySide6.QtCore import QTimer
        if not frame_indices:
            return
        if not hasattr(self, "_ram_play_timer") or self._ram_play_timer is None:
            self._ram_play_timer = QTimer(self)
        timer = self._ram_play_timer
        try:
            timer.timeout.disconnect()
        except Exception:
            pass

        fps = getattr(self._dual_viewer._scrubber, "_fps", 24.0) or 24.0
        state = {"pos": 0}
        timer.setInterval(max(16, int(round(1000.0 / float(fps)))))

        def _step() -> None:
            if not getattr(self, "_ram_frames", None):
                timer.stop()
                return
            idx = frame_indices[state["pos"] % len(frame_indices)]
            qimg = self._ram_frames.get(idx)
            if qimg is not None:
                self._dual_viewer.input_viewer.navigate_to_frame(idx)
                self._dual_viewer.output_viewer.navigate_to_frame(idx)
                self._dual_viewer.show_reprocess_preview(qimg)
            state["pos"] += 1
            if state["pos"] >= len(frame_indices):
                timer.stop()
                self._status_bar.set_message(_tr("RAM Preview complete (%d frames cached in memory).") % len(frame_indices))

        timer.timeout.connect(_step)
        self._status_bar.set_message(_tr("Playing %d-frame RAM Preview...") % len(frame_indices))
        timer.start()


def _create_ram_preview_worker(service, clip, params, indices, mode, backdrop, parent=None):
    from PySide6.QtCore import Signal

    class _RamPreviewWorker(QThread):
        frame_ready = Signal(int, object)

        def run(self) -> None:
            for idx in indices:
                if self.isInterruptionRequested():
                    break
                try:
                    res = service.reprocess_single_frame(clip, params, idx)
                    if not isinstance(res, dict) or "comp" not in res:
                        continue
                    if mode == ViewMode.MATTE and "alpha" in res:
                        arr = res["alpha"]
                        if arr.ndim == 3:
                            arr = arr[:, :, 0]
                        disp = np.power(np.clip(arr, 0.0, 1.0), 0.85)
                        g8 = (disp * 255.0).astype(np.uint8)
                        rgb = np.stack([g8, g8, g8], axis=2)
                        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
                    elif mode == ViewMode.FG and "fg" in res:
                        rgb = (np.clip(res["fg"], 0.0, 1.0) * 255.0).astype(np.uint8)
                        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
                    elif mode == ViewMode.PROCESSED and "processed" in res:
                        qimg = processed_rgba_to_qimage(res["processed"], backdrop=backdrop)
                    else:
                        if backdrop != "checker" and "fg" in res and "alpha" in res:
                            rgb = composite_over_backdrop(res["fg"], res["alpha"], backdrop=backdrop)
                        else:
                            rgb = (np.clip(res["comp"], 0.0, 1.0) * 255.0).astype(np.uint8)
                        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
                    self.frame_ready.emit(idx, qimg)
                except Exception:
                    continue

    return _RamPreviewWorker(parent)

