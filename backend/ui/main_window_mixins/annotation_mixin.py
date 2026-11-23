from __future__ import annotations







import os
import shutil
import logging
from collections import OrderedDict
from typing import Any







import cv2



import numpy as np



from PySide6.QtWidgets import QMessageBox



from PySide6.QtGui import QImage
from PySide6.QtCore import QTimer, Slot







from . import _tr







from backend import ClipEntry







logger = logging.getLogger(__name__)











class AnnotationMixin:



    """Annotation painting, SAM 3.1 tracking, and mask management for MainWindow."""







    def _is_chroma_key_active(self) -> bool:



        """True when the Chroma Key section is expanded (button checked)."""



        return self._param_panel._chroma_key_btn.isChecked()







    def _toggle_annotation_brush(self) -> None:
        """Hotkey B: toggle brush mode for SAM 3.1 (Alt=Negative)."""
        iv = self._dual_viewer.input_viewer
        ov = self._dual_viewer.output_viewer
        if iv.annotation_mode == "brush":
            iv.set_annotation_mode(None)
            ov.set_annotation_mode(None)
            self._dual_viewer.set_holdout_active(False)
        else:
            self._deactivate_eyedropper_if_active()
            iv.set_annotation_mode("brush")
            ov.set_annotation_mode(None)
            self._dual_viewer.set_holdout_active(False)

    def _toggle_annotation_point(self) -> None:
        """Hotkey P: toggle point prompt mode for SAM 3.1 (Alt=Negative)."""
        iv = self._dual_viewer.input_viewer
        ov = self._dual_viewer.output_viewer
        if iv.annotation_mode == "point":
            iv.set_annotation_mode(None)
            ov.set_annotation_mode(None)
            self._dual_viewer.set_holdout_active(False)
        else:
            self._deactivate_eyedropper_if_active()
            iv.set_annotation_mode("point")
            ov.set_annotation_mode(None)
            self._dual_viewer.set_holdout_active(False)

    def _toggle_annotation_eraser(self) -> None:
        """Hotkey E: toggle eraser mode for SAM 3.1."""
        iv = self._dual_viewer.input_viewer
        ov = self._dual_viewer.output_viewer
        if iv.annotation_mode == "eraser":
            iv.set_annotation_mode(None)
            ov.set_annotation_mode(None)
            self._dual_viewer.set_holdout_active(False)
        else:
            self._deactivate_eyedropper_if_active()
            iv.set_annotation_mode("eraser")
            ov.set_annotation_mode(None)
            self._dual_viewer.set_holdout_active(False)

    def _toggle_annotation_fg(self) -> None:
        """Hotkey 1: toggle foreground brush for SAM 3.1."""
        self._toggle_annotation_brush()

    def _toggle_annotation_bg(self) -> None:
        """Hotkey 2: toggle background / cut brush for SAM 3.1."""
        iv = self._dual_viewer.input_viewer
        ov = self._dual_viewer.output_viewer
        if iv.annotation_mode == "bg":
            iv.set_annotation_mode(None)
            ov.set_annotation_mode(None)
            self._dual_viewer.set_holdout_active(False)
        else:
            self._deactivate_eyedropper_if_active()
            iv.set_annotation_mode("bg")
            ov.set_annotation_mode(None)
            self._dual_viewer.set_holdout_active(False)

    def _toggle_annotation_hair(self) -> None:
        """Hotkey H: toggle hair & soft detail brush for SAM 3.1."""
        iv = self._dual_viewer.input_viewer
        ov = self._dual_viewer.output_viewer
        if iv.annotation_mode == "hair":
            iv.set_annotation_mode(None)
            ov.set_annotation_mode(None)
            self._dual_viewer.set_holdout_active(False)
        else:
            self._deactivate_eyedropper_if_active()
            iv.set_annotation_mode("hair")
            ov.set_annotation_mode(None)
            self._dual_viewer.set_holdout_active(False)







    def _deactivate_eyedropper_if_active(self) -> None:



        """Turn off the eyedropper if it's currently on."""



        ed_btn = self._param_panel._eyedropper_btn



        if ed_btn.isChecked():



            ed_btn.setChecked(False)







    def _cycle_fg_color(self) -> None:



        """Hotkey C: cycle foreground annotation color (green/blue)."""



        from ui.widgets.annotation_overlay import cycle_fg_color



        name = cycle_fg_color()



        self._dual_viewer.input_viewer.update()



        self._show_toast(_tr("Foreground color: %s") % name)







    def _auto_save_annotations(self) -> None:



        """Auto-save annotation strokes to disk after changes."""



        if self._current_clip is not None:



            iv = self._dual_viewer.input_viewer



            iv.annotation_model.save(self._current_clip.root_path)



            manifest_path = os.path.join(



                self._current_clip.root_path,



                ".corridorkey_mask_manifest.json",



            )



            if os.path.isfile(manifest_path):



                os.remove(manifest_path)



            _has_mask = self._clip_has_videomama_ready_mask(self._current_clip)



            self._param_panel.set_videomama_enabled(_has_mask)



            self._param_panel.set_matanyone2_enabled(_has_mask)







    def _auto_save_holdout(self) -> None:



        """Auto-save holdout strokes to disk after changes."""



        if self._current_clip is not None:



            iv = self._dual_viewer.input_viewer



            iv.holdout_model.save(self._current_clip.root_path)







    def _clip_has_videomama_ready_mask(self, clip: ClipEntry | None) -> bool:



        """True when the clip has a dense mask track that alpha generators can consume."""



        if clip is None or clip.mask_asset is None:



            return False



        # mask_asset is set only when VideoMamaMaskHint/ has actual frames —



        # that's sufficient to enable alpha generators regardless of manifest.



        return True







    def _undo_annotation(self) -> None:



        """Ctrl+Z: undo last stroke / mask paint edit / polygon edit."""



        iv = self._dual_viewer.input_viewer

        if iv.annotation_mode in ("mask_paint_fg", "mask_paint_bg") and self._undo_mask_paint():
            return
        if iv.annotation_mode in ("poly_garbage", "poly_core") and self._undo_polygon_edit():
            return

        use_holdout = (
            iv.annotation_mode == "holdout"
            or (
                self._is_chroma_key_active()
                and iv.holdout_model.can_undo(0)
                and not (iv.current_stem_index >= 0 and iv.annotation_model.can_undo(iv.current_stem_index))
            )
        )

        if use_holdout:

            if iv.holdout_model.undo(0):

                iv._split_view.update()

                self._auto_save_holdout()

                self._schedule_chroma_key_preview()
                return

        elif iv.current_stem_index >= 0:

            if iv.annotation_model.undo(iv.current_stem_index):

                iv._split_view.update()

                self._auto_save_annotations()
                self._update_annotation_info()
                self._refresh_live_sam3_preview(iv.current_stem_index, force_recompute=True)
                return

        if self._undo_mask_paint():
            return
        self._undo_polygon_edit()

    def _redo_annotation(self) -> None:

        """Ctrl+Y / Ctrl+Shift+Z: redo last undone stroke / mask paint edit / polygon edit."""

        iv = self._dual_viewer.input_viewer

        if iv.annotation_mode in ("mask_paint_fg", "mask_paint_bg") and self._redo_mask_paint():
            return
        if iv.annotation_mode in ("poly_garbage", "poly_core") and self._redo_polygon_edit():
            return

        use_holdout = (
            iv.annotation_mode == "holdout"
            or (
                self._is_chroma_key_active()
                and iv.holdout_model.can_redo(0)
                and not (iv.current_stem_index >= 0 and iv.annotation_model.can_redo(iv.current_stem_index))
            )
        )

        if use_holdout:

            if iv.holdout_model.redo(0):

                iv._split_view.update()

                self._auto_save_holdout()

                self._schedule_chroma_key_preview()
                return

        elif iv.current_stem_index >= 0:

            if iv.annotation_model.redo(iv.current_stem_index):

                iv._split_view.update()

                self._auto_save_annotations()
                self._update_annotation_info()
                self._refresh_live_sam3_preview(iv.current_stem_index, force_recompute=True)
                return

        if self._redo_mask_paint():
            return
        self._redo_polygon_edit()







    def _on_track_masks(self) -> None:



        """Preview SAM 3.1 on the annotated frame, then confirm full tracking."""



        from ui.workers.gpu_job_worker import create_job_snapshot



        from backend import JobType







        clip = self._current_clip



        if clip is None:



            return



        self._auto_save_annotations()



        iv = self._dual_viewer.input_viewer



        model = iv.annotation_model



        if not model.has_annotations():



            if not iv.annotation_mode:
                iv.set_annotation_mode("fg")
                ov = self._dual_viewer.output_viewer
                ov.set_annotation_mode(None)
            QMessageBox.information(



                self, _tr("Add SAM 3.1 Prompts"),



                _tr("Click on your subject in the viewport to place prompts:\n\n"



                    "• Left Click (Key 1): Green foreground point (Active)\n"



                    "• Key 2: Red background point\n"



                    "• Key R: Bounding box\n"



                    "• Key 3: Brush mask\n\n"



                    "Then click TRACK MASK (SAM 3.1) to generate mask and track."),



            )



            return







        if not self._warn_mps_slow("SAM 3.1 Track Mask"):



            return







        self._sam3_one_frame_mode = False
        job = create_job_snapshot(clip, self._param_panel.get_params(), job_type=JobType.SAM3_PREVIEW)
        job.params["_frame_index"] = max(0, self._dual_viewer.current_stem_index)
        if not self._service.job_queue.submit(job):
            return

        self._start_worker_if_needed(job.id, job_label="Track Preview")

    @Slot()
    def _on_run_sam3_one_frame(self) -> None:
        """Render and save SAM 3.1 mask on active frame as a One Frame Hint."""
        from ui.workers.gpu_job_worker import create_job_snapshot
        from backend import JobType

        clip = self._current_clip
        if clip is None:
            return

        self._auto_save_annotations()
        iv = self._dual_viewer.input_viewer
        model = iv.annotation_model
        stem_idx = iv.current_stem_index
        if not model.has_annotations(stem_idx):
            if not iv.annotation_mode:
                iv.set_annotation_mode("fg")
                ov = self._dual_viewer.output_viewer
                ov.set_annotation_mode(None)
            QMessageBox.information(
                self, _tr("Add SAM 3.1 Prompts"),
                _tr("Please place at least one foreground point (1) or bounding box (R) on frame %d first.") % (stem_idx + 1)
            )
            return

        if not self._warn_mps_slow("SAM 3.1 One Frame Hint"):
            return

        self._sam3_one_frame_mode = True
        job = create_job_snapshot(clip, self._param_panel.get_params(), job_type=JobType.SAM3_PREVIEW)
        job.params["_frame_index"] = max(0, stem_idx)
        job.params["_save_as_one_frame_hint"] = True

        if not self._service.job_queue.submit(job):
            self._sam3_one_frame_mode = False
            return

        self._start_worker_if_needed(job.id, job_label="SAM 3.1 One Frame Hint")







    def _submit_sam3_track_job(self, clip: ClipEntry) -> bool:



        """Queue the full SAM 3.1 tracking job after preview confirmation."""



        from ui.workers.gpu_job_worker import create_job_snapshot



        from backend import JobType







        alpha_dir = os.path.join(clip.root_path, "AlphaHint")



        if os.path.isdir(alpha_dir):



            reply = QMessageBox.question(



                self, _tr("Replace Existing Alpha?"),



                _tr("This clip already has an AlphaHint (from GVM or a previous run).\n\n"



                    "Tracking a new mask sequence will replace that alpha hint.\n\n"



                    "Remove existing AlphaHint and proceed?"),



            )



            if reply != QMessageBox.Yes:



                return False



            shutil.rmtree(alpha_dir, ignore_errors=True)



            clip.alpha_asset = None



            clip.find_assets()



            self._refresh_button_state()



            logger.info("Removed existing AlphaHint/ before SAM 3.1 tracking")







        job = create_job_snapshot(clip, self._param_panel.get_params(), job_type=JobType.SAM3_TRACK)



        if not self._service.job_queue.submit(job):



            return False







        clip.set_processing(True)



        self._start_worker_if_needed(job.id, job_label="Track Mask")



        return True







    @staticmethod
    def _sam3_preview_qimage(frame_rgb: np.ndarray, mask: np.ndarray) -> QImage:
        """Render a soft-alpha rubylith translucent cutout overlay for live preview."""
        alpha = np.clip(mask.astype(np.float32) * (1.0 / 255.0), 0.0, 1.0)[..., None]
        fg = frame_rgb.astype(np.float32)

        bg_tint = fg * 0.42 + np.array([75.0, 12.0, 24.0], dtype=np.float32) * 0.58
        comp = fg * alpha + bg_tint * (1.0 - alpha)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bin_mask = (mask > 127).astype(np.uint8)
        thin_edge = (cv2.dilate(bin_mask, k) ^ cv2.erode(bin_mask, k)) > 0
        hair_soft = (mask > 20) & (mask < 235)
        edge_only = thin_edge & (~hair_soft)
        if np.any(edge_only):
            comp[edge_only] = comp[edge_only] * 0.4 + np.array([255.0, 225.0, 40.0], dtype=np.float32) * 0.6

        rgb = np.clip(np.round(comp), 0, 255).astype(np.uint8).copy()
        h, w = rgb.shape[:2]
        return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()







    def _confirm_clear_annotations(self) -> None:



        """Ctrl+C: clear strokes. Routes to holdout or SAM 3.1 by chroma key mode."""



        if self._is_chroma_key_active():



            self._confirm_clear_holdout()



        else:



            self._confirm_clear_sam3_annotations()







    def _confirm_clear_holdout(self) -> None:



        """Clear all holdout mask strokes for this clip."""



        iv = self._dual_viewer.input_viewer



        model = iv.holdout_model



        if not model.has_annotations():



            return



        reply = QMessageBox.question(



            self, _tr("Clear Holdout Strokes"),



            _tr("Clear all holdout mask strokes for this clip?"),



            QMessageBox.Yes | QMessageBox.No,



        )



        if reply == QMessageBox.Yes:



            model.clear()



            self._auto_save_holdout()



            iv._split_view.update()



            self._schedule_chroma_key_preview()







    def _confirm_clear_sam3_annotations(self) -> None:



        """Ctrl+C: choose to clear SAM 3.1 annotations on this frame, entire clip, or cancel."""



        iv = self._dual_viewer.input_viewer



        model = iv.annotation_model



        if not model or not model.has_annotations():



            return







        stem_idx = iv._split_view._annotation_stem_idx



        has_frame = model.has_annotations(stem_idx)







        box = QMessageBox(self)



        box.setWindowTitle(_tr("Clear Paint Strokes"))



        box.setText(_tr("What would you like to clear?"))



        frame_btn = box.addButton(_tr("This Frame"), QMessageBox.AcceptRole)



        clip_btn = box.addButton(_tr("Entire Clip"), QMessageBox.DestructiveRole)



        box.addButton(QMessageBox.Cancel)







        if not has_frame:



            frame_btn.setEnabled(False)







        box.exec()



        clicked = box.clickedButton()







        if clicked == frame_btn:



            model.clear(stem_idx)



            iv._split_view.update()



            self._update_annotation_info()
            self._auto_save_annotations()
            if self._current_clip is not None:
                self._invalidate_sam3_preview_cache_for_frame(self._current_clip.root_path, stem_idx)
            self._sam3_preview_seq += 1
            self._refresh_live_sam3_preview(stem_idx)



        elif clicked == clip_btn:



            self._on_clear_annotations()







    def _clip_has_any_mask_or_hint(self, clip: ClipEntry | None) -> bool:
        """True if the clip has any generated/imported mask, alpha hint, or annotations."""
        if clip is None:
            return False
        if getattr(clip, "mask_asset", None) is not None or getattr(clip, "alpha_asset", None) is not None:
            return True
        root = getattr(clip, "root_path", None)
        if not root or not os.path.isdir(root):
            return False
        for d in ("SamMaskHint", "VideoMamaMaskHint", "AlphaHint"):
            dp = os.path.join(root, d)
            if os.path.isdir(dp) and len(os.listdir(dp)) > 0:
                return True
        for f in ("annotations.json", "holdout_strokes.json", ".corridorkey_mask_manifest.json", "one_frame_hint.json"):
            if os.path.isfile(os.path.join(root, f)):
                return True
        import glob
        for pat in ("SamMaskHint.*", "VideoMamaMaskHint.*", "AlphaHint.*"):
            candidates = glob.glob(os.path.join(root, pat))
            if any(os.path.isfile(c) for c in candidates):
                return True
        if hasattr(self, "_dual_viewer") and hasattr(self._dual_viewer, "input_viewer"):
            iv = self._dual_viewer.input_viewer
            if hasattr(iv, "annotation_model") and iv.annotation_model.has_annotations():
                return True
            if hasattr(iv, "holdout_model") and iv.holdout_model.has_annotations():
                return True
        return False

    @Slot()
    def _on_clear_mask_and_hints(self) -> None:
        """User clicked 'CLEAR MASK / HINT' or chose 'Clear Mask & Hints' from menu."""
        clip = self._current_clip
        if clip is None:
            return

        box = QMessageBox(self)
        box.setWindowTitle(_tr("Clear Mask & Hints"))
        box.setText(_tr("Clear all masks, alpha hints, and brush annotations for '%s'?") % clip.name)
        box.setInformativeText(
            _tr("This will completely remove:\n"
                "• All tracked SAM 3.1 & BiRefNet masks\n"
                "• All chroma key and imported alpha hints\n"
                "• All paint strokes and prompts\n\n"
                "The clip will be reset to RAW state so you can mask from scratch or run standalone.")
        )
        box.setIcon(QMessageBox.Question)
        clear_btn = box.addButton(_tr("Clear Mask & Hints"), QMessageBox.AcceptRole)
        cancel_btn = box.addButton(_tr("Cancel"), QMessageBox.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec()

        if box.clickedButton() != clear_btn:
            return

        self._clear_all_masks_and_hints_for_clip(clip)

    def _clear_all_masks_and_hints_for_clip(self, clip: ClipEntry) -> None:
        """Completely wipe all mask and alpha hint assets from disk and memory."""
        import glob
        from ui.preview.frame_index import ViewMode

        # 1. Delete mask hint dirs (VideoMamaMaskHint and SamMaskHint)
        for mdir_name in ("VideoMamaMaskHint", "SamMaskHint"):
            mdir = os.path.join(clip.root_path, mdir_name)
            if os.path.isdir(mdir):
                shutil.rmtree(mdir, ignore_errors=True)
                logger.info(f"Removed mask hints dir: {mdir}")
            for f in glob.glob(os.path.join(clip.root_path, f"{mdir_name}.*")):
                try:
                    os.remove(f)
                except OSError:
                    pass

        # 2. Delete alpha hint dirs (AlphaHint and BiRefNetHint)
        for adir_name in ("AlphaHint", "BiRefNetHint"):
            adir = os.path.join(clip.root_path, adir_name)
            if os.path.isdir(adir):
                shutil.rmtree(adir, ignore_errors=True)
                logger.info(f"Removed alpha hints dir: {adir}")
            for f in glob.glob(os.path.join(clip.root_path, f"{adir_name}.*")):
                try:
                    os.remove(f)
                except OSError:
                    pass

        # 3. Delete annotations, holdout, and manifest json files
        for fname in ("annotations.json", "holdout_strokes.json", ".corridorkey_mask_manifest.json", "one_frame_hint.json"):
            fpath = os.path.join(clip.root_path, fname)
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                    logger.info(f"Removed {fname}")
                except OSError:
                    pass

        # 4. Clear in-memory models on both viewports
        self._clear_sam3_preview_cache()
        self._sam3_preview_seq += 1
        if hasattr(self, "_dual_viewer"):
            iv = self._dual_viewer.input_viewer
            ov = self._dual_viewer.output_viewer
            iv.clear_annotations()
            ov.clear_annotations()
            iv.holdout_model.clear()
            ov.holdout_model.clear()
            self._dual_viewer.clear_live_preview()
            self._dual_viewer.set_live_preview_modes(False)

        # 5. Clear assets & re-scan clip
        clip.mask_asset = None
        clip.alpha_asset = None
        try:
            clip.find_assets()
        except Exception:
            pass
        self._clip_model.update_clip_state(clip.name, clip.state)

        # 6. Fallback view mode if viewing MASK or ALPHA
        if hasattr(self, "_dual_viewer"):
            if self._dual_viewer.current_output_mode in (ViewMode.MASK, ViewMode.ALPHA):
                self._dual_viewer.output_viewer.set_view_mode(ViewMode.INPUT.value)
            self._sync_selected_clip_view(clip)
            self._dual_viewer.update()

        # 7. Update UI controls
        self._update_annotation_info()
        if hasattr(self, "_param_panel"):
            self._param_panel.set_clear_mask_enabled(False)
            self._param_panel.set_matanyone2_enabled(False)
            self._param_panel.set_videomama_enabled(False)
        if hasattr(self, "_update_run_button_state"):
            self._update_run_button_state(clip)

        if hasattr(self, "_show_toast"):
            self._show_toast(_tr("Mask and alpha hints cleared successfully."))
        if hasattr(self, "_status_bar"):
            self._status_bar.set_message(_tr("Mask and alpha hints cleared for '%s'.") % clip.name)

    def _on_clear_annotations(self) -> None:
        """Route to _on_clear_mask_and_hints."""
        self._on_clear_mask_and_hints()


    def _update_annotation_info(self) -> None:



        """Update parameter panel with current annotation count and scrubber markers."""



        iv = self._dual_viewer.input_viewer



        model = iv.annotation_model



        fi = iv._frame_index



        total = fi.frame_count if fi else 0



        self._param_panel.set_annotation_info(



            model.annotated_frame_count(), total



        )



        if hasattr(self, "_param_panel") and hasattr(self._param_panel, "set_clear_mask_enabled"):
            self._param_panel.set_clear_mask_enabled(self._clip_has_any_mask_or_hint(self._current_clip))

        # Push annotation coverage to the scrubber timeline



        if fi and total > 0:



            annotated = [model.has_annotations(i) for i in range(total)]



            self._dual_viewer._scrubber.set_annotation_markers(annotated)



        else:



            self._dual_viewer._scrubber.set_annotation_markers([])





    _sam3_preview_timer: QTimer | None = None
    _sam3_preview_worker: Any = None
    _sam3_preview_pending: bool = False
    _sam3_preview_seq: int = 0
    _last_sam3_preview: tuple[int, np.ndarray, np.ndarray] | None = None

    def _ensure_sam3_preview_timer(self) -> QTimer:
        if getattr(self, "_sam3_preview_timer", None) is None:
            self._sam3_preview_timer = QTimer(self)
            self._sam3_preview_timer.setSingleShot(True)
            self._sam3_preview_timer.setInterval(120)  # 120ms debounce
            self._sam3_preview_timer.timeout.connect(self._do_sam3_preview)
        return self._sam3_preview_timer

    def _invalidate_sam3_preview_cache_for_frame(self, clip_root: str, stem_idx: int) -> None:
        cache = getattr(self, "_preview_mask_cache", None)
        if isinstance(cache, (OrderedDict, dict)):
            cache.pop((clip_root, stem_idx), None)
        if self._last_sam3_preview is not None and self._last_sam3_preview[0] == stem_idx:
            self._last_sam3_preview = None

    def _refresh_live_sam3_preview(
        self,
        stem_idx: int | None = None,
        *,
        force_recompute: bool = False,
    ) -> None:
        """Synchronize live SAM 3.1 mode availability and output preview cleanly."""
        if not hasattr(self, "_dual_viewer"):
            return
        iv = self._dual_viewer.input_viewer
        if stem_idx is None:
            stem_idx = iv.current_stem_index
        has_any = iv.annotation_model.has_annotations()
        self._dual_viewer.set_live_preview_modes(has_any)

        clip = getattr(self, "_current_clip", None)
        if clip is not None and stem_idx >= 0 and iv.annotation_model.has_annotations(stem_idx):
            if force_recompute:
                self._invalidate_sam3_preview_cache_for_frame(clip.root_path, stem_idx)
                self._schedule_sam3_preview()
                return
            cached = self._get_cached_sam3_preview(clip.root_path, stem_idx)
            if cached is not None:
                frame_rgb, mask = cached
                self._last_sam3_preview = (stem_idx, frame_rgb, mask)
                qimg = self._render_sam3_preview_for_mode(frame_rgb, mask)
                self._dual_viewer.show_reprocess_preview(qimg)
            else:
                self._schedule_sam3_preview()
        else:
            self._last_sam3_preview = None
            ov = self._dual_viewer.output_viewer
            if getattr(ov, "_live_preview_stem_idx", None) is not None or getattr(ov, "_live_extra_modes", None):
                self._dual_viewer.clear_live_preview()
            if stem_idx >= 0 and ov.current_stem_index != stem_idx:
                ov.navigate_to_frame(stem_idx)

    def _schedule_sam3_preview(self) -> None:
        """Schedule a debounced live SAM 3.1 preview on the current frame."""
        clip = self._current_clip
        if clip is None:
            return
        iv = self._dual_viewer.input_viewer
        stem_idx = iv.current_stem_index
        self._dual_viewer.set_live_preview_modes(iv.annotation_model.has_annotations())
        if not iv.annotation_model.has_annotations(stem_idx):
            self._last_sam3_preview = None
            self._dual_viewer.clear_live_preview()
            self._dual_viewer.output_viewer.navigate_to_frame(stem_idx)
            return
        self._ensure_sam3_preview_timer().start()

    def _do_sam3_preview(self) -> None:
        """Run SAM 3.1 interactive segmentation on the current frame and show live result in the side window."""
        clip = self._current_clip
        if clip is None or clip.input_asset is None:
            return

        iv = self._dual_viewer.input_viewer
        fi = iv._frame_index
        if fi is None or fi.frame_count == 0:
            return
        stem_idx = iv.current_stem_index
        if stem_idx < 0 or stem_idx >= fi.frame_count:
            return

        self._dual_viewer.set_live_preview_modes(iv.annotation_model.has_annotations())
        if not iv.annotation_model.has_annotations(stem_idx):
            self._last_sam3_preview = None
            self._dual_viewer.clear_live_preview()
            self._dual_viewer.output_viewer.navigate_to_frame(stem_idx)
            return

        worker = getattr(self, "_sam3_preview_worker", None)
        if worker is not None and worker.isRunning():
            self._sam3_preview_pending = True
            return

        from ui.preview.frame_index import ViewMode
        input_path = fi.get_path(ViewMode.INPUT, stem_idx)
        if input_path is None or not os.path.isfile(input_path):
            return

        from backend.annotation_prompts import load_annotation_prompt_frames
        from sam3_tracker import PromptFrame
        from ui.workers import SAM3LivePreviewWorker

        prompt_frames = load_annotation_prompt_frames(
            clip.root_path,
            allowed_indices=[stem_idx],
        )
        if not prompt_frames:
            return

        pf = prompt_frames[0]
        prompt = PromptFrame(
            frame_index=0,
            positive_points=[(int(round(x)), int(round(y))) for x, y in pf.positive_points],
            negative_points=[(int(round(x)), int(round(y))) for x, y in pf.negative_points],
            box=(
                None
                if pf.box is None
                else (
                    int(round(pf.box[0])),
                    int(round(pf.box[1])),
                    int(round(pf.box[2])),
                    int(round(pf.box[3])),
                )
            ),
            hair_points=[(int(round(x)), int(round(y))) for x, y in getattr(pf, "hair_points", [])],
            hair_strokes=[
                ([(float(x), float(y)) for x, y in s_pts], float(r))
                for s_pts, r in getattr(pf, "hair_strokes", [])
            ],
            negative_strokes=[
                ([(int(round(x)), int(round(y))) for x, y in s_pts], float(r))
                for s_pts, r in getattr(pf, "negative_strokes", [])
            ],
        )

        sequence_is_linear = (
            self._input_is_linear_for_clip(clip)
            if hasattr(self, "_input_is_linear_for_clip")
            else getattr(self, "_input_exr_is_linear", False)
        )

        self._sam3_preview_seq += 1
        req_seq = self._sam3_preview_seq
        self._sam3_preview_worker = SAM3LivePreviewWorker(
            input_path=input_path,
            prompt=prompt,
            stem_idx=stem_idx,
            service=self._service,
            is_linear=sequence_is_linear,
            clip_root=clip.root_path,
            parent=self,
        )
        self._sam3_preview_worker._sam3_seq = req_seq
        self._sam3_preview_worker.preview_ready.connect(self._on_sam3_preview_ready)
        self._sam3_preview_worker.finished.connect(self._on_sam3_preview_worker_finished)
        self._sam3_preview_worker.start()

    _SAM3_PREVIEW_CACHE_MAX = 16

    def _get_cached_sam3_preview(self, clip_root: str, stem_idx: int):
        cache = getattr(self, "_preview_mask_cache", None)
        if not isinstance(cache, OrderedDict):
            return None
        key = (clip_root, stem_idx)
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]

    def _cache_sam3_preview(self, clip_root: str, stem_idx: int, frame_rgb: np.ndarray, mask: np.ndarray) -> None:
        cache = getattr(self, "_preview_mask_cache", None)
        if not isinstance(cache, OrderedDict):
            cache = OrderedDict()
            self._preview_mask_cache = cache
        key = (clip_root, stem_idx)
        cache[key] = (frame_rgb, mask)
        cache.move_to_end(key)
        while len(cache) > self._SAM3_PREVIEW_CACHE_MAX:
            cache.popitem(last=False)

    def _clear_sam3_preview_cache(self) -> None:
        cache = getattr(self, "_preview_mask_cache", None)
        if isinstance(cache, (OrderedDict, dict)):
            cache.clear()
        self._last_sam3_preview = None

    @Slot(int, object, object)
    def _on_sam3_preview_ready(self, stem_idx: int, frame_rgb: np.ndarray, mask: np.ndarray) -> None:
        worker = self.sender()
        worker_root = getattr(worker, "clip_root", "") if worker is not None else ""
        worker_seq = getattr(worker, "_sam3_seq", None) if worker is not None else None
        if worker_seq is not None and worker_seq != self._sam3_preview_seq:
            return
        clip = self._current_clip
        if clip is None:
            return
        if worker_root and clip.root_path != worker_root:
            return
        iv = self._dual_viewer.input_viewer
        if not iv.annotation_model.has_annotations(stem_idx):
            return
        cache_root = worker_root or clip.root_path
        self._cache_sam3_preview(cache_root, stem_idx, frame_rgb, mask)
        if iv.current_stem_index != stem_idx:
            return
        self._dual_viewer.set_live_preview_modes(iv.annotation_model.has_annotations())
        self._last_sam3_preview = (stem_idx, frame_rgb, mask)
        qimg = self._render_sam3_preview_for_mode(frame_rgb, mask)
        self._dual_viewer.show_reprocess_preview(qimg)

    def _on_sam3_preview_worker_finished(self) -> None:
        old_worker = self._sam3_preview_worker
        self._sam3_preview_worker = None
        if old_worker is not None:
            old_worker.deleteLater()
        if getattr(self, "_sam3_preview_pending", False):
            self._sam3_preview_pending = False
            self._do_sam3_preview()

    def _render_sam3_preview_for_mode(self, frame_rgb: np.ndarray, mask: np.ndarray) -> QImage:
        from ui.preview.frame_index import ViewMode
        from ui.preview.display_transform import composite_over_backdrop

        mode = self._dual_viewer.current_output_mode
        if mode in (ViewMode.MATTE, ViewMode.MASK, ViewMode.ALPHA):
            h, w = mask.shape[:2]
            rgb = np.stack([mask, mask, mask], axis=2).copy()
            return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        if mode == ViewMode.COMP:
            comp = composite_over_backdrop(
                frame_rgb.astype(np.float32) * (1.0 / 255.0),
                mask.astype(np.float32) * (1.0 / 255.0),
                backdrop=self._dual_viewer.current_backdrop,
            ).copy()
            h, w = comp.shape[:2]
            return QImage(comp.data, w, h, w * 3, QImage.Format_RGB888).copy()
        if mode == ViewMode.FG:
            fg_blk = composite_over_backdrop(
                frame_rgb.astype(np.float32) * (1.0 / 255.0),
                mask.astype(np.float32) * (1.0 / 255.0),
                backdrop="black",
            ).copy()
            h, w = fg_blk.shape[:2]
            return QImage(fg_blk.data, w, h, w * 3, QImage.Format_RGB888).copy()
        return self._sam3_preview_qimage(frame_rgb, mask)

    # ── Multi-Subject ID, Polygon Garbage/Core Mattes & Post-Track Mask Paint ──

    def _set_active_object_id(self, object_id: int) -> None:
        oid = max(1, min(4, int(object_id)))
        self._active_object_id = oid
        if hasattr(self, "_dual_viewer"):
            self._dual_viewer.set_active_object_id(oid)

    def _toggle_annotation_mode_simple(self, mode: str) -> None:
        iv = self._dual_viewer.input_viewer
        ov = self._dual_viewer.output_viewer
        if iv.annotation_mode == mode:
            iv.set_annotation_mode(None)
            ov.set_annotation_mode(None)
        else:
            self._deactivate_eyedropper_if_active()
            iv.set_annotation_mode(mode)
            ov.set_annotation_mode(None)
            self._dual_viewer.set_holdout_active(False)

    def _toggle_poly_garbage(self) -> None:
        """Toggle Polygon Garbage Matte authoring mode."""
        self._toggle_annotation_mode_simple("poly_garbage")

    def _toggle_poly_core(self) -> None:
        """Toggle Polygon Core Matte authoring mode."""
        self._toggle_annotation_mode_simple("poly_core")

    def _toggle_mask_paint_fg(self) -> None:
        """Toggle direct post-track Mask Paint (+) mode."""
        self._toggle_annotation_mode_simple("mask_paint_fg")

    def _toggle_mask_paint_bg(self) -> None:
        """Toggle direct post-track Mask Erase (-) mode."""
        self._toggle_annotation_mode_simple("mask_paint_bg")

    def _load_clip_polygon_model(self, clip: ClipEntry | None):
        from backend.matte_shapes import PolygonMatteModel
        if clip is None or not clip.root_path:
            return PolygonMatteModel()
        return PolygonMatteModel.load_from_clip(clip.root_path)

    def _sync_polygon_model_to_viewer(self) -> None:
        model = self._load_clip_polygon_model(getattr(self, "_current_clip", None))
        if hasattr(self, "_dual_viewer"):
            self._dual_viewer.set_polygon_model(model)

    @Slot(str, object)
    def _on_polygon_completed(self, kind: str, norm_points: object) -> None:
        clip = getattr(self, "_current_clip", None)
        if clip is None or not norm_points or len(norm_points) < 3:
            return
        from backend.matte_shapes import PolygonMatteModel, PolygonMatteShape
        frame_idx = int(self._dual_viewer.input_viewer.current_stem_index)
        if frame_idx < 0:
            frame_idx = 0
        inverted = False
        feather_px = 4
        curved = False
        if hasattr(self, "_param_panel") and self._param_panel is not None:
            inverted = bool(getattr(self._param_panel, "poly_invert_checked", lambda: False)())
            feather_px = int(getattr(self._param_panel, "poly_feather_value", lambda: 4)())
            curved = bool(getattr(self._param_panel, "poly_curve_checked", lambda: False)())

        model = PolygonMatteModel.load_from_clip(clip.root_path)
        self._push_poly_undo(model)
        shape = PolygonMatteShape(
            kind="core" if kind == "core" else "garbage",
            inverted=inverted,
            feather_px=feather_px,
            curved=curved,
        )
        shape.set_keyframe(frame_idx, list(norm_points))
        model.add_shape(shape)
        model.save_to_clip(clip.root_path)
        self._dual_viewer.set_polygon_model(model)
        if hasattr(self, "_status_bar"):
            self._status_bar.set_message(
                f"Saved {shape.kind} polygon matte keyframe @ frame {frame_idx + 1} ({len(model.shapes)} shape(s))"
            )
        if hasattr(self, "_reprocess_timer") and self._param_panel.live_preview_enabled:
            self._reprocess_timer.start()

    def _get_poly_undo_stack(self):
        from backend.matte_shapes import PolygonUndoStack
        st = getattr(self, "_poly_undo_stack", None)
        if st is None:
            st = PolygonUndoStack(max_depth=40)
            self._poly_undo_stack = st
        return st

    def _push_poly_undo(self, model) -> None:
        self._get_poly_undo_stack().push(model)

    def _undo_polygon_edit(self) -> bool:
        clip = getattr(self, "_current_clip", None)
        if clip is None:
            return False
        from backend.matte_shapes import PolygonMatteModel
        model = PolygonMatteModel.load_from_clip(clip.root_path)
        if not self._get_poly_undo_stack().undo(model):
            return False
        model.save_to_clip(clip.root_path)
        self._dual_viewer.set_polygon_model(model)
        if hasattr(self, "_reprocess_timer") and self._param_panel.live_preview_enabled:
            self._reprocess_timer.start()
        return True

    def _redo_polygon_edit(self) -> bool:
        clip = getattr(self, "_current_clip", None)
        if clip is None:
            return False
        from backend.matte_shapes import PolygonMatteModel
        model = PolygonMatteModel.load_from_clip(clip.root_path)
        if not self._get_poly_undo_stack().redo(model):
            return False
        model.save_to_clip(clip.root_path)
        self._dual_viewer.set_polygon_model(model)
        if hasattr(self, "_reprocess_timer") and self._param_panel.live_preview_enabled:
            self._reprocess_timer.start()
        return True

    @Slot(str, int, int, float, float)
    def _on_polygon_vertex_moved(self, shape_id: str, frame_idx: int, vertex_idx: int, nx: float, ny: float) -> None:
        clip = getattr(self, "_current_clip", None)
        if clip is None:
            return
        from backend.matte_shapes import PolygonMatteModel
        model = PolygonMatteModel.load_from_clip(clip.root_path)
        self._push_poly_undo(model)
        changed = False
        for s in model.shapes:
            if s.shape_id == shape_id:
                changed = s.move_vertex(frame_idx, vertex_idx, nx, ny)
                break
        if not changed:
            return
        model.save_to_clip(clip.root_path)
        self._dual_viewer.set_polygon_model(model)
        if hasattr(self, "_reprocess_timer") and self._param_panel.live_preview_enabled:
            self._reprocess_timer.start()

    @Slot(bool)
    def _on_poly_curve_changed(self, curved: bool) -> None:
        clip = getattr(self, "_current_clip", None)
        if clip is None:
            return
        from backend.matte_shapes import PolygonMatteModel
        model = PolygonMatteModel.load_from_clip(clip.root_path)
        if not model.shapes:
            return
        self._push_poly_undo(model)
        model.shapes[-1].curved = bool(curved)
        model.save_to_clip(clip.root_path)
        self._dual_viewer.set_polygon_model(model)
        if hasattr(self, "_reprocess_timer") and self._param_panel.live_preview_enabled:
            self._reprocess_timer.start()

    @Slot(bool)
    def _on_poly_invert_changed(self, inverted: bool) -> None:
        clip = getattr(self, "_current_clip", None)
        if clip is None:
            return
        from backend.matte_shapes import PolygonMatteModel
        model = PolygonMatteModel.load_from_clip(clip.root_path)
        if not model.shapes:
            return
        self._push_poly_undo(model)
        model.shapes[-1].inverted = bool(inverted)
        model.save_to_clip(clip.root_path)
        self._dual_viewer.set_polygon_model(model)
        if hasattr(self, "_reprocess_timer") and self._param_panel.live_preview_enabled:
            self._reprocess_timer.start()

    @Slot(int)
    def _on_poly_feather_changed(self, feather_px: int) -> None:
        clip = getattr(self, "_current_clip", None)
        if clip is None:
            return
        from backend.matte_shapes import PolygonMatteModel
        model = PolygonMatteModel.load_from_clip(clip.root_path)
        if not model.shapes:
            return
        self._push_poly_undo(model)
        model.shapes[-1].feather_px = max(0, int(feather_px))
        model.save_to_clip(clip.root_path)
        self._dual_viewer.set_polygon_model(model)
        if hasattr(self, "_reprocess_timer") and self._param_panel.live_preview_enabled:
            self._reprocess_timer.start()

    def _clear_polygon_mattes(self) -> None:
        clip = getattr(self, "_current_clip", None)
        if clip is None:
            return
        from backend.matte_shapes import PolygonMatteModel
        model = PolygonMatteModel.load_from_clip(clip.root_path)
        if model.shapes:
            self._push_poly_undo(model)
        model = PolygonMatteModel()
        model.save_to_clip(clip.root_path)
        self._dual_viewer.set_polygon_model(model)
        if hasattr(self, "_status_bar"):
            self._status_bar.set_message("Cleared polygon mattes")
        if hasattr(self, "_reprocess_timer") and self._param_panel.live_preview_enabled:
            self._reprocess_timer.start()

    @staticmethod
    def _get_clip_frame_stems_and_size(clip: ClipEntry) -> tuple[list[str], int, int]:
        from backend.frame_io import imread_unicode
        if clip.input_asset is not None and clip.input_asset.asset_type == "sequence":
            files = clip.input_asset.get_frame_files()
            stems = [os.path.splitext(fn)[0] for fn in files]
            w, h = 1920, 1080
            if files:
                img = imread_unicode(os.path.join(clip.input_asset.path, files[0]))
                if img is not None:
                    h, w = img.shape[:2]
            return stems, w, h
        count = max(1, clip.input_asset.frame_count if clip.input_asset else 1)
        return [f"frame_{i:06d}" for i in range(count)], 1920, 1080

    @Slot(int, str, object, float)
    def _on_mask_paint_stroke(self, frame_idx: int, action: str, points: object, radius: float) -> None:
        clip = getattr(self, "_current_clip", None)
        if clip is None or not points:
            return
        import cv2
        from backend.frame_io import imread_unicode, imwrite_unicode
        from backend.clip_state import ClipAsset

        stems, w, h = self._get_clip_frame_stems_and_size(clip)
        if frame_idx < 0 or frame_idx >= len(stems):
            return
        stem = stems[frame_idx]

        # Ensure SamMaskHint exists cleanly (never VideoMamaMaskHint)
        sam_dir = os.path.join(clip.root_path, "SamMaskHint")
        os.makedirs(sam_dir, exist_ok=True)
        sam_path = os.path.join(sam_dir, f"{stem}.png")

        target_paths = [sam_path]
        alpha_hint_dir = os.path.join(clip.root_path, "AlphaHint")
        alpha_hint_path = os.path.join(alpha_hint_dir, f"{stem}.png")
        if os.path.isdir(alpha_hint_dir):
            target_paths.append(alpha_hint_path)

        base_mask = None
        for cand in (sam_path, alpha_hint_path):
            if os.path.isfile(cand):
                base_mask = imread_unicode(cand, cv2.IMREAD_GRAYSCALE)
                if base_mask is not None:
                    break
        if base_mask is None:
            base_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            h, w = base_mask.shape[:2]

        paint_val = 255 if action == "add" else 0
        r = max(1, int(round(radius)))
        pts_list = [(float(px), float(py)) for px, py in points]
        for px, py in pts_list:
            cv2.circle(base_mask, (int(round(px)), int(round(py))), r, paint_val, -1)
        for i in range(len(pts_list) - 1):
            p1 = (int(round(pts_list[i][0])), int(round(pts_list[i][1])))
            p2 = (int(round(pts_list[i + 1][0])), int(round(pts_list[i + 1][1])))
            cv2.line(base_mask, p1, p2, paint_val, r * 2)

        # Snapshot previous mask bytes for multi-step Undo/Redo (capped at 30 steps)
        snap_before: dict[str, bytes | None] = {}
        for p in target_paths:
            if os.path.isfile(p):
                with open(p, "rb") as rf:
                    snap_before[p] = rf.read()
            else:
                snap_before[p] = None
        undo_q = getattr(self, "_mask_paint_undo", None)
        if undo_q is None:
            undo_q = []
            self._mask_paint_undo = undo_q
        redo_q = getattr(self, "_mask_paint_redo", None)
        if redo_q is None:
            redo_q = []
            self._mask_paint_redo = redo_q
        undo_q.append((clip.root_path, int(frame_idx), snap_before))
        if len(undo_q) > 30:
            undo_q.pop(0)
        redo_q.clear()

        for out_p in target_paths:
            imwrite_unicode(out_p, base_mask)

        clip.find_assets()
        if hasattr(self, "_clip_model"):
            self._clip_model.update_clip_state(clip.name, clip.state)
        if hasattr(self, "_io_tray"):
            self._io_tray.refresh()
        if hasattr(self, "_refresh_button_state"):
            self._refresh_button_state()

        self._invalidate_sam3_preview_cache_for_frame(clip.root_path, frame_idx)
        from ui.preview.display_transform import clear_cache
        clear_cache()
        if hasattr(self, "_dual_viewer"):
            self._dual_viewer.input_viewer.refresh_current_frame()
            self._dual_viewer.output_viewer.refresh_current_frame()
        if hasattr(self, "_reprocess_timer") and self._param_panel.live_preview_enabled:
            self._reprocess_timer.start()

    def _restore_mask_paint_snapshot(self, entry: tuple[str, int, dict]) -> tuple[str, int, dict]:
        """Swap on-disk mask files with `entry` and return the reverse snapshot for undo/redo."""
        clip_root, f_idx, snap = entry
        reverse: dict[str, bytes | None] = {}
        for p, prev_bytes in snap.items():
            if os.path.isfile(p):
                with open(p, "rb") as rf:
                    reverse[p] = rf.read()
            else:
                reverse[p] = None
            if prev_bytes is None:
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            else:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as wf:
                    wf.write(prev_bytes)
        self._invalidate_sam3_preview_cache_for_frame(clip_root, f_idx)
        clip = next((c for c in self._clip_model.clips if c.root_path == clip_root), None) if hasattr(self, "_clip_model") else None
        if clip is not None:
            clip.find_assets()
            self._clip_model.update_clip_state(clip.name, clip.state)
            if hasattr(self, "_io_tray"):
                self._io_tray.refresh()
            if hasattr(self, "_refresh_button_state"):
                self._refresh_button_state()
        from ui.preview.display_transform import clear_cache
        clear_cache()
        if hasattr(self, "_dual_viewer"):
            self._dual_viewer.input_viewer.refresh_current_frame()
            self._dual_viewer.output_viewer.refresh_current_frame()
        if hasattr(self, "_reprocess_timer") and self._param_panel.live_preview_enabled:
            self._reprocess_timer.start()
        return (clip_root, f_idx, reverse)

    def _undo_mask_paint(self) -> bool:
        undo_q = getattr(self, "_mask_paint_undo", None)
        if not undo_q:
            return False
        redo_q = getattr(self, "_mask_paint_redo", None)
        if redo_q is None:
            redo_q = []
            self._mask_paint_redo = redo_q
        entry = undo_q.pop()
        redo_q.append(self._restore_mask_paint_snapshot(entry))
        return True

    def _redo_mask_paint(self) -> bool:
        redo_q = getattr(self, "_mask_paint_redo", None)
        if not redo_q:
            return False
        undo_q = getattr(self, "_mask_paint_undo", None)
        if undo_q is None:
            undo_q = []
            self._mask_paint_undo = undo_q
        entry = redo_q.pop()
        undo_q.append(self._restore_mask_paint_snapshot(entry))
        return True

    def _interpolate_mask_range(
        self,
        start_idx: int | None = None,
        end_idx: int | None = None,
    ) -> int:
        """Linearly interpolate saved masks between [In, Out] endpoints without GPU re-tracking."""
        clip = getattr(self, "_current_clip", None)
        if clip is None:
            return 0
        import cv2
        from backend.frame_io import imread_unicode, imwrite_unicode

        stems, w, h = self._get_clip_frame_stems_and_size(clip)
        if len(stems) < 3:
            return 0
        if start_idx is None or end_idx is None:
            io_in, io_out = self._dual_viewer.get_in_out() if hasattr(self, "_dual_viewer") else (-1, -1)
            if 0 <= io_in <= io_out < len(stems):
                start_idx, end_idx = io_in, io_out
            else:
                start_idx, end_idx = 0, len(stems) - 1
        lo = max(0, min(int(start_idx), int(end_idx)))
        hi = min(len(stems) - 1, max(int(start_idx), int(end_idx)))
        if hi - lo < 2:
            if hasattr(self, "_status_bar"):
                self._status_bar.set_message("Set In/Out range spanning at least 3 frames to interpolate")
            return 0

        updated = 0
        for folder in ("SamMaskHint", "AlphaHint"):
            mdir = os.path.join(clip.root_path, folder)
            if not os.path.isdir(mdir):
                continue
            p0 = os.path.join(mdir, f"{stems[lo]}.png")
            p1 = os.path.join(mdir, f"{stems[hi]}.png")
            if not (os.path.isfile(p0) and os.path.isfile(p1)):
                continue
            m0 = imread_unicode(p0, cv2.IMREAD_GRAYSCALE)
            m1 = imread_unicode(p1, cv2.IMREAD_GRAYSCALE)
            if m0 is None or m1 is None:
                continue
            if m1.shape != m0.shape:
                m1 = cv2.resize(m1, (m0.shape[1], m0.shape[0]), interpolation=cv2.INTER_LINEAR)
            f0 = m0.astype(np.float32)
            f1 = m1.astype(np.float32)
            for idx in range(lo + 1, hi):
                t = float(idx - lo) / float(hi - lo)
                blend = np.clip(np.round(f0 * (1.0 - t) + f1 * t), 0, 255).astype(np.uint8)
                imwrite_unicode(os.path.join(mdir, f"{stems[idx]}.png"), blend)
                self._invalidate_sam3_preview_cache_for_frame(clip.root_path, idx)
                updated += 1

        if updated > 0:
            from ui.preview.display_transform import clear_cache
            clear_cache()
            if hasattr(self, "_dual_viewer"):
                self._dual_viewer.input_viewer.refresh_current_frame()
                self._dual_viewer.output_viewer.refresh_current_frame()
            if hasattr(self, "_param_panel") and self._param_panel.live_preview_enabled and hasattr(self, "_reprocess_timer"):
                self._reprocess_timer.start()
            if hasattr(self, "_status_bar"):
                self._status_bar.set_message(f"Interpolated masks across frames {lo + 1}..{hi + 1}")
        return updated
