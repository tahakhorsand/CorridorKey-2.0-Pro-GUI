from __future__ import annotations







import os



import shutil



import logging
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



        """Ctrl+Z: undo last stroke. Routes by chroma key mode."""



        iv = self._dual_viewer.input_viewer



        if not iv.annotation_mode:



            return



        if self._is_chroma_key_active():



            if iv.holdout_model.undo(0):



                iv._split_view.update()



                self._auto_save_holdout()



                self._schedule_chroma_key_preview()



        elif iv.current_stem_index >= 0:



            if iv.annotation_model.undo(iv.current_stem_index):



                iv._split_view.update()



                self._auto_save_annotations()
                self._update_annotation_info()
                self._schedule_sam3_preview()







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



        """Render a contour-only SAM 3.1 preview for the output viewer."""



        overlay = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)



                # Primary contour: solid subject silhouette and internal cutout boundaries
        contours, _ = cv2.findContours(
            (mask > 127).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        valid_contours = [c for c in contours if cv2.contourArea(c) >= 50]
        cv2.drawContours(overlay, valid_contours, -1, (0, 230, 255), 4)

        # Secondary contour: soft hair strands ONLY (never outline the body)
        soft_hair = ((mask > 15) & (mask < 240)).astype(np.uint8)
        if np.count_nonzero(soft_hair) > 100:
            hair_contours, _ = cv2.findContours(
                soft_hair,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            hair_valid = [c for c in hair_contours if 20 <= cv2.contourArea(c) <= 5000]
            if hair_valid:
                cv2.drawContours(overlay, hair_valid, -1, (255, 230, 0), 2)



        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)



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
            self._last_sam3_preview = None
            self._dual_viewer.output_viewer.navigate_to_frame(stem_idx)



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
        for d in ("VideoMamaMaskHint", "AlphaHint"):
            dp = os.path.join(root, d)
            if os.path.isdir(dp) and len(os.listdir(dp)) > 0:
                return True
        for f in ("annotations.json", "holdout_strokes.json", ".corridorkey_mask_manifest.json", "one_frame_hint.json"):
            if os.path.isfile(os.path.join(root, f)):
                return True
        import glob
        for pat in ("VideoMamaMaskHint.*", "AlphaHint.*"):
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
        if hasattr(self, "_dual_viewer"):
            iv = self._dual_viewer.input_viewer
            ov = self._dual_viewer.output_viewer
            iv.clear_annotations()
            ov.clear_annotations()
            iv.holdout_model.clear()
            ov.holdout_model.clear()

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
                self._dual_viewer.set_output_mode(ViewMode.INPUT)
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
    _last_sam3_preview: tuple[int, np.ndarray, np.ndarray] | None = None

    def _ensure_sam3_preview_timer(self) -> QTimer:
        if getattr(self, "_sam3_preview_timer", None) is None:
            self._sam3_preview_timer = QTimer(self)
            self._sam3_preview_timer.setSingleShot(True)
            self._sam3_preview_timer.setInterval(120)  # 120ms debounce
            self._sam3_preview_timer.timeout.connect(self._do_sam3_preview)
        return self._sam3_preview_timer

    def _schedule_sam3_preview(self) -> None:
        """Schedule a debounced live SAM 3.1 preview on the current frame."""
        clip = self._current_clip
        if clip is None:
            return
        iv = self._dual_viewer.input_viewer
        stem_idx = iv.current_stem_index
        if not iv.annotation_model.has_annotations(stem_idx):
            self._last_sam3_preview = None
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

        if not iv.annotation_model.has_annotations(stem_idx):
            self._last_sam3_preview = None
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
            negative_strokes=[
                ([(int(round(x)), int(round(y))) for x, y in s_pts], float(r))
                for s_pts, r in getattr(pf, "negative_strokes", [])
            ],
        )

        sequence_is_linear = getattr(self, "_input_exr_is_linear", False)

        self._sam3_preview_worker = SAM3LivePreviewWorker(
            input_path=input_path,
            prompt=prompt,
            stem_idx=stem_idx,
            service=self._service,
            is_linear=sequence_is_linear,
            parent=self,
        )
        self._sam3_preview_worker.preview_ready.connect(self._on_sam3_preview_ready)
        self._sam3_preview_worker.finished.connect(self._on_sam3_preview_worker_finished)
        self._sam3_preview_worker.start()

    @Slot(int, object, object)
    def _on_sam3_preview_ready(self, stem_idx: int, frame_rgb: np.ndarray, mask: np.ndarray) -> None:
        if not hasattr(self, "_preview_mask_cache"):
            self._preview_mask_cache = {}
        self._preview_mask_cache[stem_idx] = (frame_rgb, mask)
        iv = self._dual_viewer.input_viewer
        if iv.current_stem_index != stem_idx:
            return
        self._last_sam3_preview = (stem_idx, frame_rgb, mask)
        qimg = self._render_sam3_preview_for_mode(frame_rgb, mask)
        self._dual_viewer.show_reprocess_preview(qimg)

    def _on_sam3_preview_worker_finished(self) -> None:
        self._sam3_preview_worker = None
        if getattr(self, "_sam3_preview_pending", False):
            self._sam3_preview_pending = False
            self._do_sam3_preview()

    def _render_sam3_preview_for_mode(self, frame_rgb: np.ndarray, mask: np.ndarray) -> QImage:
        from ui.preview.frame_index import ViewMode
        mode = self._dual_viewer.current_output_mode
        if mode in (ViewMode.MATTE, ViewMode.MASK, ViewMode.ALPHA):
            h, w = mask.shape[:2]
            rgb = np.stack([mask, mask, mask], axis=2).copy()
            return QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        else:
            return self._sam3_preview_qimage(frame_rgb, mask)
