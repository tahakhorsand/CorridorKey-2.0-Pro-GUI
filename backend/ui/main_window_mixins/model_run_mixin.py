from __future__ import annotations

import logging
import os
import shutil

from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Slot

from . import _tr
from backend import ClipState, JobType
from ui.workers.gpu_job_worker import create_job_snapshot

logger = logging.getLogger(__name__)


class ModelRunMixin:
    """Model-specific run methods (BiRefNet, SAM 3.1, MatAnyone 2) for MainWindow."""

    def _confirm_partial_alpha(self) -> bool | None:
        """Check for partial alpha from a previous interrupted run."""
        alpha_dir = os.path.join(self._current_clip.root_path, "AlphaHint")
        if not os.path.isdir(alpha_dir):
            return None
        existing = [f for f in os.listdir(alpha_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not existing:
            return None
        total = self._current_clip.input_asset.frame_count if self._current_clip.input_asset else 0
        msg = QMessageBox(self)
        msg.setWindowTitle(_tr("Partial Alpha Found"))
        msg.setText(_tr("Found %d/%d alpha frames from a previous run.") % (len(existing), total))
        msg.setInformativeText(
            _tr("Resume will skip completed frames.\nRegenerate will redo all frames from scratch.")
        )
        resume_btn = msg.addButton(_tr("Resume"), QMessageBox.AcceptRole)
        regen_btn = msg.addButton(_tr("Regenerate"), QMessageBox.DestructiveRole)
        msg.addButton(QMessageBox.Cancel)
        msg.setDefaultButton(resume_btn)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == regen_btn:
            shutil.rmtree(alpha_dir, ignore_errors=True)
            return True
        if clicked == resume_btn:
            return True
        return False

    @Slot(str)
    def _on_run_birefnet(self, usage: str) -> None:
        """Run BiRefNet alpha generation on the selected clip using local HF cache."""
        if self._current_clip is None or self._current_clip.state not in (
            ClipState.RAW, ClipState.MASKED, ClipState.READY, ClipState.COMPLETE
        ):
            return

        result = self._confirm_partial_alpha()
        if result is False:
            return

        job = create_job_snapshot(
            self._current_clip,
            job_type=JobType.BIREFNET_ALPHA,
            birefnet_usage=usage,
        )
        if not self._service.job_queue.submit(job):
            return

        self._current_clip.set_processing(True)
        self._start_worker_if_needed(job.id, job_label=f"BiRefNet ({usage})")

    @Slot(str)
    def _on_run_birefnet_one_frame(self, usage: str) -> None:
        """Run BiRefNet on active frame only as a One Frame Hint."""
        if self._current_clip is None or self._current_clip.state not in (
            ClipState.RAW, ClipState.MASKED, ClipState.READY, ClipState.COMPLETE
        ):
            return

        frame_idx = max(0, self._dual_viewer.current_stem_index)
        job = create_job_snapshot(
            self._current_clip,
            job_type=JobType.BIREFNET_ALPHA,
            birefnet_usage=usage,
        )
        job.params["_one_frame_hint"] = True
        job.params["_frame_index"] = frame_idx

        if not self._service.job_queue.submit(job):
            return

        self._current_clip.set_processing(True)
        self._start_worker_if_needed(job.id, job_label=f"BiRefNet One Frame ({usage})")

    @Slot()
    def _on_run_matanyone2(self) -> None:
        """Deprecated placeholder after MatAnyone 2 removal."""
        pass
