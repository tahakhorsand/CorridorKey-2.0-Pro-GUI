"""Dual viewer panel — side-by-side input/output comparison with synced scrubbing.



Left viewer shows the original input, right viewer shows the keyed output

(COMP by default, user-switchable). Both viewers share a single frame scrubber

at the bottom that keeps them in sync.



When no output exists yet (clip not processed), both viewers show the input.

"""

from __future__ import annotations



import logging

import os



from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel

from PySide6.QtCore import Qt, Signal, Slot

from PySide6.QtGui import QImage



from backend import ClipEntry, ClipState

from ui.preview.frame_index import FrameIndex, ViewMode

from ui.widgets.preview_viewport import PreviewViewport

from ui.widgets.frame_scrubber import FrameScrubber



logger = logging.getLogger(__name__)





class DualViewerPanel(QWidget):

    """Side-by-side input/output viewers with synced scrubbing.



    The left viewer is locked to INPUT mode. The right viewer defaults to

    COMP mode but can be switched via its ViewModeBar.

    A single shared scrubber drives both viewers.

    """



    frame_changed = Signal(int)  # current stem index (for external listeners)

    output_mode_changed = Signal(str)

    output_backdrop_changed = Signal(str)

    custom_plate_requested = Signal()

    pixel_hovered = Signal(int, int, int, int, int, float)

    polygon_completed = Signal(str, object)

    polygon_vertex_moved = Signal(str, int, int, float, float)

    mask_paint_stroke = Signal(int, str, object, float)



    def __init__(self, parent=None):

        super().__init__(parent)



        self._clip: ClipEntry | None = None



        self._wipe_active = False

        self._snapshot_image: QImage | None = None

        self._sam3_live_modes: set[ViewMode] = set()

        self._render_live_modes: set[ViewMode] = set()



        layout = QVBoxLayout(self)

        layout.setContentsMargins(0, 0, 0, 0)

        layout.setSpacing(0)



        # Horizontal splitter for the two viewers

        self._viewer_splitter = QSplitter(Qt.Horizontal)



        # Left viewer — locked to INPUT

        self._input_viewer = PreviewViewport(show_scrubber=False)

        self._input_viewer.lock_mode(ViewMode.INPUT)

        self._input_viewer._mode_bar.set_backdrop_visible(False)

        self._viewer_splitter.addWidget(self._input_viewer)



        # Right viewer — user-switchable modes (COMP default)

        self._output_viewer = PreviewViewport(show_scrubber=False)

        self._output_viewer.hide_clip_info()  # redundant with left panel

        self._viewer_splitter.addWidget(self._output_viewer)



        # Equal split

        self._viewer_splitter.setSizes([500, 500])

        self._viewer_splitter.setStretchFactor(0, 1)

        self._viewer_splitter.setStretchFactor(1, 1)



        layout.addWidget(self._viewer_splitter, 1)



        # A/B wipe overlay — sits ON TOP of the splitter, covers both viewer

        # canvases without touching the top bars.  Hidden until activated.

        from ui.widgets.split_view import SplitViewWidget

        self._wipe_overlay = SplitViewWidget(parent=self)

        self._wipe_overlay.set_wipe_mode(True)

        self._wipe_overlay.hide()

        from ui.widgets.orbital_spinner import OrbitalBusyOverlay
        self._busy_overlay = OrbitalBusyOverlay(parent=self)



        # A/B wipe button — lives in the input viewer's top bar, right edge

        self._ab_button = self._input_viewer.add_ab_button()

        self._ab_button.clicked.connect(self.toggle_wipe_mode)

        self._snap_button = self._input_viewer.add_snap_button()

        self._snap_button.clicked.connect(self.toggle_snapshot_compare)



        # Shared scrubber at the bottom

        bottom = QHBoxLayout()

        bottom.setContentsMargins(0, 0, 0, 0)

        bottom.setSpacing(4)



        self._scrubber = FrameScrubber()

        self._scrubber.frame_changed.connect(self._on_scrubber_frame)

        bottom.addWidget(self._scrubber, 1)



        self._zoom_label = QLabel("100%")

        self._zoom_label.setFixedWidth(50)

        self._zoom_label.setAlignment(Qt.AlignCenter)

        self._zoom_label.setStyleSheet("color: #808070; font-size: 10px;")

        bottom.addWidget(self._zoom_label)



        layout.addLayout(bottom)



        # Wire zoom from output viewer (primary)

        self._output_viewer._split_view.zoom_changed.connect(self._on_zoom_changed)

        self._output_viewer.view_mode_changed.connect(self.output_mode_changed.emit)



        # Tandem pan/zoom: pan or zoom on either viewer mirrors to the other.

        # set_view_transform does not re-emit view_transform_changed, so the

        # cross-connection cannot feed back.

        self._input_viewer._split_view.view_transform_changed.connect(

            self._output_viewer._split_view.set_view_transform)

        self._output_viewer._split_view.view_transform_changed.connect(

            self._input_viewer._split_view.set_view_transform)

        self._input_viewer._split_view.view_transform_changed.connect(

            self._wipe_overlay.set_view_transform)

        self._output_viewer._split_view.view_transform_changed.connect(

            self._wipe_overlay.set_view_transform)

        self._wipe_overlay.view_transform_changed.connect(

            self._input_viewer._split_view.set_view_transform)

        self._wipe_overlay.view_transform_changed.connect(

            self._output_viewer._split_view.set_view_transform)



        # Refresh wipe overlay whenever either viewer updates its displayed image (async, cache hit, mode switch, live preview)

        self._output_viewer.frame_updated.connect(self._load_wipe_images)

        self._input_viewer.frame_updated.connect(self._load_wipe_images)

        self._output_viewer.backdrop_changed.connect(self._on_backdrop_changed_wipe)

        self._output_viewer.backdrop_changed.connect(self.output_backdrop_changed.emit)

        self._output_viewer.custom_plate_requested.connect(self.custom_plate_requested.emit)

        self._input_viewer.pixel_hovered.connect(self.pixel_hovered.emit)

        self._output_viewer.pixel_hovered.connect(self.pixel_hovered.emit)

        self._wipe_overlay.pixel_hovered.connect(self.pixel_hovered.emit)

        self._input_viewer.polygon_completed.connect(self.polygon_completed.emit)

        self._input_viewer.polygon_vertex_moved.connect(self.polygon_vertex_moved.emit)

        self._input_viewer.mask_paint_stroke.connect(self.mask_paint_stroke.emit)

        self._output_viewer._mode_bar.inspect_mode_changed.connect(self._wipe_overlay.set_inspect_mode)



    @property

    def current_stem_index(self) -> int:

        """Current stem index for reprocess targeting."""

        return self._output_viewer.current_stem_index



    @property

    def current_output_mode(self) -> ViewMode:

        """Current output viewer mode."""

        return self._output_viewer._current_mode

    @property

    def current_backdrop(self) -> str:

        """Current output viewer compositing backdrop."""

        return self._output_viewer.current_backdrop



    @property

    def input_viewer(self) -> PreviewViewport:

        """Access the input (left) viewer."""

        return self._input_viewer



    @property

    def output_viewer(self) -> PreviewViewport:

        """Access the output (right) viewer."""

        return self._output_viewer



    # ── Eyedropper ──



    def set_eyedropper_mode(self, enabled: bool) -> None:

        """Toggle eyedropper on both viewports and wipe overlay.



        Always samples from the input frame so the screen color comes from

        the original footage regardless of which viewer the click lands on.

        """

        self._input_viewer.set_eyedropper_mode(enabled)

        self._output_viewer.set_eyedropper_mode(enabled)

        self._wipe_overlay.set_eyedropper_mode(enabled)

        if enabled:

            input_img = self._input_viewer._split_view._single_image

            self._output_viewer.set_eyedropper_source(input_img)

            self._wipe_overlay.set_eyedropper_source(input_img)

        else:

            self._output_viewer.set_eyedropper_source(None)

            self._wipe_overlay.set_eyedropper_source(None)



    # ── Annotation routing ──

    def setup_shared_annotations(self) -> None:
        """Configure annotation models.

        Input viewer is the dedicated prompt canvas; output viewer displays
        the keyed/segmented result without brush strokes overlay.
        """
        self._output_viewer._split_view.set_annotation_model(None)
        self._output_viewer.set_annotation_mode(None)
        self._input_viewer._split_view.set_annotation_sibling(None)
        self._output_viewer._split_view.set_annotation_sibling(None)

    def setup_shared_holdout(self) -> None:
        """Holdout mask is edited on the input viewer only."""
        self._output_viewer._split_view.set_holdout_model(None)

    def set_holdout_active(self, active: bool) -> None:
        """Route painting to holdout model on the input viewer."""
        self._input_viewer.set_holdout_active(active)
        self._output_viewer.set_holdout_active(False)



    def set_input_exr_is_linear(self, enabled: bool) -> None:

        """Keep both viewers aligned on INPUT-mode EXR display interpretation."""

        self._input_viewer.set_input_exr_is_linear(enabled)

        self._output_viewer.set_input_exr_is_linear(enabled)



    # ── Public API ──



    def set_clip(self, clip: ClipEntry) -> None:

        """Load a clip into both viewers and configure the scrubber."""

        same_clip = (
            self._clip is not None
            and clip is not None
            and self._clip.root_path == clip.root_path
            and self._clip.state != ClipState.EXTRACTING
        )
        prev_frame = self._scrubber.current_frame()

        self._clip = clip



        # Load into both viewers

        self._input_viewer.set_clip(clip)

        self._output_viewer.set_clip(clip)



        # Configure shared scrubber from the output viewer's frame index

        fi = self._output_viewer._frame_index

        if fi:

            fps = 24.0

            try:

                from backend.ffmpeg_tools import read_video_metadata

                meta = read_video_metadata(clip.root_path)

                meta_fps = float(meta.get("fps", 0.0)) if meta else 0.0

                if meta_fps > 0:

                    fps = meta_fps

                elif clip.input_asset is not None and clip.input_asset.asset_type == "video":

                    import cv2

                    from backend.frame_io import open_video

                    cap = open_video(clip.input_asset.path)

                    try:

                        if cap.isOpened():

                            cap_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

                            if 0.0 < cap_fps <= 240.0:

                                fps = cap_fps

                    finally:

                        cap.release()

            except Exception:

                pass

            self._scrubber.set_fps(fps)

            self._scrubber.set_range(fi.frame_count)

            if fi.frame_count > 0:

                if same_clip:
                    target_frame = max(0, min(prev_frame, fi.frame_count - 1))
                elif clip.in_out_range is not None:
                    target_frame = max(0, min(int(clip.in_out_range.in_point), fi.frame_count - 1))
                else:
                    target_frame = 0

                self._scrubber.set_frame(target_frame)

            self._update_coverage(clip, fi)



        # Restore in/out markers from clip data, default to full range

        if clip.in_out_range:

            self._scrubber.set_in_out(clip.in_out_range.in_point, clip.in_out_range.out_point)

        else:

            total = fi.frame_count if fi else 0

            if total > 0:

                self._scrubber.set_in_out(0, total - 1)

            else:

                # Use set_in_out(None, None) instead of clear_in_out() to avoid

                # triggering the range_cleared signal → _clear_in_out recursion loop.

                # (Fixes stack overflow crash on EXTRACTING clips — 0xc00000fd)

                self._scrubber.set_in_out(None, None)



    def load_preview_from_file(self, file_path: str, clip_name: str, frame_index: int) -> None:

        """Forward worker preview to the output viewer and advance timeline."""

        self._output_viewer.load_preview_from_file(file_path, clip_name, frame_index)

        fi = self._output_viewer._frame_index

        if fi:

            self._scrubber.set_range(fi.frame_count)

            target = max(0, min(frame_index - 1, fi.frame_count - 1))

            self._scrubber.set_frame(target)
            self._input_viewer.navigate_to_frame(target)

            if self._clip:

                self._update_coverage(self._clip, fi)



    def refresh_generated_assets(self) -> None:

        """Refresh output availability in place while preserving the current frame.



        Used for live progress updates so the scrubber coverage and mode buttons

        stay current without forcing a full clip reselection.

        """

        if self._clip is None:

            return



        current_frame = self._scrubber.current_frame()

        self._output_viewer.refresh_available_assets()



        fi = self._output_viewer._frame_index

        if fi is None:

            return



        self._scrubber.set_range(fi.frame_count)

        clamped_frame = min(current_frame, max(0, fi.frame_count - 1))

        if fi.frame_count > 0:

            self._scrubber.set_frame(clamped_frame)

            if self._input_viewer.current_stem_index != clamped_frame:

                self._input_viewer.navigate_to_frame(clamped_frame)

            self._output_viewer.navigate_to_frame(clamped_frame)

        self._update_coverage(self._clip, fi)



    def show_placeholder(self, text: str = "No clip selected") -> None:

        """Show placeholder on both viewers."""

        self._input_viewer.show_placeholder(text)

        self._output_viewer.show_placeholder(text)

        self._scrubber.set_range(0)

        self._clip = None



    def show_reprocess_preview(self, qimage: QImage) -> None:

        """Show a live reprocess result on the output viewer."""

        self._output_viewer.show_reprocess_preview(qimage)

        if self._wipe_active:

            self._load_wipe_images()

    def set_live_preview_modes(self, enabled: bool) -> None:

        """Enable or disable live SAM 3.1 view modes (MASK/ALPHA/COMP) on the output viewer."""

        self._sam3_live_modes = {ViewMode.MASK, ViewMode.ALPHA, ViewMode.COMP} if enabled else set()

        self._output_viewer.set_live_preview_modes(self._sam3_live_modes | self._render_live_modes)

    def set_live_render_modes(self, enabled: bool) -> None:

        """Enable or disable live CorridorKey render modes (FG/MATTE/COMP/PROC) in memory."""

        self._render_live_modes = (
            {ViewMode.FG, ViewMode.MATTE, ViewMode.COMP, ViewMode.PROCESSED}
            if enabled
            else set()
        )

        self._output_viewer.set_live_preview_modes(self._sam3_live_modes | self._render_live_modes)

    def clear_live_preview(self) -> None:

        """Release the live preview hold on the output viewer so disk/input decodes can paint."""

        self._output_viewer.clear_live_preview()



    def reset_zoom(self) -> None:

        """Reset zoom on both viewers."""

        self._input_viewer.reset_zoom()

        self._output_viewer.reset_zoom()

        self._wipe_overlay.reset_zoom()



    def set_busy(self, active: bool, message: str = "", progress: float | None = None) -> None:
        """Show or hide the centered Battle.net orbital busy indicator overlay."""
        if not hasattr(self, "_busy_overlay"):
            return
        if active:
            self._busy_overlay.show_busy(message, progress=progress)
        else:
            self._busy_overlay.hide_busy()

    def set_extraction_progress(self, progress: float, total: int) -> None:

        """Display extraction progress via the Battle.net orbital busy overlay."""

        self._input_viewer._split_view.set_extraction_progress(0.0, 0)
        if total > 0:
            pct = max(0, min(100, int(progress * 100)))
            self.set_busy(True, self.tr("Extracting frames... %d%%") % pct, progress=progress)
        else:
            self.set_busy(False)



    # ── In/Out Markers ──



    def set_in_out(self, in_point: int | None, out_point: int | None) -> None:

        """Forward in/out markers to the scrubber."""

        self._scrubber.set_in_out(in_point, out_point)



    def get_in_out(self) -> tuple[int | None, int | None]:

        """Return current (in_point, out_point) from the scrubber."""

        return self._scrubber.get_in_out()



    # ── Coverage ──



    def _update_coverage(self, clip: ClipEntry, fi: FrameIndex) -> None:

        """Compute alpha and inference coverage arrays and update the scrubber."""

        stems = fi.stems

        if not stems:

            self._scrubber.set_coverage([], [])

            return



        # Alpha coverage: use frame index availability for ALPHA mode

        alpha_stems = fi.availability.get(ViewMode.ALPHA, set())



        # Inference coverage: any output mode (FG, Matte, Comp, Processed) has this stem

        output_modes = (ViewMode.FG, ViewMode.MATTE, ViewMode.COMP, ViewMode.PROCESSED)

        inference_stems: set[str] = set()

        for mode in output_modes:

            inference_stems |= fi.availability.get(mode, set())



        # Build boolean arrays aligned to stems

        alpha_coverage = [s in alpha_stems for s in stems]

        inference_coverage = [s in inference_stems for s in stems]



        self._scrubber.set_coverage(alpha_coverage, inference_coverage)



    # ── Navigation ──



    @Slot(int)

    def _on_scrubber_frame(self, stem_index: int) -> None:

        """Shared scrubber drives both viewers to the same frame."""

        self._input_viewer.navigate_to_frame(stem_index)

        self._output_viewer.navigate_to_frame(stem_index)

        if self._wipe_active:

            self._load_wipe_images()

        self.frame_changed.emit(stem_index)



    def toggle_playback(self) -> None:

        """Forward play/pause to the scrubber."""

        self._scrubber.toggle_playback()



    @Slot(float)

    def _on_zoom_changed(self, zoom: float) -> None:

        self._zoom_label.setText(f"{int(zoom * 100)}%")



    # ── A/B Wipe ──



    def toggle_wipe_mode(self) -> None:

        """Toggle A/B wipe comparison mode.



        Shows/hides a full-width overlay on top of both viewer canvases.

        Nothing above the viewer (top bars, buttons) changes at all.

        """

        self._wipe_active = not self._wipe_active

        self._ab_button.setChecked(self._wipe_active)



        if self._wipe_active:

            self._wipe_overlay.set_view_transform(*self._output_viewer._split_view.view_transform())

            self._load_wipe_images()

            self._position_wipe_overlay()

            self._wipe_overlay.show()

            self._wipe_overlay.raise_()

        else:

            self._snapshot_image = None

            self._snap_button.setChecked(False)

            self._wipe_overlay.hide()

    def set_active_object_id(self, object_id: int) -> None:

        self._input_viewer.set_active_object_id(object_id)

    def set_polygon_model(self, model) -> None:

        self._input_viewer.set_polygon_model(model)

        self._output_viewer.set_polygon_model(model)

        self._wipe_overlay.set_polygon_model(model)

    def set_alpha_probe_array(self, alpha_arr) -> None:

        self._output_viewer.set_alpha_probe_array(alpha_arr)

        self._wipe_overlay.set_alpha_probe_array(alpha_arr)

    def freeze_snapshot(self, enable: bool | None = None) -> bool:

        """Freeze current output image as Side A in A/B wipe comparison against live Side B."""

        target = (self._snapshot_image is None) if enable is None else bool(enable)

        if target:

            sv = self._output_viewer._split_view

            out_img = sv._right_image if sv.split_enabled else (sv._single_image or sv._right_image)

            if out_img is None or out_img.isNull():

                self._snap_button.setChecked(False)

                return False

            self._snapshot_image = out_img.copy()

            self._snap_button.setChecked(True)

            if not self._wipe_active:

                self._wipe_active = True

                self._ab_button.setChecked(True)

                self._wipe_overlay.set_view_transform(*self._output_viewer._split_view.view_transform())

                self._position_wipe_overlay()

                self._wipe_overlay.show()

                self._wipe_overlay.raise_()

            self._load_wipe_images()

            return True

        self._snapshot_image = None

        self._snap_button.setChecked(False)

        if self._wipe_active:

            self._load_wipe_images()

        return False

    def toggle_snapshot_compare(self) -> None:

        self.freeze_snapshot()



    def _position_wipe_overlay(self) -> None:

        """Position the wipe overlay to cover both viewer canvases (below top bars)."""

        # Map the splitter's geometry to our coordinate system

        # The overlay should cover the canvas area of both viewers

        # (below the 30px top bars, above the scrubber)

        splitter_geo = self._viewer_splitter.geometry()

        top_bar_h = 30  # fixed top bar height

        x = splitter_geo.x()

        y = splitter_geo.y() + top_bar_h

        w = splitter_geo.width()

        h = splitter_geo.height() - top_bar_h

        self._wipe_overlay.setGeometry(x, y, w, h)



    def resizeEvent(self, event) -> None:

        super().resizeEvent(event)

        if self._wipe_active:

            self._position_wipe_overlay()

        if hasattr(self, "_busy_overlay") and self._busy_overlay.isVisible():

            self._busy_overlay.recenter()



    @Slot(int, str, object)

    def _on_wipe_frame_ready(self, stem_index: int, mode_value: str, qimage: object) -> None:

        """A viewer decoded a frame — refresh wipe if active."""

        if self._wipe_active:

            self._load_wipe_images()

    @Slot(str)

    def _on_backdrop_changed_wipe(self, _backdrop: str) -> None:

        """Refresh wipe image if backdrop changed while wipe is open."""

        if self._wipe_active:

            self._load_wipe_images()



    def _load_wipe_images(self) -> None:

        """Load INPUT (A=left) and current output mode (B=right) into wipe overlay."""

        if not self._wipe_active:

            return

        input_img = self._input_viewer._split_view._single_image

        output_img = self._output_viewer._split_view._single_image

        left_img = self._snapshot_image if (self._snapshot_image is not None and not self._snapshot_image.isNull()) else input_img

        if left_img:

            self._wipe_overlay.set_left_image(left_img)

        if output_img:

            self._wipe_overlay.set_right_image(output_img)

