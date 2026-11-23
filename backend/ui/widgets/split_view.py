"""Split view widget with draggable yellow divider, zoom, and pan.

Renders two QImages (left=before, right=after) with a split divider.
When split is disabled, renders a single image full-width.

Uses QImage as internal currency (Codex finding: QPixmap can use
platform-native GPU backing, QImage is guaranteed CPU-only).

Zoom/pan: Ctrl+wheel zooms, middle-click pans, double-click resets.
Hit-test precedence: divider drag > pan > annotation > zoom (Codex finding).

Annotation mode: hotkey 1/2 activates green/red brush. Left-click draws
strokes in image-pixel coordinates. Shift+drag resizes brush.
"""
from __future__ import annotations

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QPainter, QPen, QColor, QImage, QMouseEvent, QWheelEvent,
    QCursor,
)

from ui.widgets.annotation_overlay import (
    AnnotationModel, paint_annotations, paint_brush_cursor,
    paint_resize_indicator, paint_annotation_hud,
)
from ui.widgets.wipe_controller import (
    wipe_line_endpoints, wipe_handle_rect, wipe_distance_to_line,
    paint_wipe, handle_wipe_press, handle_wipe_drag,
    wipe_cursor_for_pos, handle_wipe_scroll,
)

import os as _os
from functools import lru_cache as _lru_cache
from PySide6.QtGui import QPixmap as _QPixmap


@_lru_cache(maxsize=1)
def _eyedropper_cursor() -> QCursor:
    """Build a QCursor from the eyedropper PNG. Hotspot at tip (1, 29)."""
    from ui.theme import THEME_DIR
    png_path = _os.path.join(THEME_DIR, 'icons', 'eyedropper.png')
    pix = _QPixmap(png_path)
    return QCursor(pix, 1, 29)


class SplitViewWidget(QWidget):
    """Image display with optional split view, zoom, and pan."""

    zoom_changed = Signal(float)  # current zoom level
    view_transform_changed = Signal(float, QPointF)  # user pan/zoom, for tandem viewer sync
    stroke_finished = Signal()    # emitted when an annotation stroke completes
    color_sampled = Signal(int, int, int)  # eyedropper: final averaged RGB on release
    color_preview = Signal(int, int, int)  # eyedropper: live running-average RGB during drag
    screen_samples_ready = Signal(object)  # eyedropper: full list of (r,g,b) samples on release
    pixel_hovered = Signal(int, int, int, int, int, float)  # x, y, r, g, b, alpha
    polygon_completed = Signal(str, object)                 # kind ("garbage"|"core"), [(nx, ny), ...]
    polygon_vertex_moved = Signal(str, int, int, float, float)  # shape_id, frame_idx, vert_idx, nx, ny
    mask_paint_stroke = Signal(int, str, object, float)     # frame_idx, "add"|"erase", [(x, y), ...], radius

    # Divider hit zone (pixels from divider line)
    _DIVIDER_HIT_ZONE = 8
    _DIVIDER_WIDTH = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Images
        self._left_image: QImage | None = None
        self._right_image: QImage | None = None
        self._single_image: QImage | None = None  # used when split disabled

        # Split state
        self._split_enabled = False
        self._divider_pos = 0.5  # normalized 0.0-1.0

        # Wipe mode (A/B comparison — diagonal divider, A=left, B=right)
        self._wipe_mode = False
        self._wipe_angle = -45.0      # degrees, default: bottom-left to top-right. Range -90 to 90
        self._wipe_offset = 0.0       # perpendicular offset from center (-0.5 to 0.5)
        self._wipe_dragging: str | None = None  # "handle" or "line" or None
        self._wipe_drag_start = QPointF()
        self._wipe_drag_start_offset = 0.0
        self._wipe_drag_start_angle = 45.0

        # Drag state
        self._dragging_divider = False

        # Zoom/pan
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_start = QPointF()
        self._pan_start_offset = QPointF()
        self._pan_pressed = False   # left button down, becomes a pan past threshold
        self._pan_did_drag = False  # last press panned; suppress double-click reset
        self._space_pressed = False

        # Zoom limits
        self._zoom_min = 0.25
        self._zoom_max = 8.0

        # Placeholder text
        self._placeholder = "No clip selected"

        # Extraction progress overlay
        self._extraction_progress = 0.0  # 0.0 to 1.0
        self._extraction_total = 0

        # Annotation state
        self._annotation_mode: str | None = None  # "fg", "bg", or None
        self._annotation_model: AnnotationModel | None = None
        self._annotation_sibling: SplitViewWidget | None = None
        self._annotation_stem_idx: int = -1
        self._brush_radius: float = 15.0  # image pixels
        self._drawing: bool = False
        self._resizing_brush: bool = False
        self._resize_start_y: float = 0.0
        self._resize_start_radius: float = 0.0
        self._mouse_pos: QPointF = QPointF()  # last known mouse position (display)
        self._straight_line: bool = False    # Alt+click straight-line mode
        self._line_anchor: tuple[float, float] | None = None  # anchor in image-pixel coords
        self._active_object_id: int = 1
        self._polygon_model = None
        self._poly_points: list[tuple[float, float]] = []
        self._dragging_poly_vertex: tuple[str, int, float, float] | None = None
        self._mask_paint_points: list[tuple[float, float]] = []
        self._alpha_probe_array = None
        self._inspect_mode: str = "rgb"

        # Holdout mask state (chroma key forced regions)
        self._holdout_model: AnnotationModel | None = None
        self._holdout_active: bool = False

        # Eyedropper state
        self._eyedropper_mode: bool = False
        self._eyedropper_source: QImage | None = None  # input frame to sample from
        self._eyedropper_sampling: bool = False  # True while drag-sampling
        self._eyedropper_samples: list[tuple[int, int, int]] = []  # accumulated RGB samples
        self._eyedropper_last_img_pos: tuple[int, int] | None = None  # last sampled image coord
        self._eyedropper_preview_color: tuple[int, int, int] | None = None  # running avg for overlay

    # ── Public API ──

    @property
    def is_annotating(self) -> bool:
        """True when annotation mode is active (brush tool enabled)."""
        return self._annotation_mode is not None

    def set_image(self, image: QImage | None) -> None:
        """Set the single (non-split) image."""
        self._single_image = image
        self.update()

    def set_left_image(self, image: QImage | None) -> None:
        """Set the left (before/input) image for split view."""
        self._left_image = image
        self.update()

    def set_right_image(self, image: QImage | None) -> None:
        """Set the right (after/output) image for split view."""
        self._right_image = image
        self.update()

    def set_split_enabled(self, enabled: bool) -> None:
        """Toggle split view on/off."""
        self._split_enabled = enabled
        self.update()

    @property
    def split_enabled(self) -> bool:
        return self._split_enabled

    def set_wipe_mode(self, enabled: bool) -> None:
        """Toggle A/B wipe comparison (diagonal divider, A above/left, B below/right)."""
        self._wipe_mode = enabled
        self._wipe_angle = -45.0
        self._wipe_offset = 0.0
        self._wipe_dragging = None
        self.update()

    @property
    def wipe_mode(self) -> bool:
        return self._wipe_mode

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self._single_image = None
        self._left_image = None
        self._right_image = None
        self.update()

    def set_extraction_progress(self, progress: float, total: int) -> None:
        """Update extraction progress overlay. Set total=0 to clear."""
        self._extraction_progress = progress
        self._extraction_total = total
        self.update()

    def reset_zoom(self) -> None:
        """Reset zoom to fit and center pan."""
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.zoom_changed.emit(self._zoom)
        self.view_transform_changed.emit(self._zoom, QPointF(self._pan))
        self.update()

    # ── Tandem view sync API ──

    def view_transform(self) -> tuple[float, QPointF]:
        """Current (zoom, pan) of this viewer."""
        return self._zoom, QPointF(self._pan)

    def set_view_transform(self, zoom: float, pan: QPointF) -> None:
        """Apply a zoom/pan from a synced sibling viewer.

        Does not re-emit view_transform_changed, so two viewers can be
        cross-connected without a feedback loop. Emits zoom_changed so
        zoom labels stay accurate.
        """
        zoom = max(self._zoom_min, min(self._zoom_max, zoom))
        if (abs(zoom - self._zoom) < 1e-4
                and abs(pan.x() - self._pan.x()) < 0.01
                and abs(pan.y() - self._pan.y()) < 0.01):
            return
        self._zoom = zoom
        self._pan = QPointF(pan)
        self.zoom_changed.emit(self._zoom)
        self.update()

    # ── Eyedropper API ──

    def set_eyedropper_mode(self, enabled: bool) -> None:
        """Toggle eyedropper color sampling mode."""
        self._eyedropper_mode = enabled
        if enabled:
            self.setCursor(_eyedropper_cursor())
        elif not self._annotation_mode:
            self.unsetCursor()
        self.update()

    def set_eyedropper_source(self, image: QImage | None) -> None:
        """Set the source image for eyedropper sampling (always the input frame)."""
        self._eyedropper_source = image

    def _sample_eyedropper_at(self, display_pos: QPointF) -> None:
        """Sample pixel(s) at display_pos, interpolating from last position."""
        source = self._eyedropper_source or self._annotation_target_image()
        if source is None:
            return
        displayed = self._single_image or self._left_image
        if displayed is None:
            return
        dest = self._image_rect(displayed)
        iw, ih = source.width(), source.height()
        dw, dh = displayed.width(), displayed.height()
        if iw <= 0 or ih <= 0 or dw <= 0 or dh <= 0 or dest.width() <= 0 or dest.height() <= 0:
            return

        img_x = (display_pos.x() - dest.x()) * dw / dest.width()
        img_y = (display_pos.y() - dest.y()) * dh / dest.height()
        sx = int(min(max(img_x * iw / dw, 0), iw - 1))
        sy = int(min(max(img_y * ih / dh, 0), ih - 1))

        # Interpolate between last and current position so fast drags don't skip
        points = [(sx, sy)]
        if self._eyedropper_last_img_pos is not None:
            lx, ly = self._eyedropper_last_img_pos
            dx, dy = sx - lx, sy - ly
            dist = max(abs(dx), abs(dy))
            if dist > 1:
                for i in range(1, dist):
                    t = i / dist
                    points.append((int(lx + dx * t), int(ly + dy * t)))

        for px, py in points:
            pixel = source.pixelColor(px, py)
            self._eyedropper_samples.append((pixel.red(), pixel.green(), pixel.blue()))

        self._eyedropper_last_img_pos = (sx, sy)

        # Emit running average as preview and store for overlay painting
        n = len(self._eyedropper_samples)
        avg_r = sum(s[0] for s in self._eyedropper_samples) // n
        avg_g = sum(s[1] for s in self._eyedropper_samples) // n
        avg_b = sum(s[2] for s in self._eyedropper_samples) // n
        self._eyedropper_preview_color = (avg_r, avg_g, avg_b)
        self.color_preview.emit(avg_r, avg_g, avg_b)
        self.update()

    # ── Annotation API ──

    def set_annotation_mode(self, mode: str | None) -> None:
        """Set annotation mode: 'fg', 'bg', 'poly_garbage', 'poly_core', 'mask_paint_fg', 'mask_paint_bg', or None."""
        if mode != self._annotation_mode:
            self._poly_points.clear()
            self._mask_paint_points.clear()
            self._drawing = False
        self._annotation_mode = mode
        if mode:
            self.setCursor(Qt.CrossCursor)
        elif not self._eyedropper_mode:
            self.unsetCursor()
        self.update()

    def set_active_object_id(self, object_id: int) -> None:
        self._active_object_id = max(1, min(4, int(object_id)))
        self.update()

    def set_polygon_model(self, model) -> None:
        self._polygon_model = model
        self.update()

    def set_alpha_probe_array(self, alpha_arr) -> None:
        """Optional float32 [H, W] array used by Pixel Inspector for exact alpha readout."""
        self._alpha_probe_array = alpha_arr

    def set_inspect_mode(self, mode: str) -> None:
        """Set channel / key inspection overlay ('rgb', 'r', 'g', 'b', 'status', 'rubylith')."""
        m = (mode or "rgb").strip().lower()
        if m not in ("rgb", "r", "g", "b", "status", "rubylith"):
            m = "rgb"
        if m != self._inspect_mode:
            self._inspect_mode = m
            self.update()

    def _transform_display_qimage(self, qimg: QImage | None) -> QImage | None:
        """Apply viewer-only inspection view and/or Display .cube LUT without touching source files."""
        if qimg is None or qimg.isNull():
            return qimg
        from backend.color_management import get_display_lut_state
        lut_on, _ = get_display_lut_state()
        if self._inspect_mode == "rgb" and not lut_on:
            return qimg
        from ui.preview.display_transform import apply_inspection_filter
        return apply_inspection_filter(qimg, self._inspect_mode, self._alpha_probe_array)

    def set_annotation_model(self, model: AnnotationModel | None) -> None:
        self._annotation_model = model

    def set_annotation_sibling(self, sibling: "SplitViewWidget | None") -> None:
        """Set the sibling viewer so annotation strokes repaint both sides live."""
        self._annotation_sibling = sibling

    def set_annotation_stem_index(self, idx: int) -> None:
        self._annotation_stem_idx = idx
        self.update()

    @property
    def annotation_mode(self) -> str | None:
        return self._annotation_mode

    def set_holdout_model(self, model: AnnotationModel | None) -> None:
        self._holdout_model = model

    def set_holdout_active(self, active: bool) -> None:
        self._holdout_active = active
        self.update()

    @property
    def _active_annotation_model(self) -> AnnotationModel | None:
        """The model that should receive mouse strokes right now."""
        if self._holdout_active and self._holdout_model is not None:
            return self._holdout_model
        return self._annotation_model

    @property
    def _active_stem_idx(self) -> int:
        """Stem index for the active model. Holdout uses 0 (frame-independent)."""
        if self._holdout_active:
            return 0
        return self._annotation_stem_idx

    # ── Paint ──

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Background
        painter.fillRect(self.rect(), QColor("#0A0A00"))

        if self._wipe_mode and self._left_image and self._right_image:
            self._paint_wipe(painter)
        elif self._split_enabled and (self._left_image or self._right_image):
            self._paint_split(painter)
        elif self._single_image:
            self._paint_single(painter)
        elif self._extraction_total <= 0:
            self._paint_placeholder(painter)

        # Annotation overlay (after image, before extraction)
        if self._active_annotation_model is not None and self._annotation_mode:
            self._paint_annotations(painter)

        # Polygon matte & post-track mask paint overlays
        self._paint_polygon_and_mask_overlays(painter)

        # Extraction progress overlay (replaces placeholder during extraction)
        if self._extraction_total > 0:
            self._paint_extraction_overlay(painter)

        # Eyedropper HUD + floating color chip
        if self._eyedropper_mode:
            self._paint_eyedropper_hud(painter)
        if self._eyedropper_sampling and self._eyedropper_preview_color:
            self._paint_eyedropper_chip(painter)

        painter.end()

    def _paint_single(self, painter: QPainter) -> None:
        """Draw single image with zoom/pan."""
        img = self._transform_display_qimage(self._single_image)
        if img is None:
            return

        dest = self._image_rect(img)
        painter.drawImage(dest, img)

    def _paint_split(self, painter: QPainter) -> None:
        """Draw split view with left/right images and divider."""
        w = self.width()
        h = self.height()
        divider_x = int(w * self._divider_pos)

        # Left side — fit image into left panel
        if self._left_image:
            left_vp = QRectF(0, 0, divider_x, h)
            dest = self._image_rect(self._left_image, viewport=left_vp)
            painter.setClipRect(0, 0, divider_x, h)
            painter.drawImage(dest, self._left_image)

        # Right side — fit image into right panel
        if self._right_image:
            right_img = self._transform_display_qimage(self._right_image)
            right_vp = QRectF(divider_x, 0, w - divider_x, h)
            dest = self._image_rect(right_img, viewport=right_vp)
            painter.setClipRect(divider_x, 0, w - divider_x, h)
            painter.drawImage(dest, right_img)

        # Remove clip for divider drawing
        painter.setClipping(False)

        # Divider line
        pen = QPen(QColor("#FFF203"), self._DIVIDER_WIDTH)
        painter.setPen(pen)
        painter.drawLine(divider_x, 0, divider_x, self.height())

        # Handle triangles at top and bottom
        handle_size = 8
        painter.setBrush(QColor("#FFF203"))
        painter.setPen(Qt.NoPen)

        # Top triangle
        top_points = [
            (divider_x - handle_size, 0),
            (divider_x + handle_size, 0),
            (divider_x, handle_size),
        ]
        from PySide6.QtGui import QPolygon
        from PySide6.QtCore import QPoint
        painter.drawPolygon(QPolygon([QPoint(x, y) for x, y in top_points]))

        # Bottom triangle
        h = self.height()
        bot_points = [
            (divider_x - handle_size, h),
            (divider_x + handle_size, h),
            (divider_x, h - handle_size),
        ]
        painter.drawPolygon(QPolygon([QPoint(x, y) for x, y in bot_points]))

    # ── Wipe mode rendering (delegated to wipe_controller) ──

    def _wipe_line_endpoints(self):
        """Compute the wipe line endpoints from angle + offset."""
        return wipe_line_endpoints(
            self.width(), self.height(), self._wipe_angle, self._wipe_offset)

    def _wipe_handle_rect(self, center: QPointF, hit=False) -> QRectF:
        """Return the center square handle rect. hit=True returns 2x hitbox."""
        return wipe_handle_rect(center, hit)

    def _paint_wipe(self, painter: QPainter) -> None:
        """Draw A/B wipe comparison with diagonal divider."""
        paint_wipe(
            painter,
            self.width(), self.height(),
            self._wipe_angle, self._wipe_offset,
            self._left_image, self._transform_display_qimage(self._right_image),
            self._image_rect,
            self._DIVIDER_WIDTH,
        )

    def _paint_placeholder(self, painter: QPainter) -> None:
        """Draw placeholder text."""
        painter.setPen(QColor("#808070"))
        font = painter.font()
        font.setPointSize(16)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder)

    def _paint_eyedropper_hud(self, painter: QPainter) -> None:
        """Draw a HUD overlay when eyedropper mode is active."""
        painter.save()
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        text = "Click + drag across varied background tones for best key"
        hud_rect = QRectF(12, 12, 420, 24)
        painter.fillRect(hud_rect, QColor(0, 0, 0, 150))
        painter.setPen(QColor(255, 242, 3, 220))
        painter.drawText(hud_rect, Qt.AlignCenter, text)
        painter.restore()

    def _paint_eyedropper_chip(self, painter: QPainter) -> None:
        """Draw a floating color chip near the cursor during eyedropper drag."""
        r, g, b = self._eyedropper_preview_color
        mx, my = int(self._mouse_pos.x()), int(self._mouse_pos.y())
        chip_size = 36
        # Offset chip to bottom-right of cursor
        cx, cy = mx + 20, my + 20
        # Keep on screen
        if cx + chip_size > self.width():
            cx = mx - chip_size - 8
        if cy + chip_size > self.height():
            cy = my - chip_size - 8

        # Drop shadow
        painter.fillRect(cx + 2, cy + 2, chip_size, chip_size, QColor(0, 0, 0, 100))
        # Color fill
        painter.fillRect(cx, cy, chip_size, chip_size, QColor(r, g, b))
        # Border
        painter.setPen(QPen(QColor("#FFF203"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(cx, cy, chip_size, chip_size)
        # Sample count
        n = len(self._eyedropper_samples)
        if n > 1:
            painter.setPen(QColor("#FFFFFF"))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawText(cx, cy + chip_size + 12, f"{n}px")

    def _paint_extraction_overlay(self, painter: QPainter) -> None:
        """Draw extraction progress bar and percentage centered on the viewer."""
        w, h = self.width(), self.height()
        pct = int(self._extraction_progress * 100)
        current = int(self._extraction_progress * self._extraction_total)

        bar_w = int(w * 0.6)
        bar_h = 8
        bar_x = (w - bar_w) // 2

        # "Extracting frames..." header above the bar
        font = painter.font()
        font.setPointSize(14)
        painter.setFont(font)
        header_rect = QRectF(bar_x, h // 2 - 30, bar_w, 28)
        # Semi-transparent background for readability
        painter.fillRect(header_rect.adjusted(-8, -2, 8, 2), QColor(0, 0, 0, 140))
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(header_rect, Qt.AlignCenter, self.tr("Extracting frames..."))

        # Progress bar below header
        bar_y = h // 2

        # Track background
        painter.fillRect(bar_x, bar_y, bar_w, bar_h, QColor(30, 29, 0))

        # Fill
        fill_w = int(bar_w * self._extraction_progress)
        if fill_w > 0:
            painter.fillRect(bar_x, bar_y, fill_w, bar_h, QColor("#FFF203"))

        # Track border
        painter.setPen(QColor("#454430"))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(bar_x, bar_y, bar_w - 1, bar_h - 1)

        # Progress text below bar
        font.setPointSize(11)
        painter.setFont(font)
        text_rect = QRectF(bar_x, bar_y + bar_h + 8, bar_w, 24)
        # Semi-transparent background for readability
        painter.fillRect(text_rect.adjusted(-8, -2, 8, 2), QColor(0, 0, 0, 140))
        painter.setPen(QColor("#FF8C00"))
        painter.drawText(
            text_rect, Qt.AlignCenter,
            self.tr("%d%%  (%d/%d frames)") % (pct, current, self._extraction_total),
        )

    def _paint_annotations(self, painter: QPainter) -> None:
        """Draw annotation strokes and brush cursor on the viewport."""
        img = self._annotation_target_image()
        active_model = self._active_annotation_model
        if img is None or active_model is None:
            return

        dest = self._image_rect(img, viewport=self._annotation_viewport())
        paintable_rect = self._annotation_paint_rect(img)
        if paintable_rect is None:
            return
        iw, ih = img.width(), img.height()

        # Draw existing strokes + in-progress stroke
        strokes = active_model.get_strokes(self._active_stem_idx)
        current = active_model.current_stroke
        painter.save()
        painter.setClipRect(paintable_rect)
        paint_annotations(painter, strokes, current, dest, iw, ih,
                          outline_only=self._holdout_active)
        painter.restore()
        is_alt = bool(QApplication.keyboardModifiers() & Qt.AltModifier)
        curr_mode = self._annotation_mode
        if is_alt and curr_mode in ("brush", "fg", "point"):
            eff_brush = "bg"
        else:
            eff_brush = curr_mode

        if self._holdout_active:
            hud_label = "Holdout mask"
        elif curr_mode == "poly_garbage":
            hud_label = "Polygon Garbage Matte (Click vertices, Double-click/Enter to close, Right-click undo)"
        elif curr_mode == "poly_core":
            hud_label = "Polygon Core Matte (Click vertices, Double-click/Enter to close, Right-click undo)"
        elif curr_mode in ("mask_paint_fg", "mask_paint_bg"):
            action_word = "Erase -" if (curr_mode == "mask_paint_bg" or is_alt) else "Paint +"
            hud_label = f"Post-Track Mask Fix [{action_word}] (Edits saved mask directly, Shift+drag resize)"
        elif curr_mode == "eraser":
            hud_label = "SAM 3.1 Eraser [E] (Click/drag to delete points & strokes | Right-click anytime)"
        elif curr_mode == "hair":
            hud_label = "SAM 3.1 Hair & Soft Detail Brush [H] (Paint hair, fur, whiskers, semi-transparent edges)"
        elif curr_mode == "box":
            hud_label = "SAM 3.1 Bounding Box Prompt [R] (Drag rectangle over subject)"
        elif curr_mode == "point":
            if is_alt:
                hud_label = "SAM 3.1 Negative Point [Alt+Click] (Click to cut out / exclude background)"
            else:
                hud_label = "SAM 3.1 Foreground Point [P] (Click to include subject | Alt: Negative)"
        else:
            if is_alt:
                hud_label = "SAM 3.1 Negative Cut Brush [Alt+Drag] (Drag to cut out unwanted background)"
            else:
                hud_label = "SAM 3.1 Foreground Brush [B] (Drag to include subject | Alt: Negative Cut)"

        paint_annotation_hud(
            painter,
            image_rect=paintable_rect,
            brush_type=eff_brush,
            radius_image=self._brush_radius,
            label_suffix=hud_label,
        )

        # Brush cursor at mouse position
        if (
            self._annotation_mode
            and self._annotation_mode not in ("poly_garbage", "poly_core")
            and self.underMouse()
            and paintable_rect.contains(self._mouse_pos)
        ):
            radius_display = self._brush_radius * dest.width() / iw
            cursor_brush = (
                "fg"
                if eff_brush == "mask_paint_fg"
                else ("bg" if eff_brush == "mask_paint_bg" else eff_brush)
            )
            if self._resizing_brush:
                paint_resize_indicator(
                    painter, self._mouse_pos, radius_display,
                    self._brush_radius, cursor_brush,
                )
            else:
                paint_brush_cursor(
                    painter, self._mouse_pos, radius_display,
                    cursor_brush,
                )

    def _finish_polygon(self) -> None:
        if len(self._poly_points) >= 3 and self._annotation_mode in ("poly_garbage", "poly_core"):
            kind = "core" if self._annotation_mode == "poly_core" else "garbage"
            pts = list(self._poly_points)
            self._poly_points.clear()
            self.polygon_completed.emit(kind, pts)
            self.update()
        else:
            self._poly_points.clear()
            self.update()

    def _paint_polygon_and_mask_overlays(self, painter: QPainter) -> None:
        img = self._annotation_target_image()
        if img is None or img.width() <= 0 or img.height() <= 0:
            return
        dest = self._image_rect(img, viewport=self._annotation_viewport())
        iw, ih = img.width(), img.height()

        has_shapes = bool(self._polygon_model and getattr(self._polygon_model, "shapes", None))
        is_poly_mode = self._annotation_mode in ("poly_garbage", "poly_core")
        has_mask_paint = bool(self._mask_paint_points)
        if not has_shapes and not is_poly_mode and not has_mask_paint:
            return

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        paintable = self._annotation_paint_rect(img)
        if paintable is not None:
            painter.setClipRect(paintable)

        if has_shapes:
            frame_idx = max(0, int(self._annotation_stem_idx))
            for shape in self._polygon_model.shapes:
                if not getattr(shape, "enabled", True):
                    continue
                pts = shape.points_at(frame_idx)
                if len(pts) < 3:
                    continue
                col = QColor(0, 220, 140, 210) if shape.kind == "core" else QColor(255, 165, 0, 210)
                fill = QColor(0, 220, 140, 28) if shape.kind == "core" else QColor(255, 165, 0, 28)
                pen = QPen(col, 1.8 if is_poly_mode else 1.2, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(fill if is_poly_mode else Qt.NoBrush)
                from PySide6.QtGui import QPolygonF
                qpoly = QPolygonF([
                    QPointF(
                        dest.x() + (x * (iw - 1) / iw) * dest.width(),
                        dest.y() + (y * (ih - 1) / ih) * dest.height(),
                    )
                    for x, y in pts
                ])
                painter.drawPolygon(qpoly)
                if is_poly_mode:
                    ctrl_pts = shape.raw_points_at(frame_idx)
                    painter.setBrush(col)
                    for x, y in ctrl_pts:
                        qp = QPointF(
                            dest.x() + (x * (iw - 1) / iw) * dest.width(),
                            dest.y() + (y * (ih - 1) / ih) * dest.height(),
                        )
                        painter.drawEllipse(qp, 4.0, 4.0)

        if self._poly_points:
            col = QColor(0, 240, 180, 240) if self._annotation_mode == "poly_core" else QColor(255, 190, 20, 240)
            painter.setPen(QPen(col, 2.0, Qt.SolidLine))
            dpts = [
                QPointF(
                    dest.x() + (x * (iw - 1) / iw) * dest.width(),
                    dest.y() + (y * (ih - 1) / ih) * dest.height(),
                )
                for x, y in self._poly_points
            ]
            for i in range(len(dpts) - 1):
                painter.drawLine(dpts[i], dpts[i + 1])
            if paintable is not None and paintable.contains(self._mouse_pos):
                painter.setPen(QPen(col, 1.2, Qt.DashLine))
                painter.drawLine(dpts[-1], self._mouse_pos)
            painter.setBrush(col)
            for idx, dp in enumerate(dpts):
                r = 5.0 if idx == 0 else 3.5
                painter.drawEllipse(dp, r, r)

        if self._mask_paint_points:
            is_erase = (
                self._annotation_mode == "mask_paint_bg"
                or bool(QApplication.keyboardModifiers() & Qt.AltModifier)
            )
            col = QColor(230, 55, 55, 160) if is_erase else QColor(50, 220, 120, 160)
            disp_r = max(1.0, self._brush_radius * dest.width() / iw)
            pen = QPen(col, disp_r * 2)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            dpts = [
                QPointF(
                    dest.x() + (px / iw) * dest.width(),
                    dest.y() + (py / ih) * dest.height(),
                )
                for px, py in self._mask_paint_points
            ]
            if len(dpts) == 1:
                painter.setPen(Qt.NoPen)
                painter.setBrush(col)
                painter.drawEllipse(dpts[0], disp_r, disp_r)
            else:
                for i in range(len(dpts) - 1):
                    painter.drawLine(dpts[i], dpts[i + 1])

        painter.restore()

    def _emit_pixel_hover(self, display_pos: QPointF) -> None:
        img = None
        dest = None
        if self._split_enabled:
            divider_x = self.width() * self._divider_pos
            if display_pos.x() < divider_x and self._left_image is not None:
                img = self._left_image
                dest = self._image_rect(img, viewport=QRectF(0, 0, divider_x, float(self.height())))
            elif self._right_image is not None:
                img = self._right_image
                dest = self._image_rect(img, viewport=QRectF(divider_x, 0, float(self.width()) - divider_x, float(self.height())))
        else:
            img = self._single_image or self._right_image or self._left_image
            if img is not None:
                dest = self._image_rect(img)
        if img is None or dest is None or img.width() <= 0 or img.height() <= 0 or not dest.contains(display_pos):
            return
        ix = max(0, min(img.width() - 1, int((display_pos.x() - dest.x()) * img.width() / max(1.0, dest.width()))))
        iy = max(0, min(img.height() - 1, int((display_pos.y() - dest.y()) * img.height() / max(1.0, dest.height()))))
        px_col = img.pixelColor(ix, iy)
        alpha_val = px_col.alphaF()
        arr = self._alpha_probe_array
        if arr is not None and iy < arr.shape[0] and ix < arr.shape[1]:
            raw = arr[iy, ix] if arr.ndim == 2 else arr[iy, ix, 0]
            alpha_val = float(raw) / 255.0 if arr.dtype.kind in ("u", "i") else float(raw)
        self.pixel_hovered.emit(ix, iy, px_col.red(), px_col.green(), px_col.blue(), max(0.0, min(1.0, float(alpha_val))))

    def _display_to_image(self, display_pos: QPointF) -> tuple[float, float] | None:
        """Convert display coordinates to image-pixel coordinates.

        Returns None if the position is outside the image bounds.
        """
        img = self._annotation_target_image()
        if img is None:
            return None
        dest = self._image_rect(img, viewport=self._annotation_viewport())
        paintable_rect = self._annotation_paint_rect(img)
        if paintable_rect is None or not paintable_rect.contains(display_pos):
            return None
        iw, ih = img.width(), img.height()
        img_x = (display_pos.x() - dest.x()) * iw / dest.width()
        img_y = (display_pos.y() - dest.y()) * ih / dest.height()
        img_x = min(max(float(img_x), 0.0), float(iw - 1))
        img_y = min(max(float(img_y), 0.0), float(ih - 1))
        return (img_x, img_y)

    def _annotation_target_image(self) -> QImage | None:
        return self._single_image or self._left_image

    def _annotation_viewport(self) -> QRectF | None:
        """Return the left-panel viewport rect when in split mode, else None."""
        if self._split_enabled:
            divider_x = float(self.width()) * self._divider_pos
            return QRectF(0, 0, divider_x, float(self.height()))
        return None

    def _annotation_paint_rect(self, img: QImage) -> QRectF | None:
        rect = self._image_rect(img, viewport=self._annotation_viewport())
        if self._split_enabled:
            divider_x = float(self.width()) * self._divider_pos
            rect = rect.intersected(QRectF(0.0, 0.0, divider_x, float(self.height())))
        if rect.isEmpty():
            return None
        return rect

    def _image_rect(self, img: QImage, viewport: QRectF | None = None) -> QRectF:
        """Calculate the destination rect for an image with zoom/pan.

        Parameters
        ----------
        img : QImage
            The image to compute a destination rect for.
        viewport : QRectF, optional
            Sub-region of the widget to fit the image into (used by split view
            so each panel gets its own fit calculation). Defaults to the full
            widget area.
        """
        iw, ih = img.width(), img.height()
        if viewport is not None:
            vx, vy = viewport.x(), viewport.y()
            vw, vh = viewport.width(), viewport.height()
        else:
            vx, vy = 0.0, 0.0
            vw, vh = self.width(), self.height()

        if iw <= 0 or ih <= 0 or vw <= 0 or vh <= 0:
            return QRectF(vx, vy, max(1.0, vw), max(1.0, vh))

        # Fit to viewport at zoom=1.0
        scale_fit = min(vw / iw, vh / ih)
        display_w = iw * scale_fit * self._zoom
        display_h = ih * scale_fit * self._zoom

        # Center within viewport + pan offset
        x = vx + (vw - display_w) / 2 + self._pan.x()
        y = vy + (vh - display_h) / 2 + self._pan.y()

        return QRectF(x, y, display_w, display_h)

    # ── Mouse Events ──

    def _wipe_distance_to_line(self, pos: QPointF) -> float:
        """Signed perpendicular distance from pos to the wipe line (pixels)."""
        return wipe_distance_to_line(
            pos, self.width(), self.height(),
            self._wipe_angle, self._wipe_offset)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # Wipe mode: check handle and line hit
        if event.button() == Qt.LeftButton and self._wipe_mode:
            drag_type, drag_start, start_offset, start_angle = handle_wipe_press(
                event.position(),
                self.width(), self.height(),
                self._wipe_angle, self._wipe_offset,
                self._DIVIDER_HIT_ZONE,
            )
            if drag_type is not None:
                self._wipe_dragging = drag_type
                self._wipe_drag_start = drag_start
                self._wipe_drag_start_offset = start_offset
                self._wipe_drag_start_angle = start_angle
                return

        if event.button() == Qt.LeftButton and self._split_enabled:
            # Check divider hit
            divider_x = self.width() * self._divider_pos
            if abs(event.position().x() - divider_x) < self._DIVIDER_HIT_ZONE:
                self._dragging_divider = True
                return

        # Eyedropper: left-click starts drag-sampling screen color from input frame
        if (event.button() == Qt.LeftButton
                and self._eyedropper_mode):
            self._eyedropper_sampling = True
            self._eyedropper_samples.clear()
            self._eyedropper_last_img_pos = None
            self._sample_eyedropper_at(event.position())
            return

        # Annotation: Shift+left-click = brush resize
        if (event.button() == Qt.LeftButton
                and self._annotation_mode
                and event.modifiers() & Qt.ShiftModifier):
            self._resizing_brush = True
            self._resize_start_y = event.position().y()
            self._resize_start_radius = self._brush_radius
            self.update()
            return


        # Polygon matte authoring clicks
        if self._annotation_mode in ("poly_garbage", "poly_core"):
            if event.button() == Qt.RightButton:
                if self._poly_points:
                    self._poly_points.pop()
                    self.update()
                return
            if event.button() == Qt.LeftButton:
                img = self._annotation_target_image()
                pos = self._display_to_image(event.position())
                if img is not None and pos is not None and img.width() > 1 and img.height() > 1:
                    iw, ih = float(img.width() - 1), float(img.height() - 1)
                    nx, ny = pos[0] / iw, pos[1] / ih
                    dest = self._image_rect(img, viewport=self._annotation_viewport())
                    if not self._poly_points and self._polygon_model is not None:
                        fidx = max(0, int(self._annotation_stem_idx))
                        for shape in self._polygon_model.shapes:
                            if not getattr(shape, "enabled", True):
                                continue
                            for v_idx, (vx, vy) in enumerate(shape.raw_points_at(fidx)):
                                sx = dest.x() + (vx * iw / img.width()) * dest.width()
                                sy = dest.y() + (vy * ih / img.height()) * dest.height()
                                if ((event.position().x() - sx) ** 2 + (event.position().y() - sy) ** 2) ** 0.5 < 10.0:
                                    self._dragging_poly_vertex = (shape.shape_id, v_idx, vx, vy)
                                    self.update()
                                    return
                    if len(self._poly_points) >= 3:
                        sx = dest.x() + (self._poly_points[0][0] * iw / img.width()) * dest.width()
                        sy = dest.y() + (self._poly_points[0][1] * ih / img.height()) * dest.height()
                        if ((event.position().x() - sx) ** 2 + (event.position().y() - sy) ** 2) ** 0.5 < 12.0:
                            self._finish_polygon()
                            return
                    self._poly_points.append((nx, ny))
                    self.update()
                return

        # Direct Post-Track Mask Paint clicks (paints onto saved SamMaskHint/AlphaHint, not SAM strokes)
        if self._annotation_mode in ("mask_paint_fg", "mask_paint_bg") and event.button() == Qt.LeftButton:
            pos = self._display_to_image(event.position())
            if pos is not None:
                self._drawing = True
                self._mask_paint_points = [pos]
                self.update()
            return

        # Annotation: right-click = erase stroke/point under cursor
        if (event.button() == Qt.RightButton
                and self._annotation_mode
                and self._active_annotation_model is not None):
            pos = self._display_to_image(event.position())
            if pos is not None:
                if self._active_annotation_model.erase_at(self._active_stem_idx, pos[0], pos[1], self._brush_radius):
                    self.update()
                    self.stroke_finished.emit()
                    if self._annotation_sibling is not None:
                        self._annotation_sibling.update()
                    return

        # Annotation: eraser mode left-click
        if (event.button() == Qt.LeftButton
                and self._annotation_mode == "eraser"
                and self._active_annotation_model is not None):
            pos = self._display_to_image(event.position())
            if pos is not None:
                self._drawing = True
                if self._active_annotation_model.erase_at(self._active_stem_idx, pos[0], pos[1], self._brush_radius):
                    self.update()
                    self.stroke_finished.emit()
                    if self._annotation_sibling is not None:
                        self._annotation_sibling.update()
                return

        # Annotation: left-click = start drawing stroke or point prompt
        if (event.button() == Qt.LeftButton
                and self._annotation_mode
                and self._active_annotation_model is not None):
            if getattr(self, "_space_pressed", False):
                self._panning = True
                self._pan_start = event.position()
                self._pan_start_offset = QPointF(self._pan)
                self.setCursor(Qt.ClosedHandCursor)
                return

            pos = self._display_to_image(event.position())
            if pos is not None:
                self._drawing = True
                self._straight_line = False

                is_alt = bool(event.modifiers() & Qt.AltModifier)
                if self._annotation_mode in ("brush", "fg"):
                    eff_type = "bg" if is_alt else "fg"
                elif self._annotation_mode == "point":
                    eff_type = "bg" if is_alt else "fg"
                elif self._annotation_mode == "bg":
                    eff_type = "fg" if is_alt else "bg"
                else:
                    eff_type = self._annotation_mode

                self._active_annotation_model.start_stroke(
                    self._active_stem_idx,
                    pos[0], pos[1],
                    eff_type,
                    self._brush_radius,
                    object_id=self._active_object_id,
                )
                self.update()
                return

        # Middle-click on wipe line: reset to default angle
        if event.button() == Qt.MiddleButton and self._wipe_mode:
            dist = abs(self._wipe_distance_to_line(event.position()))
            if dist < self._DIVIDER_HIT_ZONE:
                self._wipe_angle = -45.0
                self._wipe_offset = 0.0
                self.update()
                return

        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self._pan_start_offset = QPointF(self._pan)
            self.setCursor(Qt.ClosedHandCursor)
            return

        # Left-drag pan fallback: nothing else claimed this left press
        # (annotation, eyedropper, divider, and wipe line all return above).
        # The pan only engages after the drag threshold in mouseMoveEvent,
        # so plain clicks and double-click reset keep working. Some mice
        # have no middle button.
        if event.button() == Qt.LeftButton and not self._annotation_mode:
            self._pan_pressed = True
            self._pan_did_drag = False
            self._pan_start = event.position()
            self._pan_start_offset = QPointF(self._pan)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self._mouse_pos = event.position()
        self._emit_pixel_hover(self._mouse_pos)

        if self._drawing and self._annotation_mode in ("mask_paint_fg", "mask_paint_bg"):
            pos = self._display_to_image(event.position())
            if pos is not None:
                self._mask_paint_points.append(pos)
                self.update()
            return

        if self._dragging_poly_vertex is not None and self._polygon_model is not None:
            img = self._annotation_target_image()
            pos = self._display_to_image(event.position())
            if img is not None and pos is not None and img.width() > 1 and img.height() > 1:
                nx = pos[0] / float(img.width() - 1)
                ny = pos[1] / float(img.height() - 1)
                sid, v_idx, _, _ = self._dragging_poly_vertex
                fidx = max(0, int(self._annotation_stem_idx))
                for shape in self._polygon_model.shapes:
                    if shape.shape_id == sid:
                        shape.move_vertex(fidx, v_idx, nx, ny)
                        self._dragging_poly_vertex = (sid, v_idx, nx, ny)
                        break
                self.update()
                if self._annotation_sibling is not None:
                    self._annotation_sibling.update()
            return

        if self._annotation_mode in ("poly_garbage", "poly_core"):
            self.update()
            return

        # Eyedropper: accumulate samples while dragging
        if self._eyedropper_sampling:
            self._sample_eyedropper_at(event.position())
            return

        # Wipe mode dragging
        if self._wipe_dragging:
            new_offset, new_angle = handle_wipe_drag(
                self._wipe_dragging,
                event.position(),
                self._wipe_drag_start,
                self._wipe_drag_start_offset,
                self._wipe_drag_start_angle,
                self.width(), self.height(),
                self._wipe_angle,
            )
            if self._wipe_dragging == "handle":
                self._wipe_offset = new_offset
            elif self._wipe_dragging == "line":
                self._wipe_angle = new_angle
            self.update()
            return

        # Wipe mode cursor feedback
        if self._wipe_mode and not self._panning:
            cursor = wipe_cursor_for_pos(
                event.position(),
                self.width(), self.height(),
                self._wipe_angle, self._wipe_offset,
                self._DIVIDER_HIT_ZONE,
            )
            if cursor is not None:
                self.setCursor(cursor)
            elif not self._annotation_mode:
                self.unsetCursor()

        if self._dragging_divider:
            w = self.width()
            if w > 0:
                self._divider_pos = max(0.05, min(0.95, event.position().x() / w))
                self.update()
            return

        # Annotation: brush resize (Shift+drag)
        if self._resizing_brush:
            delta_y = self._resize_start_y - event.position().y()
            # 2 image-pixels per display-pixel of drag
            new_radius = self._resize_start_radius + delta_y * 2.0
            self._brush_radius = max(2.0, min(200.0, new_radius))
            self.update()
            return

        # Annotation: straight-line preview (Alt+drag)
        if self._drawing and self._straight_line and self._active_annotation_model is not None:
            pos = self._display_to_image(event.position())
            if pos is not None and self._line_anchor is not None:
                # Replace stroke points with anchor + current endpoint for live preview
                stroke = self._active_annotation_model.current_stroke
                if stroke is not None:
                    stroke.points = [self._line_anchor, pos]
                self.update()
                if self._annotation_sibling is not None:
                    self._annotation_sibling.update()
            return

        # Annotation: eraser dragging
        if self._drawing and self._annotation_mode == "eraser" and self._active_annotation_model is not None:
            pos = self._display_to_image(event.position())
            if pos is not None:
                if self._active_annotation_model.erase_at(self._active_stem_idx, pos[0], pos[1], self._brush_radius):
                    self.update()
                    self.stroke_finished.emit()
                    if self._annotation_sibling is not None:
                        self._annotation_sibling.update()
            return

        # Annotation: point mode only places single point prompts (no dragging snake)
        if self._drawing and self._annotation_mode == "point":
            return

        # Annotation: box mode updates only the two corner endpoints
        if self._drawing and self._annotation_mode == "box" and self._active_annotation_model is not None:
            pos = self._display_to_image(event.position())
            if pos is not None:
                stroke = self._active_annotation_model.current_stroke
                if stroke is not None and stroke.points:
                    stroke.points = [stroke.points[0], pos]
                    self.update()
                    if self._annotation_sibling is not None:
                        self._annotation_sibling.update()
            return

        if self._drawing and self._active_annotation_model is not None:
            pos = self._display_to_image(event.position())
            if pos is not None:
                self._active_annotation_model.add_point(pos[0], pos[1])
                self.update()
                if self._annotation_sibling is not None:
                    self._annotation_sibling.update()
            return

        # Promote a pressed left button to a pan once past the drag threshold
        if self._pan_pressed and not self._panning:
            dist = (event.position() - self._pan_start).manhattanLength()
            if dist >= QApplication.startDragDistance():
                self._panning = True
                self._pan_did_drag = True
                self.setCursor(Qt.ClosedHandCursor)

        if self._panning:
            delta = event.position() - self._pan_start
            self._pan = QPointF(
                self._pan_start_offset.x() + delta.x(),
                self._pan_start_offset.y() + delta.y(),
            )
            self.view_transform_changed.emit(self._zoom, QPointF(self._pan))
            self.update()
            return

        # Cursor feedback for divider hover
        if self._split_enabled:
            divider_x = self.width() * self._divider_pos
            if abs(event.position().x() - divider_x) < self._DIVIDER_HIT_ZONE:
                self.setCursor(Qt.SplitHCursor)
            elif not self._annotation_mode:
                self.unsetCursor()

        # Update brush cursor position when annotation mode active
        if self._annotation_mode:
            self.update()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        # Eyedropper: finalize on release - emit averaged color + full sample list
        if self._eyedropper_sampling:
            self._eyedropper_sampling = False
            self._eyedropper_preview_color = None
            if self._eyedropper_samples:
                n = len(self._eyedropper_samples)
                avg_r = sum(s[0] for s in self._eyedropper_samples) // n
                avg_g = sum(s[1] for s in self._eyedropper_samples) // n
                avg_b = sum(s[2] for s in self._eyedropper_samples) // n
                self.screen_samples_ready.emit(list(self._eyedropper_samples))
                self.color_sampled.emit(avg_r, avg_g, avg_b)
            self._eyedropper_samples.clear()
            self._eyedropper_last_img_pos = None
            self.update()
            return

        if self._wipe_dragging:
            self._wipe_dragging = None
            return

        if self._dragging_divider:
            self._dragging_divider = False
            return

        if self._resizing_brush:
            self._resizing_brush = False
            self.update()
            return

        if self._dragging_poly_vertex is not None:
            sid, v_idx, nx, ny = self._dragging_poly_vertex
            self._dragging_poly_vertex = None
            fidx = max(0, int(self._annotation_stem_idx))
            self.polygon_vertex_moved.emit(sid, fidx, v_idx, nx, ny)
            self.update()
            return

        if self._drawing and self._annotation_mode in ("mask_paint_fg", "mask_paint_bg"):
            self._drawing = False
            pts = list(self._mask_paint_points)
            self._mask_paint_points.clear()
            is_erase = (
                self._annotation_mode == "mask_paint_bg"
                or bool(event.modifiers() & Qt.AltModifier)
            )
            action = "erase" if is_erase else "add"
            if pts:
                self.mask_paint_stroke.emit(
                    int(self._annotation_stem_idx),
                    action,
                    pts,
                    float(self._brush_radius),
                )
            self.update()
            return

        if self._drawing and self._active_annotation_model is not None:
            # Straight-line: finalize with anchor + release point
            if self._straight_line and self._line_anchor is not None:
                pos = self._display_to_image(event.position())
                if pos is not None:
                    stroke = self._active_annotation_model.current_stroke
                    if stroke is not None:
                        stroke.points = [self._line_anchor, pos]
                self._straight_line = False
                self._line_anchor = None
            self._active_annotation_model.finish_stroke()
            self._drawing = False
            self.stroke_finished.emit()
            self.update()
            if self._annotation_sibling is not None:
                self._annotation_sibling.update()
            return

        if self._panning:
            self._panning = False
            self._pan_pressed = False
            if self._annotation_mode:
                self.setCursor(Qt.CrossCursor)
            elif self._eyedropper_mode:
                self.setCursor(_eyedropper_cursor())
            else:
                self.unsetCursor()
            return

        self._pan_pressed = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Double-click to close polygon or reset zoom/pan (disabled during brush annotation)."""
        if event.button() == Qt.LeftButton and self._annotation_mode in ("poly_garbage", "poly_core"):
            if len(self._poly_points) >= 3:
                self._finish_polygon()
            return
        if event.button() == Qt.LeftButton and not self._annotation_mode:
            if self._pan_did_drag:
                # The preceding click was a left-drag pan; a quick follow-up
                # click must not reset the view the user just framed.
                self._pan_did_drag = False
                return
            self.reset_zoom()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Ctrl+Wheel to zoom, plain wheel in wipe mode to slide divider."""
        mods = event.modifiers()
        # Scroll in wipe mode: slide the wipe line (up = A, down = B)
        # Shift+scroll for fine-grain control
        if self._wipe_mode and mods in (Qt.KeyboardModifier(0), Qt.ShiftModifier):
            delta = event.angleDelta().y()
            self._wipe_offset = handle_wipe_scroll(
                delta, bool(mods & Qt.ShiftModifier), self._wipe_offset)
            self.update()
            return
        # macOS: Cmd (MetaModifier) is the primary modifier, not Ctrl
        import sys as _sys
        _zoom_mod = Qt.MetaModifier if _sys.platform == "darwin" else Qt.ControlModifier
        if mods & _zoom_mod:
            delta = event.angleDelta().y()
            factor = 1.1 if delta > 0 else 1.0 / 1.1
            new_zoom = self._zoom * factor
            new_zoom = max(self._zoom_min, min(self._zoom_max, new_zoom))

            # Adjust pan so the point under the cursor stays fixed
            cursor = event.position()
            vw, vh = self.width(), self.height()
            # Point relative to viewport center + current pan
            cx = cursor.x() - vw / 2 - self._pan.x()
            cy = cursor.y() - vh / 2 - self._pan.y()
            # Scale the offset by the zoom ratio
            ratio = new_zoom / self._zoom
            self._pan = QPointF(
                self._pan.x() - cx * (ratio - 1),
                self._pan.y() - cy * (ratio - 1),
            )

            self._zoom = new_zoom
            self.zoom_changed.emit(self._zoom)
            self.view_transform_changed.emit(self._zoom, QPointF(self._pan))
            self.update()
        elif mods & Qt.ShiftModifier:
            # Shift+Wheel: horizontal pan (left/right)
            delta = event.angleDelta().y()
            self._pan = QPointF(self._pan.x() + delta * 0.5, self._pan.y())
            self.view_transform_changed.emit(self._zoom, QPointF(self._pan))
            self.update()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Alt:
            self.update()
            if self._annotation_sibling:
                self._annotation_sibling.update()
        elif key == Qt.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            self.setCursor(Qt.OpenHandCursor)
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self._zoom = min(self._zoom_max, self._zoom * 1.2)
            self.zoom_changed.emit(self._zoom)
            self.view_transform_changed.emit(self._zoom, QPointF(self._pan))
            self.update()
        elif key == Qt.Key_Minus:
            self._zoom = max(self._zoom_min, self._zoom / 1.2)
            self.zoom_changed.emit(self._zoom)
            self.view_transform_changed.emit(self._zoom, QPointF(self._pan))
            self.update()
        elif key == Qt.Key_0:
            self.reset_zoom()
        elif key in (Qt.Key_Return, Qt.Key_Enter) and self._annotation_mode in ("poly_garbage", "poly_core"):
            if len(self._poly_points) >= 3:
                self._finish_polygon()
        elif key == Qt.Key_Escape and self._annotation_mode in ("poly_garbage", "poly_core"):
            if self._poly_points:
                self._poly_points.clear()
                self.update()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Alt:
            self.update()
            if self._annotation_sibling:
                self._annotation_sibling.update()
        elif key == Qt.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            if not self._panning:
                if self._annotation_mode:
                    self.setCursor(Qt.CrossCursor)
                else:
                    self.unsetCursor()
        super().keyReleaseEvent(event)
