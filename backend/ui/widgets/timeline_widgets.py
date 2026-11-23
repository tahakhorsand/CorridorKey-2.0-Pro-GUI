"""Professional NLE studio timeline canvas for CorridorKey 2.0.

Replaces the legacy stacked QSlider + transparent MarkerOverlay + CoverageBar
with a single unified custom-painted DaVinci Resolve / Nuke style timeline widget.
Provides conflict-free hit-testing (top cap prioritizes playhead, track band
gives clean priority to prominent In/Out bracket handles, Shift/middle drag
slides the whole trim range), adaptive frame/timecode ticks, hover preview pill,
and an integrated 3-lane coverage strip (Alpha, Inference, Annotations).
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Sequence

from PySide6.QtCore import Qt, Signal, QPoint, QPointF, QRect, QRectF
from PySide6.QtGui import (
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QMenu, QSlider, QWidget


class _DragMode(Enum):
    NONE = auto()
    SCRUB = auto()
    IN_HANDLE = auto()
    OUT_HANDLE = auto()
    RANGE_SHIFT = auto()


def _format_timecode(frame: int, fps: float) -> str:
    """Format a 0-based frame index as HH:MM:SS:FF at `fps`."""
    safe_fps = max(1, int(round(fps if fps > 0 else 24.0)))
    f = max(0, int(frame))
    ff = f % safe_fps
    total_sec = f // safe_fps
    ss = total_sec % 60
    total_min = total_sec // 60
    mm = total_min % 60
    hh = total_min // 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


class StudioTimelineCanvas(QWidget):
    """Unified NLE-grade custom-painted timeline scrubber, trim bar, and coverage strip."""

    frame_selected = Signal(int)
    scrub_started = Signal()
    scrub_finished = Signal(int)
    in_point_changed = Signal(int)
    out_point_changed = Signal(int)
    range_cleared = Signal()

    # Backwards-compatible signal aliases matching legacy MarkerOverlay expectations
    in_point_dragged = in_point_changed
    out_point_dragged = out_point_changed
    scrub_to_frame = frame_selected

    _MARGIN_X = 12
    _HANDLE_HIT_PX = 12.0
    _PLAYHEAD_HIT_PX = 11.0
    _RULER_H = 14
    _TRACK_TOP = 15
    _TRACK_H = 12
    _COV_TOP = 29
    _COV_LANE_H = 3
    _COV_LANE_GAP = 1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setMinimumWidth(140)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.NoFocus)

        self._total: int = 0
        self._frame: int = 0
        self._fps: float = 24.0

        self._in_point: int | None = None
        self._out_point: int | None = None

        self._alpha_cov: list[bool] = []
        self._infer_cov: list[bool] = []
        self._anno_cov: list[bool] = []
        self._waveform_peaks: list[float] = []

        self._drag_mode: _DragMode = _DragMode.NONE
        self._dragging: bool = False  # kept for compatibility checks
        self._range_anchor_frame: int = 0
        self._range_orig_in: int = 0
        self._range_orig_out: int = 0

        self._hover_pos: QPointF | None = None
        self._hover_target: str | None = None  # "playhead" | "in" | "out" | "range" | "track"
        self._hover_frame: int | None = None
        self._show_timecode_tooltip: bool = False

    # ── State & Data API ──

    def set_range(self, total_frames: int) -> None:
        self._total = max(0, int(total_frames))
        if self._total == 0:
            self._frame = 0
            self._in_point = None
            self._out_point = None
        else:
            self._frame = max(0, min(self._frame, self._total - 1))
            if self._in_point is not None:
                self._in_point = max(0, min(self._in_point, self._total - 1))
            if self._out_point is not None:
                self._out_point = max(0, min(self._out_point, self._total - 1))
            if self._in_point is not None and self._out_point is not None and self._in_point > self._out_point:
                self._in_point, self._out_point = self._out_point, self._in_point
        self.setEnabled(self._total > 0)
        self.update()

    def set_frame(self, frame: int) -> None:
        if self._total <= 0:
            self._frame = 0
        else:
            self._frame = max(0, min(int(frame), self._total - 1))
        self.update()

    def current_frame(self) -> int:
        return self._frame

    def set_fps(self, fps: float) -> None:
        if fps > 0:
            self._fps = float(fps)
            self.update()

    def set_in_out(
        self,
        in_point: int | None,
        out_point: int | None,
        total_frames: int | None = None,
    ) -> None:
        if total_frames is not None and total_frames >= 0:
            self._total = int(total_frames)
        if self._total > 0:
            self._in_point = None if in_point is None else max(0, min(int(in_point), self._total - 1))
            self._out_point = None if out_point is None else max(0, min(int(out_point), self._total - 1))
        else:
            self._in_point = in_point
            self._out_point = out_point
        self.update()

    def get_in_out(self) -> tuple[int | None, int | None]:
        return self._in_point, self._out_point

    def set_coverage(self, alpha: Sequence[bool], inference: Sequence[bool]) -> None:
        self._alpha_cov = [bool(v) for v in alpha]
        self._infer_cov = [bool(v) for v in inference]
        self.update()

    def set_annotation_markers(self, annotated: Sequence[bool]) -> None:
        self._anno_cov = [bool(v) for v in annotated]
        self.update()

    def set_waveform(self, peaks: Sequence[float]) -> None:
        self._waveform_peaks = [max(0.0, min(1.0, float(v))) for v in peaks]
        self.update()

    def set_timecode_mode(self, show_tc: bool) -> None:
        self._show_timecode_tooltip = bool(show_tc)
        self.update()

    # ── Coordinate Mapping & Backwards-Compatible Marker Queries ──

    def _track_span(self) -> tuple[float, float]:
        x0 = float(self._MARGIN_X)
        w = max(1.0, float(self.width() - 2 * self._MARGIN_X))
        return x0, w

    def _frame_to_x(self, frame: int) -> float:
        x0, w = self._track_span()
        if self._total <= 1:
            return x0
        frac = max(0.0, min(1.0, float(frame) / float(self._total - 1)))
        return x0 + frac * w

    def _x_to_frame(self, x: float) -> int:
        if self._total <= 1:
            return 0
        x0, w = self._track_span()
        frac = max(0.0, min(1.0, (float(x) - x0) / w))
        return int(round(frac * float(self._total - 1)))

    def _marker_x(self, which: str) -> float:
        """Return pixel X coordinate for `"in"` or `"out"` handle (test-compatible API)."""
        if which == "in" and self._in_point is not None:
            return self._frame_to_x(self._in_point)
        if which == "out" and self._out_point is not None:
            return self._frame_to_x(self._out_point)
        return -1000.0

    def _hit_marker(self, pos) -> str | None:
        """Return `'in'`, `'out'`, or `None` for horizontal proximity to trim handles."""
        if self._total <= 0:
            return None
        px = float(pos.x()) if hasattr(pos, "x") else float(pos)
        candidates: list[tuple[float, str]] = []
        if self._in_point is not None:
            d_in = abs(px - self._marker_x("in"))
            if d_in <= self._HANDLE_HIT_PX:
                candidates.append((d_in, "in"))
        if self._out_point is not None:
            d_out = abs(px - self._marker_x("out"))
            if d_out <= self._HANDLE_HIT_PX:
                candidates.append((d_out, "out"))
        if not candidates:
            return None
        if (
            len(candidates) == 2
            and self._in_point is not None
            and self._out_point is not None
            and abs(self._marker_x("out") - self._marker_x("in")) <= 2.0
        ):
            # When In and Out overlap, left side selects '[' (in), right side selects ']' (out)
            center = 0.5 * (self._marker_x("in") + self._marker_x("out"))
            return "out" if px > center else "in"
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _classify_hit(self, x: float, y: float, modifiers: Qt.KeyboardModifiers, button: Qt.MouseButton) -> str:
        """Determine hit target with vertical-band separation to prevent grab fights."""
        if self._total <= 0:
            return "none"

        ph_x = self._frame_to_x(self._frame)
        near_ph = abs(x - ph_x) <= self._PLAYHEAD_HIT_PX
        marker_hit = self._hit_marker(x)

        # Band 1: Ruler / Playhead Cap zone -> Playhead ALWAYS wins
        if y <= self._RULER_H + 1 and near_ph:
            return "playhead"

        # Explicit Shift or Middle click between markers shifts entire range
        if (
            (button == Qt.MiddleButton or (modifiers & Qt.ShiftModifier))
            and self._in_point is not None
            and self._out_point is not None
        ):
            in_x = self._marker_x("in")
            out_x = self._marker_x("out")
            if min(in_x, out_x) - 4 <= x <= max(in_x, out_x) + 4:
                return "range"

        # Band 2: Bracket/track body -> check In/Out handles unless center of playhead
        if marker_hit is not None:
            m_x = self._marker_x(marker_hit)
            # If playhead needle isn't right on top, or user grabbed outside ruler cap
            if abs(x - m_x) <= abs(x - ph_x) + 4.0 or y >= self._TRACK_TOP + 2:
                return marker_hit

        if near_ph:
            return "playhead"

        # Trim highlight upper strip allows dragging [in, out] range cleanly without Shift
        if (
            self._in_point is not None
            and self._out_point is not None
            and self._TRACK_TOP <= y <= self._TRACK_TOP + 4
        ):
            in_x = self._marker_x("in")
            out_x = self._marker_x("out")
            if in_x + self._HANDLE_HIT_PX < x < out_x - self._HANDLE_HIT_PX:
                return "range"

        return "track"

    # ── Mouse / Interaction ──

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._total <= 0:
            return

        if event.button() == Qt.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())
            event.accept()
            return

        if event.button() not in (Qt.LeftButton, Qt.MiddleButton):
            super().mousePressEvent(event)
            return

        pos = event.position()
        target = self._classify_hit(pos.x(), pos.y(), event.modifiers(), event.button())
        self._dragging = True

        if target == "in":
            self._drag_mode = _DragMode.IN_HANDLE
            self.setCursor(Qt.SizeHorCursor)
        elif target == "out":
            self._drag_mode = _DragMode.OUT_HANDLE
            self.setCursor(Qt.SizeHorCursor)
        elif target == "range":
            self._drag_mode = _DragMode.RANGE_SHIFT
            self._range_anchor_frame = self._x_to_frame(pos.x())
            self._range_orig_in = int(self._in_point or 0)
            self._range_orig_out = int(self._out_point or max(0, self._total - 1))
            self.setCursor(Qt.ClosedHandCursor)
        else:
            self._drag_mode = _DragMode.SCRUB
            self.scrub_started.emit()
            clicked_frame = self._x_to_frame(pos.x())
            if clicked_frame != self._frame or target == "track":
                self._frame = clicked_frame
                self.frame_selected.emit(clicked_frame)
            self.setCursor(Qt.SizeHorCursor)

        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        self._hover_pos = pos
        if self._total > 0:
            self._hover_frame = self._x_to_frame(pos.x())

        if self._drag_mode == _DragMode.SCRUB and self._total > 0:
            f = self._x_to_frame(pos.x())
            self._hover_frame = f
            if f != self._frame:
                self._frame = f
                self.frame_selected.emit(f)
            self.update()
            event.accept()
            return

        if self._drag_mode == _DragMode.IN_HANDLE and self._total > 0:
            f = self._x_to_frame(pos.x())
            if self._out_point is not None:
                f = min(f, self._out_point)
            self._hover_frame = f
            if f != self._in_point:
                self._in_point = f
                self.in_point_changed.emit(f)
            if f != self._frame:
                self._frame = f
                self.frame_selected.emit(f)
            self.update()
            event.accept()
            return

        if self._drag_mode == _DragMode.OUT_HANDLE and self._total > 0:
            f = self._x_to_frame(pos.x())
            if self._in_point is not None:
                f = max(f, self._in_point)
            self._hover_frame = f
            if f != self._out_point:
                self._out_point = f
                self.out_point_changed.emit(f)
            if f != self._frame:
                self._frame = f
                self.frame_selected.emit(f)
            self.update()
            event.accept()
            return

        if self._drag_mode == _DragMode.RANGE_SHIFT and self._total > 0:
            f = self._x_to_frame(pos.x())
            delta = f - self._range_anchor_frame
            span = self._range_orig_out - self._range_orig_in
            new_in = max(0, min(self._total - 1 - span, self._range_orig_in + delta))
            new_out = min(self._total - 1, new_in + span)
            changed = (new_in != self._in_point) or (new_out != self._out_point)
            if changed:
                self._in_point = new_in
                self._out_point = new_out
                self.in_point_changed.emit(new_in)
                self.out_point_changed.emit(new_out)
            self.update()
            event.accept()
            return

        # Idle hover feedback
        target = self._classify_hit(pos.x(), pos.y(), event.modifiers(), Qt.NoButton)
        self._hover_target = target
        if target in ("playhead", "in", "out"):
            self.setCursor(Qt.SizeHorCursor)
        elif target == "range":
            self.setCursor(Qt.OpenHandCursor)
        elif target == "track":
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_mode != _DragMode.NONE:
            was_scrub = self._drag_mode == _DragMode.SCRUB
            self._drag_mode = _DragMode.NONE
            self._dragging = False
            if was_scrub:
                self.scrub_finished.emit(self._frame)
            pos = event.position()
            target = self._classify_hit(pos.x(), pos.y(), event.modifiers(), Qt.NoButton)
            self._hover_target = target
            self.setCursor(
                Qt.SizeHorCursor if target in ("playhead", "in", "out")
                else Qt.PointingHandCursor if target == "track"
                else Qt.ArrowCursor
            )
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover_pos = None
        self._hover_frame = None
        self._hover_target = None
        if self._drag_mode == _DragMode.NONE:
            self.setCursor(Qt.ArrowCursor)
        self.update()
        super().leaveEvent(event)

    def _show_context_menu(self, global_pos: QPoint) -> None:
        if self._total <= 0:
            return
        menu = QMenu(self)
        act_in = menu.addAction(self.tr("Set In Point at Current Frame ([)"))
        act_out = menu.addAction(self.tr("Set Out Point at Current Frame (])"))
        menu.addSeparator()
        act_clear = menu.addAction(self.tr("Clear In / Out Range"))
        act_clear.setEnabled(self._in_point is not None or self._out_point is not None)

        chosen = menu.exec(global_pos)
        if chosen == act_in:
            new_in = self._frame
            if self._out_point is not None and new_in > self._out_point:
                self._out_point = new_in
                self.out_point_changed.emit(new_in)
            self._in_point = new_in
            self.in_point_changed.emit(new_in)
            self.update()
        elif chosen == act_out:
            new_out = self._frame
            if self._in_point is not None and new_out < self._in_point:
                self._in_point = new_out
                self.in_point_changed.emit(new_out)
            self._out_point = new_out
            self.out_point_changed.emit(new_out)
            self.update()
        elif chosen == act_clear:
            self._in_point = None
            self._out_point = None
            self.range_cleared.emit()
            self.update()

    # ── Custom NLE Painting ──

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())
        x0, track_w = self._track_span()

        # Background tray groove
        painter.fillRect(QRectF(0, 0, w, h), QColor("#0D0D06"))
        track_rect = QRectF(x0, self._TRACK_TOP, track_w, self._TRACK_H)
        painter.setPen(QPen(QColor("#2A2918"), 1.0))
        painter.setBrush(QColor("#16160D"))
        painter.drawRoundedRect(track_rect, 2.0, 2.0)

        if self._total <= 0:
            painter.end()
            return

        # 1. Adaptive ruler ticks + subtle labels
        self._paint_ruler_ticks(painter, x0, track_w)

        # 2. Trim range shading & active bar
        self._paint_trim_region(painter, x0, track_w)

        # 2b. Audio waveform peaks inside track groove
        if self._waveform_peaks:
            n_peaks = len(self._waveform_peaks)
            mid_y = self._TRACK_TOP + self._TRACK_H * 0.5
            max_half = self._TRACK_H * 0.45
            painter.setPen(QPen(QColor(135, 210, 140, 145), 1.0))
            for i, amp in enumerate(self._waveform_peaks):
                if amp <= 0.02:
                    continue
                px = x0 + (float(i) / max(1, n_peaks - 1)) * track_w
                hh = amp * max_half
                painter.drawLine(QPointF(px, mid_y - hh), QPointF(px, mid_y + hh))

        # 3. 3-lane bottom coverage strip
        self._paint_coverage_lanes(painter, x0, track_w)

        # 4. Hover hairline guide (under playhead & handles)
        if self._hover_frame is not None and self._drag_mode == _DragMode.NONE:
            hx = self._frame_to_x(self._hover_frame)
            painter.setPen(QPen(QColor(255, 255, 255, 55), 1.0, Qt.DashLine))
            painter.drawLine(QPointF(hx, self._RULER_H), QPointF(hx, h - 2))

        # 5. In / Out NLE bracket handles
        if self._in_point is not None:
            self._paint_bracket_handle(
                painter,
                self._frame_to_x(self._in_point),
                kind="in",
                highlighted=(self._hover_target == "in" or self._drag_mode == _DragMode.IN_HANDLE),
            )
        if self._out_point is not None:
            self._paint_bracket_handle(
                painter,
                self._frame_to_x(self._out_point),
                kind="out",
                highlighted=(self._hover_target == "out" or self._drag_mode == _DragMode.OUT_HANDLE),
            )

        # 6. Playhead needle + diamond/pentagon cap
        self._paint_playhead(painter, self._frame_to_x(self._frame))

        # 7. Hover frame / timecode badge pill
        if self._hover_frame is not None and self._hover_pos is not None:
            self._paint_hover_pill(painter, w)

        painter.end()

    def _paint_ruler_ticks(self, painter: QPainter, x0: float, track_w: float) -> None:
        if self._total <= 1:
            return
        px_per_frame = track_w / float(max(1, self._total - 1))
        for step in (1, 2, 5, 10, 25, 50, 100, 250, 500):
            if step * px_per_frame >= 46.0:
                major = step
                break
        else:
            major = 1000
        minor = max(1, major // 5)

        font = QFont("Consolas", 8)
        if not font.exactMatch():
            font = QFont(self.font().family(), 8)
        painter.setFont(font)

        for f in range(0, self._total, minor):
            x = self._frame_to_x(f)
            is_major = (f % major == 0)
            if is_major:
                painter.setPen(QPen(QColor("#5C5B44"), 1.0))
                painter.drawLine(QPointF(x, 4.0), QPointF(x, float(self._RULER_H)))
                painter.setPen(QColor("#807F65"))
                lbl = str(f + 1)
                tx = x + 3.0 if x + 28.0 < x0 + track_w else x - 26.0
                painter.drawText(QPointF(tx, 10.5), lbl)
            elif minor * px_per_frame >= 7.0:
                painter.setPen(QPen(QColor("#343324"), 1.0))
                painter.drawLine(QPointF(x, 9.0), QPointF(x, float(self._RULER_H)))

    def _paint_trim_region(self, painter: QPainter, x0: float, track_w: float) -> None:
        if self._in_point is None and self._out_point is None:
            return
        in_x = self._frame_to_x(self._in_point if self._in_point is not None else 0)
        out_x = self._frame_to_x(self._out_point if self._out_point is not None else max(0, self._total - 1))
        lo_x = min(in_x, out_x)
        hi_x = max(in_x, out_x)

        # Dim out-of-range track segments
        if lo_x > x0:
            painter.fillRect(
                QRectF(x0, self._TRACK_TOP, lo_x - x0, self._TRACK_H),
                QColor(0, 0, 0, 145),
            )
        right_edge = x0 + track_w
        if hi_x < right_edge:
            painter.fillRect(
                QRectF(hi_x, self._TRACK_TOP, right_edge - hi_x, self._TRACK_H),
                QColor(0, 0, 0, 145),
            )

        # Highlight active work-area span
        active_rect = QRectF(lo_x, self._TRACK_TOP, max(2.0, hi_x - lo_x), self._TRACK_H)
        fill_alpha = 52 if (self._hover_target == "range" or self._drag_mode == _DragMode.RANGE_SHIFT) else 34
        painter.fillRect(active_rect, QColor(255, 242, 3, fill_alpha))

        top_bar_color = QColor("#FFF203") if self._hover_target == "range" else QColor("#B9AD18")
        painter.fillRect(QRectF(lo_x, self._TRACK_TOP, max(2.0, hi_x - lo_x), 2.5), top_bar_color)

    def _paint_coverage_lanes(self, painter: QPainter, x0: float, track_w: float) -> None:
        n = max(self._total, len(self._alpha_cov), len(self._infer_cov), len(self._anno_cov))
        if n <= 0:
            return
        lanes = (
            (self._alpha_cov, QColor("#2CC350"), QColor("#151C16")),
            (self._infer_cov, QColor("#FFF203"), QColor("#1C1B10")),
            (self._anno_cov, QColor("#00D4FF"), QColor("#101A1E")),
        )
        for lane_idx, (flags, fg_color, bg_color) in enumerate(lanes):
            ly = float(self._COV_TOP + lane_idx * (self._COV_LANE_H + self._COV_LANE_GAP))
            painter.fillRect(QRectF(x0, ly, track_w, float(self._COV_LANE_H)), bg_color)
            if not flags or not any(flags):
                continue
            m = len(flags)
            # Coalesce contiguous runs for ultra-fast painting even on long sequences
            run_start: int | None = None
            for i, val in enumerate(flags):
                if val and run_start is None:
                    run_start = i
                elif not val and run_start is not None:
                    sx = x0 + (run_start / max(1, m - 1 if m > 1 else 1)) * track_w if m > 1 else x0
                    ex = x0 + ((i - 1) / max(1, m - 1)) * track_w if m > 1 else x0 + track_w
                    sw = max(3.0, ex - sx)
                    painter.fillRect(QRectF(sx - 1.0, ly, sw, float(self._COV_LANE_H)), fg_color)
                    run_start = None
            if run_start is not None:
                sx = x0 + (run_start / max(1, m - 1)) * track_w if m > 1 else x0
                ex = x0 + track_w
                sw = max(3.0, ex - sx)
                painter.fillRect(QRectF(sx - 1.0, ly, sw, float(self._COV_LANE_H)), fg_color)

    def _paint_bracket_handle(
        self,
        painter: QPainter,
        x: float,
        *,
        kind: str,
        highlighted: bool,
    ) -> None:
        base_color = QColor("#00D4FF") if kind == "in" else QColor("#FF8C00")
        color = base_color.lighter(125) if highlighted else base_color
        glow = QColor(color)
        glow.setAlpha(75 if highlighted else 35)

        top_y = float(self._TRACK_TOP - 2)
        bot_y = float(self._TRACK_TOP + self._TRACK_H + 3)
        tab_dir = 1.0 if kind == "in" else -1.0

        # Subtle wider grip halo so targets feel responsive
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawRoundedRect(QRectF(x - 4.0, top_y, 8.0, bot_y - top_y), 2.0, 2.0)

        pen = QPen(color, 2.2 if highlighted else 1.8)
        pen.setCapStyle(Qt.SquareCap)
        pen.setJoinStyle(Qt.MiterJoin)
        painter.setPen(pen)
        # Bracket `[` or `]`
        painter.drawLine(QPointF(x, top_y), QPointF(x, bot_y))
        painter.drawLine(QPointF(x, top_y), QPointF(x + tab_dir * 6.5, top_y))
        painter.drawLine(QPointF(x, bot_y), QPointF(x + tab_dir * 6.5, bot_y))

    def _paint_playhead(self, painter: QPainter, x: float) -> None:
        active = self._hover_target == "playhead" or self._drag_mode == _DragMode.SCRUB
        color = QColor("#FFF740") if active else QColor("#FFF203")

        # Subtle vertical needle halo
        halo = QColor(color)
        halo.setAlpha(65 if active else 30)
        painter.setPen(QPen(halo, 6.0))
        painter.drawLine(QPointF(x, 12.0), QPointF(x, float(self.height()) - 2.0))

        # Crisp playhead vertical needle
        painter.setPen(QPen(color, 2.0))
        painter.drawLine(QPointF(x, 11.0), QPointF(x, float(self.height()) - 2.0))

        # Pentagon / diamond playhead handle cap at the top ruler band
        w_half = 6.5 if active else 5.5
        cap = QPolygonF([
            QPointF(x - w_half, 1.5),
            QPointF(x + w_half, 1.5),
            QPointF(x + w_half, 7.5),
            QPointF(x, 13.5),
            QPointF(x - w_half, 7.5),
        ])
        painter.setPen(QPen(QColor("#1A1900"), 1.0))
        painter.setBrush(color)
        painter.drawPolygon(cap)

    def _paint_hover_pill(self, painter: QPainter, w: float) -> None:
        if self._hover_frame is None:
            return
        if self._show_timecode_tooltip:
            text = _format_timecode(self._hover_frame, self._fps)
        else:
            text = f"F {self._hover_frame + 1:04d}"

        font = QFont("Consolas", 8, QFont.Bold)
        if not font.exactMatch():
            font = QFont(self.font().family(), 8, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text) + 10.0
        th = 14.0

        hx = self._frame_to_x(self._hover_frame)
        px = max(2.0, min(w - tw - 2.0, hx - tw * 0.5))
        py = 1.0
        rect = QRectF(px, py, tw, th)

        painter.setPen(QPen(QColor("#555432"), 1.0))
        painter.setBrush(QColor(22, 22, 12, 232))
        painter.drawRoundedRect(rect, 3.0, 3.0)
        painter.setPen(QColor("#FFF203"))
        painter.drawText(rect, Qt.AlignCenter, text)


# ── Backwards-Compatible Thin Helpers (safe if imported externally) ──


class _FatSlider(QSlider):
    """Legacy slider compatibility class."""


class CoverageBar(QWidget):
    """Legacy standalone coverage bar shim."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(12)
        self._alpha: list[bool] = []
        self._inference: list[bool] = []
        self._annotated: list[bool] = []

    def set_coverage(self, alpha: list[bool], inference: list[bool]) -> None:
        self._alpha = list(alpha)
        self._inference = list(inference)
        self.update()

    def set_annotation_markers(self, annotated: list[bool]) -> None:
        self._annotated = list(annotated)
        self.update()


class MarkerOverlay(StudioTimelineCanvas):
    """Legacy MarkerOverlay compatibility alias backed by StudioTimelineCanvas."""


__all__ = [
    "_DragMode",
    "_FatSlider",
    "CoverageBar",
    "MarkerOverlay",
    "StudioTimelineCanvas",
]
