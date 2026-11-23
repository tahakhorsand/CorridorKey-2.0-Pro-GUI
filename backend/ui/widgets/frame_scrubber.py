"""Frame scrubber timeline with transport controls and Studio NLE canvas.

Uses `StudioTimelineCanvas` for crisp custom-painted DaVinci Resolve / Nuke style
frame navigation, prominent In/Out trim handles, range sliding, and 3-lane
coverage indicators without overlay click-grab conflicts.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSettings, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from ui.widgets.timeline_widgets import StudioTimelineCanvas, _format_timecode


def _make_timeline_icon(kind: str, color: str = "#D4D4B8") -> QIcon:
    """Render a crisp HiDPI vector icon (32x32 @2x) for timeline transport buttons."""
    pm = QPixmap(32, 32)
    pm.fill(Qt.transparent)
    pm.setDevicePixelRatio(2.0)

    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    col = QColor(color)
    pen = QPen(col, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    if kind == "in":
        # Classic NLE Mark-In '[' bracket with inside anchor dot (matches timeline handle)
        painter.drawPolyline(QPolygonF([
            QPointF(10.5, 2.8), QPointF(5.2, 2.8), QPointF(5.2, 13.2), QPointF(10.5, 13.2)
        ]))
        painter.setBrush(col)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(9.5, 8.0), 1.7, 1.7)
    elif kind == "out":
        # Classic NLE Mark-Out ']' bracket with inside anchor dot (matches timeline handle)
        painter.drawPolyline(QPolygonF([
            QPointF(5.5, 2.8), QPointF(10.8, 2.8), QPointF(10.8, 13.2), QPointF(5.5, 13.2)
        ]))
        painter.setBrush(col)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(6.5, 8.0), 1.7, 1.7)
    elif kind == "pingpong":
        # Opposing arrows: top -> right, bottom <- left
        painter.drawLine(QPointF(3.0, 5.2), QPointF(12.2, 5.2))
        painter.drawPolyline(QPolygonF([QPointF(9.8, 3.0), QPointF(12.8, 5.2), QPointF(9.8, 7.4)]))
        painter.drawLine(QPointF(13.0, 10.8), QPointF(3.8, 10.8))
        painter.drawPolyline(QPolygonF([QPointF(6.2, 8.6), QPointF(3.2, 10.8), QPointF(6.2, 13.0)]))
    elif kind in ("sound_on", "sound_off"):
        # Speaker cone polygon
        painter.setPen(Qt.NoPen)
        painter.setBrush(col)
        painter.drawPolygon(QPolygonF([
            QPointF(2.5, 6.2), QPointF(5.0, 6.2), QPointF(8.2, 3.2),
            QPointF(8.2, 12.8), QPointF(5.0, 9.8), QPointF(2.5, 9.8),
        ]))
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(col, 1.5, Qt.SolidLine, Qt.RoundCap))
        if kind == "sound_on":
            painter.drawArc(QRectF(7.2, 5.0, 4.4, 6.0), -52 * 16, 104 * 16)
            painter.drawArc(QRectF(7.2, 3.0, 7.6, 10.0), -52 * 16, 104 * 16)
        else:
            painter.drawLine(QPointF(10.0, 5.6), QPointF(13.8, 10.4))
            painter.drawLine(QPointF(13.8, 5.6), QPointF(10.0, 10.4))

    painter.end()
    return QIcon(pm)


class FrameScrubber(QWidget):
    """Frame navigation scrubber with transport controls and responsive debounce."""

    frame_changed = Signal(int)      # emitted to viewers (stem index)
    in_point_changed = Signal(int)   # in-point set at stem index
    out_point_changed = Signal(int)  # out-point set at stem index
    range_cleared = Signal()         # in/out range cleared

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 2, 8, 2)
        row.setSpacing(4)

        self._fps: float = 24.0
        self._total: int = 0
        self._current: int = 0
        self._show_timecode: bool = False
        self._suppress_signal: bool = False
        self._in_point: int | None = None
        self._out_point: int | None = None

        # Clickable frame / timecode badge (left)
        self._frame_label = QPushButton("0000 / 0000")
        self._frame_label.setFixedWidth(96)
        self._frame_label.setFixedHeight(24)
        self._frame_label.setCursor(Qt.PointingHandCursor)
        self._frame_label.setToolTip(self.tr("Click to toggle Frame / Timecode display"))
        self._frame_label.setStyleSheet(
            "QPushButton { background: #15150B; color: #D8D8BA; border: 1px solid #2A2918; "
            "border-radius: 3px; font-family: Consolas, monospace; font-size: 11px; padding: 0 4px; }"
            "QPushButton:hover { border-color: #FFF203; color: #FFFFFF; }"
        )
        self._frame_label.clicked.connect(self._toggle_timecode_mode)
        row.addWidget(self._frame_label)

        btn_style = (
            "QPushButton { background: transparent; color: #90907A; border: none; "
            "font-size: 13px; padding: 0 3px; font-family: sans-serif; }"
            "QPushButton:hover { color: #FFF203; }"
            "QPushButton:disabled { color: #3A3A30; }"
        )
        io_btn_style = (
            "QPushButton { background: #18180E; color: #D4D4B8; border: 1px solid #2A2918; "
            "border-radius: 3px; font-family: Consolas, monospace; font-size: 11px; "
            "font-weight: bold; padding: 0px; margin: 0px; }"
            "QPushButton:hover { border-color: #FFF203; color: #FFF203; background: #222214; }"
            "QPushButton:disabled { color: #3A3A30; border-color: #1D1D14; }"
        )

        # Jump to start
        self._start_btn = QPushButton("\u25C0\u25C0")
        self._start_btn.setFixedWidth(26)
        self._start_btn.setStyleSheet(btn_style)
        self._start_btn.setToolTip(self.tr("Go to first frame (Home)"))
        self._start_btn.clicked.connect(self._go_start)
        row.addWidget(self._start_btn)

        # Step back
        self._prev_btn = QPushButton("\u25C0")
        self._prev_btn.setFixedWidth(22)
        self._prev_btn.setStyleSheet(btn_style)
        self._prev_btn.setToolTip(self.tr("Previous frame (Left)"))
        self._prev_btn.clicked.connect(self._step_back)
        row.addWidget(self._prev_btn)

        # Play/Pause toggle
        self._play_btn = QPushButton("\u25B6")
        self._play_btn.setFixedWidth(24)
        self._play_btn.setStyleSheet(btn_style)
        self._play_btn.setToolTip(self.tr("Play / Pause (Space)"))
        self._play_btn.clicked.connect(self.toggle_playback)
        row.addWidget(self._play_btn)

        # Quick Set-In button
        self._set_in_btn = QPushButton("")
        self._set_in_btn.setFixedSize(24, 22)
        self._set_in_btn.setStyleSheet(io_btn_style)
        self._set_in_btn.setIcon(_make_timeline_icon("in", "#38C8FF"))
        self._set_in_btn.setIconSize(QSize(14, 14))
        self._set_in_btn.setToolTip(self.tr("Set In point at current frame (I)"))
        self._set_in_btn.clicked.connect(self.set_in_at_current)
        row.addWidget(self._set_in_btn)

        # Unified NLE timeline canvas (preserve _center and _marker_overlay aliases)
        self._canvas = StudioTimelineCanvas(self)
        self._center = self._canvas
        self._marker_overlay = self._canvas
        self._canvas.setToolTip(
            self.tr(
                "Drag yellow playhead cap to scrub, drag [ / ] handles to set In/Out trim.\n"
                "Shift+drag inside range bar to move window. Bottom lanes: Green=Alpha, Yellow=Output, Cyan=Annotations."
            )
        )
        self._canvas.frame_selected.connect(self._on_canvas_frame_selected)
        self._canvas.scrub_finished.connect(self._on_canvas_scrub_finished)
        self._canvas.in_point_changed.connect(self._on_canvas_in_changed)
        self._canvas.out_point_changed.connect(self._on_canvas_out_changed)
        self._canvas.range_cleared.connect(self._on_canvas_range_cleared)
        row.addWidget(self._canvas, 1)

        # Quick Set-Out button
        self._set_out_btn = QPushButton("")
        self._set_out_btn.setFixedSize(24, 22)
        self._set_out_btn.setStyleSheet(io_btn_style)
        self._set_out_btn.setIcon(_make_timeline_icon("out", "#FF9838"))
        self._set_out_btn.setIconSize(QSize(14, 14))
        self._set_out_btn.setToolTip(self.tr("Set Out point at current frame (O)"))
        self._set_out_btn.clicked.connect(self.set_out_at_current)
        row.addWidget(self._set_out_btn)

        # Step forward
        self._next_btn = QPushButton("\u25B6")
        self._next_btn.setFixedWidth(22)
        self._next_btn.setStyleSheet(btn_style)
        self._next_btn.setToolTip(self.tr("Next frame (Right)"))
        self._next_btn.clicked.connect(self._step_forward)
        row.addWidget(self._next_btn)

        # Jump to end
        self._end_btn = QPushButton("\u25B6\u25B6")
        self._end_btn.setFixedWidth(26)
        self._end_btn.setStyleSheet(btn_style)
        self._end_btn.setToolTip(self.tr("Go to last frame (End)"))
        self._end_btn.clicked.connect(self._go_end)
        row.addWidget(self._end_btn)

        # Playback speed selector (1x -> 0.5x -> 0.25x)
        self._speed_steps: tuple[float, ...] = (1.0, 0.5, 0.25)
        self._speed_idx: int = 0
        self._speed_btn = QPushButton("1x")
        self._speed_btn.setFixedSize(40, 22)
        self._speed_btn.setStyleSheet(
            "QPushButton { background: #18180E; color: #D4D4B8; border: 1px solid #2A2918; "
            "border-radius: 3px; font-family: Consolas, monospace; font-size: 11px; "
            "font-weight: bold; padding: 0px 2px; margin: 0px; }"
            "QPushButton:hover { border-color: #FFF203; color: #FFF203; }"
            "QPushButton:disabled { color: #3A3A30; border-color: #1D1D14; }"
        )
        self._speed_btn.setToolTip(self.tr("Cycle playback speed (1x -> 0.5x -> 0.25x)"))
        self._speed_btn.clicked.connect(self.cycle_playback_speed)
        row.addWidget(self._speed_btn)

        # Ping-pong loop toggle
        self._pingpong: bool = False
        self._play_dir: int = 1
        self._pingpong_btn = QPushButton("")
        self._pingpong_btn.setFixedSize(26, 22)
        self._pingpong_btn.setCheckable(True)
        self._pingpong_btn.setIcon(_make_timeline_icon("pingpong", "#D4D4B8"))
        self._pingpong_btn.setIconSize(QSize(15, 15))
        self._pingpong_btn.setStyleSheet(
            "QPushButton { background: #18180E; color: #D4D4B8; border: 1px solid #2A2918; "
            "border-radius: 3px; padding: 0px; margin: 0px; }"
            "QPushButton:checked { background: #FFF203; color: #000000; border-color: #FFF203; }"
            "QPushButton:hover { border-color: #FFF203; background: #222214; }"
            "QPushButton:disabled { border-color: #1D1D14; }"
        )
        self._pingpong_btn.setToolTip(self.tr("Toggle Ping-Pong playback (bounce forward/backward inside range)"))
        self._pingpong_btn.toggled.connect(self.set_pingpong)
        row.addWidget(self._pingpong_btn)

        # Timeline audio playback + waveform mute toggle
        from backend.audio_preview import TimelineAudioPlayer
        self._audio_player = TimelineAudioPlayer(self)
        self._audio_seq = 0
        self._mute_btn = QPushButton("")
        self._mute_btn.setFixedSize(26, 22)
        self._mute_btn.setCheckable(True)
        self._mute_btn.setIcon(_make_timeline_icon("sound_on", "#D4D4B8"))
        self._mute_btn.setIconSize(QSize(15, 15))
        self._mute_btn.setStyleSheet(
            "QPushButton { background: #18180E; color: #D4D4B8; border: 1px solid #2A2918; "
            "border-radius: 3px; padding: 0px; margin: 0px; }"
            "QPushButton:checked { background: #2A1414; color: #FF7070; border-color: #803030; }"
            "QPushButton:hover { border-color: #FFF203; background: #222214; }"
        )
        self._mute_btn.setToolTip(self.tr("Mute / Unmute timeline audio playback"))
        self._mute_btn.toggled.connect(self._on_audio_mute_toggled)
        row.addWidget(self._mute_btn)

        # Responsive 25ms coalescing timer during continuous scrubbing
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(25)
        self._debounce.timeout.connect(self._emit_frame)

        # Playback timer (default ~24 FPS)
        self._playback_timer = QTimer(self)
        self._playback_timer.setTimerType(Qt.PreciseTimer)
        self._playback_timer.setInterval(41)
        self._playback_timer.timeout.connect(self._playback_tick)
        self._loop_enabled_cached = True
        self._playing = False

        self.set_range(0)

    # ── Range & Navigation API ──

    def set_range(self, total_frames: int) -> None:
        """Configure scrubber for a clip with `total_frames` stems."""
        if self._playing:
            self._stop_playback()
        self._total = max(0, int(total_frames))
        if self._total == 0:
            self._current = 0
            self._in_point = None
            self._out_point = None
        else:
            self._current = max(0, min(self._current, self._total - 1))
        enabled = self._total > 0
        self._canvas.set_range(self._total)
        for btn in (
            self._start_btn,
            self._prev_btn,
            self._play_btn,
            self._set_in_btn,
            self._set_out_btn,
            self._next_btn,
            self._end_btn,
            self._speed_btn,
            self._pingpong_btn,
        ):
            btn.setEnabled(enabled)
        self._update_label()

    def set_frame(self, index: int) -> None:
        """Set current frame externally without re-emitting `frame_changed`."""
        self._suppress_signal = True
        if self._total > 0:
            self._current = max(0, min(int(index), self._total - 1))
        else:
            self._current = 0
        self._canvas.set_frame(self._current)
        self._suppress_signal = False
        self._update_label()

    def current_frame(self) -> int:
        return self._current

    def set_fps(self, fps: float) -> None:
        """Set playback timer interval and timecode base FPS."""
        if fps > 0:
            self._fps = float(fps)
            self._apply_playback_interval()
            self._canvas.set_fps(self._fps)

    def _apply_playback_interval(self) -> None:
        speed = self._speed_steps[self._speed_idx]
        eff_fps = max(1.0, self._fps * speed)
        interval = max(16, int(round(1000.0 / eff_fps)))
        self._playback_timer.setInterval(interval)

    @property
    def playback_speed(self) -> float:
        return self._speed_steps[self._speed_idx]

    def set_playback_speed(self, speed: float) -> None:
        if speed in self._speed_steps:
            self._speed_idx = self._speed_steps.index(speed)
        else:
            self._speed_idx = 0
        val = self._speed_steps[self._speed_idx]
        self._speed_btn.setText(f"{val:g}x")
        self._apply_playback_interval()

    def cycle_playback_speed(self) -> float:
        self._speed_idx = (self._speed_idx + 1) % len(self._speed_steps)
        val = self._speed_steps[self._speed_idx]
        self._speed_btn.setText(f"{val:g}x")
        self._apply_playback_interval()
        return val

    @property
    def is_pingpong(self) -> bool:
        return self._pingpong

    def set_pingpong(self, enabled: bool) -> None:
        self._pingpong = bool(enabled)
        if not self._pingpong:
            self._play_dir = 1
        old = self._pingpong_btn.blockSignals(True)
        try:
            self._pingpong_btn.setChecked(self._pingpong)
        finally:
            self._pingpong_btn.blockSignals(old)
            self._pingpong_btn.setIcon(
                _make_timeline_icon("pingpong", "#000000" if self._pingpong else "#D4D4B8")
            )
            self._update_label()

    def set_coverage(self, alpha: list[bool], inference: list[bool]) -> None:
        self._canvas.set_coverage(alpha, inference)

    def set_annotation_markers(self, annotated: list[bool]) -> None:
        self._canvas.set_annotation_markers(annotated)

    # ── In / Out Markers ──

    def set_in_point(self, index: int | None) -> None:
        self._in_point = index
        if index is not None and self._out_point is not None and index > self._out_point:
            self._out_point = index
            self.out_point_changed.emit(index)
        self._sync_markers()
        if index is not None:
            self.in_point_changed.emit(index)

    def set_out_point(self, index: int | None) -> None:
        self._out_point = index
        if index is not None and self._in_point is not None and index < self._in_point:
            self._in_point = index
            self.in_point_changed.emit(index)
        self._sync_markers()
        if index is not None:
            self.out_point_changed.emit(index)

    def set_in_at_current(self) -> None:
        if self._total > 0:
            self.set_in_point(self._current)

    def set_out_at_current(self) -> None:
        if self._total > 0:
            self.set_out_point(self._current)

    def set_in_out(self, in_point: int | None, out_point: int | None) -> None:
        self._in_point = in_point
        self._out_point = out_point
        self._sync_markers()

    def clear_in_out(self) -> None:
        self._in_point = None
        self._out_point = None
        self._sync_markers()
        self.range_cleared.emit()

    def get_in_out(self) -> tuple[int | None, int | None]:
        return self._in_point, self._out_point

    @property
    def has_range(self) -> bool:
        return self._in_point is not None and self._out_point is not None

    def _sync_markers(self) -> None:
        self._canvas.set_in_out(self._in_point, self._out_point, self._total)

    # ── Canvas Callbacks ──

    def _on_canvas_frame_selected(self, frame: int) -> None:
        self._current = frame
        self._update_label()
        if not self._suppress_signal:
            self._debounce.start()

    def _on_canvas_scrub_finished(self, frame: int) -> None:
        self._current = frame
        self._update_label()
        if self._debounce.isActive():
            self._debounce.stop()
        if not self._suppress_signal:
            self.frame_changed.emit(self._current)

    def _on_canvas_in_changed(self, frame: int) -> None:
        self._in_point = frame
        self.in_point_changed.emit(frame)

    def _on_canvas_out_changed(self, frame: int) -> None:
        self._out_point = frame
        self.out_point_changed.emit(frame)

    def _on_canvas_range_cleared(self) -> None:
        self._in_point = None
        self._out_point = None
        self.range_cleared.emit()

    def _emit_frame(self) -> None:
        self.frame_changed.emit(self._current)

    def _toggle_timecode_mode(self) -> None:
        self._show_timecode = not self._show_timecode
        self._canvas.set_timecode_mode(self._show_timecode)
        self._update_label()

    def _update_label(self) -> None:
        if self._total <= 0:
            self._frame_label.setText("00:00:00:00" if self._show_timecode else "0 / 0")
            return
        if self._show_timecode:
            self._frame_label.setText(_format_timecode(self._current, self._fps))
        else:
            self._frame_label.setText(f"{self._current + 1:04d} / {self._total:04d}")

    # ── Transport & Stepping ──

    def _jump_to(self, index: int, immediate: bool = True) -> None:
        if self._total <= 0:
            return
        clamped = max(0, min(int(index), self._total - 1))
        self._current = clamped
        self._suppress_signal = True
        self._canvas.set_frame(clamped)
        self._suppress_signal = False
        self._update_label()
        if self._debounce.isActive():
            self._debounce.stop()
        if immediate:
            self.frame_changed.emit(clamped)
        else:
            self._debounce.start()

    def _go_start(self) -> None:
        self._jump_to(0)

    def _step_back(self) -> None:
        self._jump_to(max(0, self._current - 1))

    def _step_forward(self) -> None:
        if self._total <= 0:
            return
        self._jump_to(min(self._total - 1, self._current + 1))

    def _go_end(self) -> None:
        if self._total > 0:
            self._jump_to(self._total - 1)

    step_backward = _step_back
    step_forward = _step_forward

    # ── Playback ──

    def toggle_playback(self) -> None:
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def stop_playback(self) -> None:
        if self._playing:
            self._stop_playback()

    def _start_playback(self) -> None:
        if self._total <= 1:
            return
        self._playing = True
        self._loop_enabled_cached = QSettings().value("playback/loop", True, type=bool)
        self._play_btn.setText("\u275A\u275A")
        self._play_btn.setToolTip(self.tr("Pause (Space)"))
        self._playback_timer.start()
        if hasattr(self, "_audio_player"):
            sec = float(self._current) / max(1.0, float(self._fps))
            self._audio_player.play_from(sec)

    def _stop_playback(self) -> None:
        self._playing = False
        self._play_dir = 1
        self._playback_timer.stop()
        if hasattr(self, "_audio_player"):
            self._audio_player.pause()
        self._play_btn.setText("\u25B6")
        self._play_btn.setToolTip(self.tr("Play (Space)"))

    def _on_audio_mute_toggled(self, muted: bool) -> None:
        self._mute_btn.setIcon(
            _make_timeline_icon("sound_off", "#FF6666")
            if muted
            else _make_timeline_icon("sound_on", "#D4D4B8")
        )
        if hasattr(self, "_audio_player"):
            self._audio_player.set_muted(muted)

    def set_audio_source(self, video_path: str | None) -> None:
        """Load audio track for timeline playback and compute waveform peaks in background."""
        self._audio_seq += 1
        cur_seq = self._audio_seq
        self._canvas.set_waveform([])
        if hasattr(self, "_audio_player"):
            self._audio_player.load(video_path)
        if not video_path:
            return
        import threading
        from PySide6.QtCore import QTimer
        from backend.audio_preview import extract_audio_waveform

        def _worker() -> None:
            peaks = extract_audio_waveform(video_path, num_bins=320)
            if self._audio_seq == cur_seq:
                QTimer.singleShot(0, self, lambda: self._apply_waveform_if_current(cur_seq, peaks))

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_waveform_if_current(self, seq: int, peaks: list[float]) -> None:
        if seq == self._audio_seq:
            self._canvas.set_waveform(peaks)

    def _playback_tick(self) -> None:
        loop_enabled = getattr(self, "_loop_enabled_cached", True)
        if self._in_point is not None and self._out_point is not None:
            lo, hi = self._in_point, self._out_point
        else:
            lo, hi = 0, max(0, self._total - 1)

        if self._pingpong:
            next_frame = self._current + self._play_dir
            if next_frame > hi:
                self._play_dir = -1
                next_frame = max(lo, hi - 1)
            elif next_frame < lo:
                self._play_dir = 1
                next_frame = min(hi, lo + 1)
        else:
            next_frame = self._current + 1
            if next_frame > hi:
                if loop_enabled:
                    next_frame = lo
                else:
                    self._stop_playback()
                    return

        self._current = next_frame
        self._suppress_signal = True
        self._canvas.set_frame(next_frame)
        self._suppress_signal = False
        self._update_label()
        self.frame_changed.emit(next_frame)

    def wheelEvent(self, event) -> None:
        if self._total <= 0:
            return
        step = max(1, self._total * 2 // 100)
        delta = event.angleDelta().y()
        if delta > 0:
            self._jump_to(min(self._total - 1, self._current + step), immediate=False)
        elif delta < 0:
            self._jump_to(max(0, self._current - step), immediate=False)
        event.accept()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Space:
            self.toggle_playback()
        elif key == Qt.Key_Left:
            self._step_back()
        elif key == Qt.Key_Right:
            self._step_forward()
        elif key == Qt.Key_Home:
            self._go_start()
        elif key == Qt.Key_End:
            self._go_end()
        elif key == Qt.Key_I:
            self.set_in_at_current()
        elif key == Qt.Key_O:
            self.set_out_at_current()
        else:
            super().keyPressEvent(event)
