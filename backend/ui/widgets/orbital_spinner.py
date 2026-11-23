"""Battle.net-inspired 3-ring orbital animated spinner, busy overlay, and progress dialog."""
from __future__ import annotations

import math
import time

from PySide6.QtCore import Qt, QPointF, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


def _ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    if t < 0.5:
        return 4.0 * t * t * t
    return 1.0 - ((-2.0 * t + 2.0) ** 3) * 0.5


class OrbitalSpinnerWidget(QWidget):
    """Smooth 60 FPS Blizzard Battle.net-authentic 3-blade tapered orbital swirl."""

    def __init__(self, size: int = 56, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._size = max(16, int(size))
        self.setFixedSize(self._size, self._size)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._spinning = False
        self._elapsed = 0.0
        self._last_t = time.monotonic()

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 FPS
        self._timer.timeout.connect(self._on_tick)

    def set_size(self, px: int) -> None:
        self._size = max(16, int(px))
        self.setFixedSize(self._size, self._size)
        self.update()

    def is_spinning(self) -> bool:
        return self._spinning

    def start(self) -> None:
        if not self._spinning:
            self._spinning = True
            self._last_t = time.monotonic()
            self._timer.start()
        self.update()

    def stop(self) -> None:
        if self._spinning:
            self._spinning = False
            self._timer.stop()
        self.update()

    def hideEvent(self, event) -> None:
        if self._spinning:
            self._timer.stop()
        super().hideEvent(event)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._spinning and not self._timer.isActive():
            self._last_t = time.monotonic()
            self._timer.start()

    def _on_tick(self) -> None:
        now = time.monotonic()
        dt = max(0.008, min(0.05, now - self._last_t))
        self._last_t = now
        self._elapsed += dt
        self.update()

    @staticmethod
    def _build_crescent_path(
        rx: float,
        ry: float,
        orbit_offset: float,
        max_hw: float,
        start_rad: float,
        arc_span_rad: float,
        steps: int = 44,
    ) -> tuple[QPainterPath, QPointF, QPointF]:
        """Build a sharp-tipped tapered Battle.net crescent ribbon along a tilted ellipse."""
        outer_pts: list[QPointF] = []
        inner_pts: list[QPointF] = []
        for i in range(steps + 1):
            t = float(i) / float(steps)
            th = start_rad + t * arc_span_rad
            sx = rx * math.cos(th)
            sy = -orbit_offset + ry * math.sin(th)
            # Tangent vector & outward unit normal
            dx = -rx * math.sin(th)
            dy = ry * math.cos(th)
            mag = max(1e-5, math.hypot(dx, dy))
            nx = dy / mag
            ny = -dx / mag
            # Tapered asymmetric belly (0 at both tips, swells cleanly in the outer curve)
            taper = math.sin(math.pi * t) ** 0.78
            belly = 0.62 + 0.45 * math.sin(math.pi * (t ** 0.82))
            hw = max_hw * taper * belly
            outer_pts.append(QPointF(sx + nx * hw, sy + ny * hw))
            inner_pts.append(QPointF(sx - nx * hw * 0.85, sy - ny * hw * 0.85))

        poly = QPolygonF(outer_pts + list(reversed(inner_pts)))
        path = QPainterPath()
        path.addPolygon(poly)
        path.closeSubpath()
        return path, outer_pts[0], outer_pts[steps // 2]

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = float(self.width())
        h = float(self.height())
        cx = w * 0.5
        cy = h * 0.5
        r = min(w, h) * 0.43

        # Battle.net momentum motion: continuous drift + cubic ease-in-out surge per 1.45s cycle
        if self._spinning:
            period = 1.45
            cycles = self._elapsed / period
            whole = math.floor(cycles)
            frac = cycles - whole
            surge = whole * 240.0 + _ease_in_out_cubic(frac) * 240.0
            drift = self._elapsed * 55.0
            global_rot_deg = (surge + drift) % 360.0
            # Subtle trim-path arc breathing during each momentum surge (235° <-> 285°)
            trim_wave = math.sin(frac * math.tau) * 0.22
        else:
            global_rot_deg = 0.0
            trim_wave = 0.0

        # Subtle localized blue ambient glow only directly behind the vortex (no box/frame)
        glow = QRadialGradient(QPointF(cx, cy), r * 1.18)
        glow.setColorAt(0.0, QColor(0, 174, 255, 42))
        glow.setColorAt(0.55, QColor(0, 130, 230, 16))
        glow.setColorAt(1.0, QColor(0, 110, 210, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(QPointF(cx, cy), r * 1.18, r * 1.18)

        rx = r * 0.86
        ry = r * 0.35
        orbit_offset = r * 0.31
        max_hw = max(1.2, r * 0.115)

        for k in range(3):
            painter.save()
            painter.translate(cx, cy)
            painter.rotate(global_rot_deg + k * 120.0 + 18.0)

            # Sequential staggered phase across the 3 blades for fluid Battle.net energy flow
            blade_phase = (self._elapsed * 1.38 + k * 0.333) % 1.0 if self._spinning else 0.0
            span_deg = 204.0 + (22.0 * math.sin(blade_phase * math.tau) + trim_wave * 18.0 if self._spinning else 0.0)
            start_deg = -188.0 + (12.0 * math.sin(blade_phase * math.tau) if self._spinning else 0.0)
            path, p_start, p_mid = self._build_crescent_path(
                rx=rx,
                ry=ry,
                orbit_offset=orbit_offset,
                max_hw=max_hw,
                start_rad=math.radians(start_deg),
                arc_span_rad=math.radians(span_deg),
                steps=40,
            )

            # Soft outer cyan aura stroke
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(0, 174, 255, 58), max(1.0, max_hw * 0.9)))
            painter.drawPath(path)

            # Main Battle.net tapered blade fill with traveling cyan highlight wave
            hl = 0.5 + 0.5 * math.sin(blade_phase * math.tau) if self._spinning else 0.35
            grad = QLinearGradient(p_start, p_mid)
            grad.setColorAt(0.0, QColor(0, 118, 224, 210))
            grad.setColorAt(0.55, QColor(0, 178, 255, 245))
            grad.setColorAt(1.0, QColor(int(70 + 135 * hl), int(218 + 34 * hl), 255, 255))
            painter.setPen(Qt.NoPen)
            painter.setBrush(grad)
            painter.drawPath(path)

            painter.restore()

        painter.end()


class OrbitalBusyOverlay(QWidget):
    """Floating transparent Battle.net swirl centered over a parent widget (no card/text)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent; border: none;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignCenter)

        self._spinner = OrbitalSpinnerWidget(size=58, parent=self)
        layout.addWidget(self._spinner, 0, Qt.AlignCenter)

        self.adjustSize()
        self.hide()

    def show_busy(self, message: str = "", progress: float | None = None) -> None:
        _ = (message, progress)  # clean swirl-only overlay (no card, text, or bar)
        self.adjustSize()
        self.recenter()
        self._spinner.start()
        self.show()
        self.raise_()

    def hide_busy(self) -> None:
        self._spinner.stop()
        self.hide()

    def recenter(self) -> None:
        p = self.parentWidget()
        if p is None:
            return
        self.adjustSize()
        x = max(0, (p.width() - self.width()) // 2)
        y = max(0, (p.height() - self.height()) // 2)
        self.move(x, y)


class OrbitalProgressDialog(QDialog):
    """Dark Battle.net-themed modal progress dialog (QProgressDialog-compatible API)."""

    canceled = Signal()

    def __init__(
        self,
        label_text: str = "",
        cancel_text: str = "Cancel",
        minimum: int = 0,
        maximum: int = 100,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(380)
        self.setWindowModality(Qt.WindowModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self._was_canceled = False
        self._auto_close = False
        self._auto_reset = False

        self.setStyleSheet(
            "QDialog {"
            "  background-color: #0D1522;"
            "  border: 1px solid #1E4968;"
            "  border-radius: 10px;"
            "}"
            "QLabel {"
            "  color: #E6F4FF;"
            "  font-size: 12px;"
            "  background: transparent;"
            "}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        top_row = QHBoxLayout()
        top_row.setSpacing(16)
        self._spinner = OrbitalSpinnerWidget(size=54, parent=self)
        top_row.addWidget(self._spinner, 0, Qt.AlignVCenter)

        self._label = QLabel(label_text)
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        top_row.addWidget(self._label, 1)
        root.addLayout(top_row)

        self._bar = QProgressBar(self)
        self._bar.setFixedHeight(8)
        self._bar.setTextVisible(False)
        self._bar.setRange(minimum, maximum)
        self._bar.setStyleSheet(
            "QProgressBar {"
            "  background-color: #162233;"
            "  border: 1px solid #244560;"
            "  border-radius: 4px;"
            "}"
            "QProgressBar::chunk {"
            "  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #008CEB, stop:1 #6CE5FF);"
            "  border-radius: 3px;"
            "}"
        )
        root.addWidget(self._bar)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel_btn = QPushButton(cancel_text or "Cancel", self)
        self._cancel_btn.setCursor(Qt.PointingHandCursor)
        self._cancel_btn.setMinimumWidth(88)
        self._cancel_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #162233; color: #CFE8FF; border: 1px solid #2B5373;"
            "  border-radius: 5px; padding: 6px 16px; font-size: 11px; font-weight: 600;"
            "}"
            "QPushButton:hover { border-color: #00AEFF; background-color: #1C2E45; color: #FFFFFF; }"
        )
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self._cancel_btn)
        root.addLayout(btn_row)

    def _on_cancel_clicked(self) -> None:
        self._was_canceled = True
        self.canceled.emit()

    def reject(self) -> None:
        if not self._was_canceled:
            self._was_canceled = True
            self.canceled.emit()
        self._spinner.stop()
        super().reject()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._spinner.start()

    def closeEvent(self, event) -> None:
        self._spinner.stop()
        super().closeEvent(event)

    def setLabelText(self, text: str) -> None:
        self._label.setText(text)

    def setMaximum(self, maximum: int) -> None:
        self._bar.setMaximum(maximum)

    def maximum(self) -> int:
        return self._bar.maximum()

    def setMinimum(self, minimum: int) -> None:
        self._bar.setMinimum(minimum)

    def setRange(self, minimum: int, maximum: int) -> None:
        self._bar.setRange(minimum, maximum)

    def setValue(self, value: int) -> None:
        self._bar.setValue(value)
        if self._auto_close and self._bar.maximum() > self._bar.minimum() and value >= self._bar.maximum():
            self.close()

    def wasCanceled(self) -> bool:
        return self._was_canceled

    def setMinimumDuration(self, ms: int) -> None:
        # API compatibility with QProgressDialog (we render cleanly immediately)
        _ = ms

    def setAutoClose(self, auto_close: bool) -> None:
        self._auto_close = bool(auto_close)

    def setAutoReset(self, auto_reset: bool) -> None:
        self._auto_reset = bool(auto_reset)


__all__ = ["OrbitalSpinnerWidget", "OrbitalBusyOverlay", "OrbitalProgressDialog"]
