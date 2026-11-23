"""Right panel — alpha generation, tracking, and CorridorKey 2.0 native parameters."""

from __future__ import annotations



import json
import sys

from PySide6.QtCore import QEvent, QLocale, QRectF, QSettings, Qt, Signal

from PySide6.QtGui import QColor, QPainter, QPen

from PySide6.QtWidgets import (

    QApplication,

    QCheckBox,

    QComboBox,

    QDoubleSpinBox,

    QFrame,

    QGridLayout,

    QGroupBox,

    QHBoxLayout,

    QInputDialog,

    QLabel,

    QPushButton,

    QScrollArea,

    QSizePolicy,

    QSlider,

    QSpinBox,

    QVBoxLayout,

    QWidget,

)



from backend import InferenceParams, OutputConfig





def _stretch_ignore(widget: QWidget) -> QWidget:

    widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    return widget





def _no_scroll_wheel(widget: QWidget) -> QWidget:

    widget.setFocusPolicy(Qt.ClickFocus)

    widget.installEventFilter(_WheelGuard.instance())

    return widget





class _WheelGuard(QWidget):

    """Event filter: ignores wheel on unfocused widgets, enforces single-step on focused sliders."""



    _singleton = None



    @classmethod

    def instance(cls):

        if cls._singleton is None:

            cls._singleton = cls()

        return cls._singleton

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            if not obj.hasFocus():
                event.ignore()
                return True
            if isinstance(obj, QSlider):
                delta = event.angleDelta().y()
                if delta > 0:
                    obj.setValue(obj.value() + obj.singleStep())
                elif delta < 0:
                    obj.setValue(obj.value() - obj.singleStep())
                return True
        return False


class _PillSwitch(QPushButton):
    """High-contrast animated-style capsule ON/OFF toggle switch."""

    def __init__(self, checked: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(bool(checked))
        self.setFixedSize(42, 20)
        self.setCursor(Qt.PointingHandCursor)
        self.setFlat(True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setStyleSheet("QPushButton { background: transparent; border: none; padding: 0px; margin: 0px; }")

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())
        on = self.isChecked()

        track_rect = QRectF(1.0, 1.0, w - 2.0, h - 2.0)
        radius = (h - 2.0) * 0.5
        if on:
            track_col = QColor("#10B975")
            border_col = QColor("#45F5AC")
            knob_col = QColor("#FFFFFF")
            knob_x = w - h + 3.0
        else:
            track_col = QColor("#1D222B")
            border_col = QColor("#485263")
            knob_col = QColor("#9AA3B0")
            knob_x = 3.0

        painter.setPen(QPen(border_col, 1.2))
        painter.setBrush(track_col)
        painter.drawRoundedRect(track_rect, radius, radius)

        knob_d = h - 6.0
        painter.setPen(Qt.NoPen)
        painter.setBrush(knob_col)
        painter.drawEllipse(QRectF(knob_x, 3.0, knob_d, knob_d))
        painter.end()


class _ClickableHeader(QWidget):
    """Clickable header bar that toggles its parent Pro section on click."""

    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class _ProToggleSection(QFrame):
    """Modern collapsible Pro Engine card with explicit ON/OFF badge and pill switch."""

    toggled = Signal(bool)

    def __init__(self, title: str, *, checked: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("proToggleSection")
        self._checked = bool(checked)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 7, 8, 8)
        root.setSpacing(6)

        self._header = _ClickableHeader(self)
        self._header.setCursor(Qt.PointingHandCursor)
        self._header.setToolTip(self.tr("Click or flip the switch to Enable / Disable this engine"))
        hdr_row = QHBoxLayout(self._header)
        hdr_row.setContentsMargins(2, 1, 2, 1)
        hdr_row.setSpacing(6)

        self._title_lbl = QLabel(title)
        self._title_lbl.setTextFormat(Qt.PlainText)
        self._title_lbl.setWordWrap(False)
        self._title_lbl.setStyleSheet("background: transparent; border: none;")
        hdr_row.addWidget(self._title_lbl, 1)

        self._state_badge = QLabel("OFF")
        self._state_badge.setAlignment(Qt.AlignCenter)
        self._state_badge.setFixedSize(34, 18)
        hdr_row.addWidget(self._state_badge, 0, Qt.AlignVCenter)

        self._switch = _PillSwitch(checked=self._checked, parent=self._header)
        hdr_row.addWidget(self._switch, 0, Qt.AlignVCenter)

        root.addWidget(self._header)

        self._body = QWidget(self)
        self._body.setStyleSheet("background: transparent;")
        self.body_layout = QGridLayout(self._body)
        self.body_layout.setContentsMargins(2, 4, 2, 2)
        self.body_layout.setHorizontalSpacing(6)
        self.body_layout.setVerticalSpacing(5)
        self.body_layout.setColumnStretch(1, 1)
        root.addWidget(self._body)

        self._header.clicked.connect(lambda: self._switch.setChecked(not self._switch.isChecked()))
        self._switch.toggled.connect(self._on_switch_toggled)
        self._apply_visual_state()

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        val = bool(checked)
        if val == self._checked and self._switch.isChecked() == val:
            self._apply_visual_state()
            return
        self._checked = val
        old = self._switch.blockSignals(True)
        try:
            self._switch.setChecked(val)
        finally:
            self._switch.blockSignals(old)
        self._apply_visual_state()
        self.toggled.emit(val)

    def _on_switch_toggled(self, val: bool) -> None:
        self._checked = bool(val)
        self._apply_visual_state()
        self.toggled.emit(self._checked)

    def _apply_visual_state(self) -> None:
        on = self._checked
        self._body.setVisible(on)
        if on:
            self.setStyleSheet(
                "QFrame#proToggleSection {"
                "  background-color: #101B18;"
                "  border: 1px solid #1E7351;"
                "  border-radius: 6px;"
                "}"
            )
            self._title_lbl.setStyleSheet(
                "color: #34F0A3; font-size: 10px; font-weight: 700; "
                "background: transparent; border: none;"
            )
            self._state_badge.setText("ON")
            self._state_badge.setStyleSheet(
                "color: #071911; background-color: #2CE89C; border: 1px solid #45F5AC; "
                "border-radius: 4px; font-size: 9px; font-weight: 800;"
            )
        else:
            self.setStyleSheet(
                "QFrame#proToggleSection {"
                "  background-color: #12151B;"
                "  border: 1px solid #2A303C;"
                "  border-radius: 6px;"
                "}"
                "QFrame#proToggleSection:hover {"
                "  border-color: #465166;"
                "}"
            )
            self._title_lbl.setStyleSheet(
                "color: #9AA3B2; font-size: 10px; font-weight: 700; "
                "background: transparent; border: none;"
            )
            self._state_badge.setText("OFF")
            self._state_badge.setStyleSheet(
                "color: #7D8696; background-color: #1A1F28; border: 1px solid #303746; "
                "border-radius: 4px; font-size: 9px; font-weight: 700;"
            )





class ParameterPanel(QWidget):

    """Right panel with all CorridorKey 2.0 native parameters."""



    params_changed = Signal()

    parallel_frames_changed = Signal(int)

    screen_color_changed = Signal(str)

    birefnet_requested = Signal(str)
    birefnet_one_frame_requested = Signal(str)

    track_masks_requested = Signal()
    sam3_one_frame_requested = Signal()
    clear_mask_requested = Signal()

    import_alpha_requested = Signal()

    chroma_key_requested = Signal(dict)

    chroma_key_preview = Signal()

    eyedropper_requested = Signal(bool)

    active_object_changed = Signal(int)
    poly_garbage_requested = Signal()
    poly_core_requested = Signal()
    poly_invert_changed = Signal(bool)
    poly_feather_changed = Signal(int)
    poly_curve_changed = Signal(bool)
    poly_clear_requested = Signal()
    mask_paint_fg_requested = Signal()
    mask_paint_bg_requested = Signal()
    mask_interp_range_requested = Signal()
    lut_changed = Signal(bool, str)
    auto_clean_plate_requested = Signal()



    def __init__(self, parent=None):

        super().__init__(parent)

        self.setObjectName("paramPanel")

        self.setMinimumWidth(260)



        self._suppress_signals = False



        outer = QVBoxLayout(self)

        outer.setContentsMargins(0, 0, 0, 0)

        outer.setSpacing(0)



        scroll = QScrollArea()

        scroll.setObjectName("paramPanelScroll")

        scroll.setWidgetResizable(True)

        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll.setFrameShape(QScrollArea.NoFrame)



        inner = QWidget()

        inner.setObjectName("paramPanelInner")

        layout = QVBoxLayout(inner)

        layout.setContentsMargins(10, 8, 10, 8)

        layout.setSpacing(6)

        self._copied_params_dict: dict | None = None
        from backend.presets import PresetManager
        self._preset_mgr = PresetManager()

        # ── Section 0: PRESETS & COPY/PASTE SETTINGS ──
        preset_group = QGroupBox(self.tr("PRESETS & WORKFLOW"))
        preset_v = QVBoxLayout(preset_group)
        preset_v.setSpacing(4)
        p_row1 = QHBoxLayout()
        p_row1.setSpacing(4)
        self._preset_combo = _no_scroll_wheel(QComboBox())
        self._preset_combo.setFocusPolicy(Qt.ClickFocus)
        for pname in self._preset_mgr.list_names():
            self._preset_combo.addItem(pname)
        p_row1.addWidget(self._preset_combo, 1)
        self._apply_preset_btn = QPushButton(self.tr("Apply"))
        self._apply_preset_btn.setFixedWidth(48)
        self._apply_preset_btn.clicked.connect(self._on_apply_preset)
        p_row1.addWidget(self._apply_preset_btn)
        preset_v.addLayout(p_row1)

        p_row2 = QHBoxLayout()
        p_row2.setSpacing(4)
        self._save_preset_btn = QPushButton(self.tr("Save Preset"))
        self._save_preset_btn.clicked.connect(self._on_save_preset)
        p_row2.addWidget(self._save_preset_btn, 1)
        self._copy_params_btn = QPushButton(self.tr("Copy"))
        self._copy_params_btn.clicked.connect(self._on_copy_params)
        p_row2.addWidget(self._copy_params_btn)
        self._paste_params_btn = QPushButton(self.tr("Paste"))
        self._paste_params_btn.clicked.connect(self._on_paste_params)
        p_row2.addWidget(self._paste_params_btn)
        preset_v.addLayout(p_row2)
        layout.addWidget(preset_group)

        # ── Section 1: ALPHA GENERATION & GUIDANCE ──

        alpha_group = QGroupBox(self.tr("ALPHA GENERATION & GUIDANCE"))

        alpha_layout = QVBoxLayout(alpha_group)

        alpha_layout.setSpacing(6)



        # 1. Native Rust Chroma Key

        self._chroma_key_btn = QPushButton(self.tr("RUST CHROMA KEY (ck2_native)"))

        self._chroma_key_btn.setEnabled(False)

        self._chroma_key_btn.setCheckable(True)

        self._chroma_key_btn.setToolTip(

            self.tr("Accelerated parallel chroma keying in Rust Rayon (`ck2_native`). Hotkey: `")

        )

        self._chroma_key_btn.toggled.connect(self._on_chroma_key_toggled)

        alpha_layout.addWidget(self._chroma_key_btn)



        # Collapsible Chroma Key Subpanel

        self._chroma_params_widget = QWidget()

        ck_layout = QVBoxLayout(self._chroma_params_widget)

        ck_layout.setContentsMargins(0, 4, 0, 4)

        ck_layout.setSpacing(4)



        eyedropper_row = QHBoxLayout()

        eyedropper_row.setSpacing(4)

        self._eyedropper_btn = QPushButton(self.tr("💧 Pick Screen Color"))

        self._eyedropper_btn.setCheckable(True)

        self._eyedropper_btn.setToolTip(self.tr("Sample screen color from viewports. Hotkey: C"))

        self._eyedropper_btn.toggled.connect(self._on_eyedropper_toggled)

        eyedropper_row.addWidget(self._eyedropper_btn, 1)



        self._color_swatch = QLabel("")

        self._color_swatch.setFixedSize(30, 24)

        self._color_swatch.setStyleSheet("background: #00B140; border: 1px solid #374151; border-radius: 3px;")

        eyedropper_row.addWidget(self._color_swatch)

        ck_layout.addLayout(eyedropper_row)



        self._ck_strength_label = QLabel(self.tr("Key Strength: 1.0"))

        ck_layout.addWidget(self._ck_strength_label)

        self._ck_strength = _no_scroll_wheel(QSlider(Qt.Horizontal))

        self._ck_strength.setRange(1, 100)

        self._ck_strength.setValue(10)

        self._ck_strength.valueChanged.connect(

            lambda v: self._ck_strength_label.setText(self.tr("Key Strength: %s") % f"{v / 10:.1f}")

        )

        self._ck_strength.valueChanged.connect(self._emit_chroma_preview)

        ck_layout.addWidget(self._ck_strength)



        self._ck_clip_black_label = QLabel(self.tr("Clip Black: 0.08"))

        ck_layout.addWidget(self._ck_clip_black_label)

        self._ck_clip_black = _no_scroll_wheel(QSlider(Qt.Horizontal))

        self._ck_clip_black.setRange(0, 100)

        self._ck_clip_black.setValue(8)

        self._ck_clip_black.valueChanged.connect(

            lambda v: self._ck_clip_black_label.setText(self.tr("Clip Black: %s") % f"{v / 100:.2f}")

        )

        self._ck_clip_black.valueChanged.connect(self._emit_chroma_preview)

        ck_layout.addWidget(self._ck_clip_black)



        self._ck_clip_white_label = QLabel(self.tr("Clip White: 0.92"))

        ck_layout.addWidget(self._ck_clip_white_label)

        self._ck_clip_white = _no_scroll_wheel(QSlider(Qt.Horizontal))

        self._ck_clip_white.setRange(0, 100)

        self._ck_clip_white.setValue(92)

        self._ck_clip_white.valueChanged.connect(

            lambda v: self._ck_clip_white_label.setText(self.tr("Clip White: %s") % f"{v / 100:.2f}")

        )

        self._ck_clip_white.valueChanged.connect(self._emit_chroma_preview)

        ck_layout.addWidget(self._ck_clip_white)



        self._ck_generate_btn = QPushButton(self.tr("GENERATE CHROMA MATTE"))

        self._ck_generate_btn.clicked.connect(self._on_chroma_key_generate)

        ck_layout.addWidget(self._ck_generate_btn)



        self._chroma_params_widget.hide()

        alpha_layout.addWidget(self._chroma_params_widget)

        self._ck_screen_color = None

        self._ck_screen_samples = []



        # 2. BiRefNet Local Cache
        birefnet_model_row = QHBoxLayout()
        birefnet_model_row.setSpacing(4)
        birefnet_lbl = QLabel(self.tr("BiRefNet Model:"))
        birefnet_lbl.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        birefnet_model_row.addWidget(birefnet_lbl)

        self._birefnet_model = _no_scroll_wheel(QComboBox())
        self._birefnet_model.setFocusPolicy(Qt.ClickFocus)
        self._birefnet_has_installed = False
        self._birefnet_clip_enabled = False
        birefnet_model_row.addWidget(self._birefnet_model, 1)
        alpha_layout.addLayout(birefnet_model_row)

        birefnet_btn_row = QHBoxLayout()
        birefnet_btn_row.setSpacing(4)

        self._birefnet_one_frame_btn = QPushButton(self.tr("ONE FRAME HINT"))
        self._birefnet_one_frame_btn.setEnabled(False)
        self._birefnet_one_frame_btn.setFocusPolicy(Qt.ClickFocus)
        self._birefnet_one_frame_btn.setToolTip(self.tr("Generate BiRefNet alpha matte on the active frame only as a single-frame guide."))
        self._birefnet_one_frame_btn.clicked.connect(self._on_birefnet_one_frame_clicked)
        birefnet_btn_row.addWidget(self._birefnet_one_frame_btn, 1)

        self._birefnet_btn = QPushButton(self.tr("AUTO ALPHA (ALL)"))
        self._birefnet_btn.setEnabled(False)
        self._birefnet_btn.setFocusPolicy(Qt.ClickFocus)
        self._birefnet_btn.setToolTip(self.tr("Runs offline BiRefNet across all video frames from local HuggingFace cache."))
        self._birefnet_btn.clicked.connect(self._on_birefnet_clicked)
        birefnet_btn_row.addWidget(self._birefnet_btn, 1)

        alpha_layout.addLayout(birefnet_btn_row)
        self.refresh_birefnet_models()

        # 3. SAM 3.1 Guided Tracking
        sam3_btn_row = QHBoxLayout()
        sam3_btn_row.setSpacing(4)

        self._sam3_one_frame_btn = QPushButton(self.tr("ONE FRAME HINT"))
        self._sam3_one_frame_btn.setEnabled(False)
        self._sam3_one_frame_btn.setFocusPolicy(Qt.ClickFocus)
        self._sam3_one_frame_btn.setToolTip(
            self.tr("Generate SAM 3.1 mask on active frame prompts only as a single-frame guide.")
        )
        self._sam3_one_frame_btn.clicked.connect(self.sam3_one_frame_requested.emit)
        sam3_btn_row.addWidget(self._sam3_one_frame_btn, 1)

        self._track_masks_btn = QPushButton(self.tr("TRACK MASK (SAM 3.1)"))
        self._track_masks_btn.setEnabled(False)
        self._track_masks_btn.setFocusPolicy(Qt.ClickFocus)
        self._track_masks_btn.setToolTip(
            self.tr("Track point prompts (1=green FG, 2=red BG, H=cyan Hair & Soft Details) or box (R) with native SAM 3.1 across all frames.")
        )
        self._track_masks_btn.clicked.connect(self.track_masks_requested.emit)
        sam3_btn_row.addWidget(self._track_masks_btn, 1)

        alpha_layout.addLayout(sam3_btn_row)

        # Subject ID & Multi-Object + Mask Fix / Polygon Matte Tools
        subj_row = QHBoxLayout()
        subj_row.setSpacing(4)
        subj_row.addWidget(QLabel(self.tr("Subject:")))
        self._subject_id_combo = _no_scroll_wheel(QComboBox())
        self._subject_id_combo.setFocusPolicy(Qt.ClickFocus)
        for sid in (1, 2, 3, 4):
            self._subject_id_combo.addItem(f"Subject #{sid}", sid)
        self._subject_id_combo.setToolTip(self.tr("Active SAM 3.1 Subject ID (1-4) for multi-person/object tracking."))
        self._subject_id_combo.currentIndexChanged.connect(
            lambda: self.active_object_changed.emit(int(self._subject_id_combo.currentData() or 1))
        )
        subj_row.addWidget(self._subject_id_combo, 1)
        alpha_layout.addLayout(subj_row)

        mfix_row = QHBoxLayout()
        mfix_row.setSpacing(4)
        self._mask_paint_fg_btn = QPushButton(self.tr("Paint +"))
        self._mask_paint_fg_btn.setToolTip(self.tr("Directly paint white onto tracked SamMaskHint / AlphaHint on current frame."))
        self._mask_paint_fg_btn.clicked.connect(self.mask_paint_fg_requested.emit)
        mfix_row.addWidget(self._mask_paint_fg_btn, 1)

        self._mask_paint_bg_btn = QPushButton(self.tr("Erase -"))
        self._mask_paint_bg_btn.setToolTip(self.tr("Directly erase parts of saved SamMaskHint / AlphaHint on current frame."))
        self._mask_paint_bg_btn.clicked.connect(self.mask_paint_bg_requested.emit)
        mfix_row.addWidget(self._mask_paint_bg_btn, 1)

        self._mask_interp_btn = QPushButton(self.tr("Interp [I..O]"))
        self._mask_interp_btn.setToolTip(self.tr("Linearly interpolate masks between [In] and [Out] frames."))
        self._mask_interp_btn.clicked.connect(self.mask_interp_range_requested.emit)
        mfix_row.addWidget(self._mask_interp_btn, 1)
        alpha_layout.addLayout(mfix_row)

        poly_row = QHBoxLayout()
        poly_row.setSpacing(4)
        self._poly_garbage_btn = QPushButton(self.tr("Poly Garbage"))
        self._poly_garbage_btn.setToolTip(self.tr("Draw polygon garbage matte (Double-click/Enter to close)."))
        self._poly_garbage_btn.clicked.connect(self.poly_garbage_requested.emit)
        poly_row.addWidget(self._poly_garbage_btn, 1)

        self._poly_core_btn = QPushButton(self.tr("Poly Core"))
        self._poly_core_btn.setToolTip(self.tr("Draw polygon core holdout matte (Double-click/Enter to close)."))
        self._poly_core_btn.clicked.connect(self.poly_core_requested.emit)
        poly_row.addWidget(self._poly_core_btn, 1)

        self._poly_invert_check = QCheckBox(self.tr("Inv"))
        self._poly_invert_check.setToolTip(self.tr("Invert the most recent polygon matte."))
        self._poly_invert_check.toggled.connect(self.poly_invert_changed.emit)
        poly_row.addWidget(self._poly_invert_check)

        self._poly_curve_check = QCheckBox(self.tr("Curve"))
        self._poly_curve_check.setToolTip(self.tr("Smooth closed spline curve through polygon vertices."))
        self._poly_curve_check.toggled.connect(self.poly_curve_changed.emit)
        poly_row.addWidget(self._poly_curve_check)

        self._poly_feather_spin = _no_scroll_wheel(QSpinBox())
        self._poly_feather_spin.setRange(0, 60)
        self._poly_feather_spin.setValue(4)
        self._poly_feather_spin.setSuffix("px")
        self._poly_feather_spin.setFixedWidth(54)
        self._poly_feather_spin.setToolTip(self.tr("Feather radius (px) for polygon mattes."))
        self._poly_feather_spin.valueChanged.connect(self.poly_feather_changed.emit)
        poly_row.addWidget(self._poly_feather_spin)

        self._poly_clear_btn = QPushButton(self.tr("Clr"))
        self._poly_clear_btn.setFixedWidth(30)
        self._poly_clear_btn.setToolTip(self.tr("Clear all polygon mattes for this clip."))
        self._poly_clear_btn.clicked.connect(self.poly_clear_requested.emit)
        poly_row.addWidget(self._poly_clear_btn)
        alpha_layout.addLayout(poly_row)



        self._annotation_info = QLabel("")

        self._annotation_info.setStyleSheet("color: #9CA3AF; font-size: 10px;")

        alpha_layout.addWidget(self._annotation_info)



        # 4. External Alpha Import

        self._import_alpha_btn = QPushButton(self.tr("IMPORT ALPHA"))

        self._import_alpha_btn.setEnabled(False)

        self._import_alpha_btn.setToolTip(self.tr("Import external PNG/EXR sequence or Alpha video."))

        self._import_alpha_btn.clicked.connect(self.import_alpha_requested.emit)

        alpha_layout.addWidget(self._import_alpha_btn)

        # 5. Clear Mask & Hints
        self._clear_mask_btn = QPushButton(self.tr("CLEAR MASK / HINT"))
        self._clear_mask_btn.setObjectName("clearMaskBtn")
        self._clear_mask_btn.setEnabled(False)
        self._clear_mask_btn.setToolTip(
            self.tr("Delete all generated masks, alpha hints, and brush annotations for this clip to start over.")
        )
        self._clear_mask_btn.clicked.connect(self.clear_mask_requested.emit)
        alpha_layout.addWidget(self._clear_mask_btn)



        layout.addWidget(alpha_group)



        # ── Section 2: CORRIDORKY 2.0 NATIVE INFERENCE PARAMETERS ──

        ck2_group = QGroupBox(self.tr("CORRIDORKEY 2.0 INFERENCE"))

        ck2_layout = QVBoxLayout(ck2_group)

        ck2_layout.setSpacing(6)



        ck2_grid = QGridLayout()

        ck2_grid.setHorizontalSpacing(4)

        ck2_grid.setVerticalSpacing(4)



        # Background Screen Color

        ck2_grid.addWidget(QLabel(self.tr("Screen Color")), 0, 0)

        self._bg_color = _no_scroll_wheel(QComboBox())

        self._bg_color.addItems([self.tr("Auto"), self.tr("Green"), self.tr("Blue")])

        self._bg_color.currentIndexChanged.connect(self._on_bg_color_changed)

        ck2_grid.addWidget(self._bg_color, 0, 1)



        # Model Lanczos Scale

        ck2_grid.addWidget(QLabel(self.tr("Model Scale")), 1, 0)

        self._model_scale = _no_scroll_wheel(QDoubleSpinBox())

        self._model_scale.setRange(0.25, 2.0)

        self._model_scale.setSingleStep(0.25)

        self._model_scale.setValue(1.0)

        self._model_scale.setToolTip(self.tr("Model Lanczos input scale factor (1.0 = native resolution)."))

        self._model_scale.valueChanged.connect(self._emit_changed)

        ck2_grid.addWidget(self._model_scale, 1, 1)



        # VRAM Mode / Low VRAM

        ck2_grid.addWidget(QLabel(self.tr("VRAM Mode")), 2, 0)

        self._vram_mode = _no_scroll_wheel(QComboBox())

        self._vram_mode.addItems([self.tr("Auto VRAM"), self.tr("Low VRAM"), self.tr("Custom")])

        self._vram_mode.setToolTip(self.tr("Auto VRAM automatically computes window sizes from free GPU memory."))

        self._vram_mode.currentIndexChanged.connect(self._emit_changed)

        ck2_grid.addWidget(self._vram_mode, 2, 1)



        # Hann Window Chunk & Stride

        ck2_grid.addWidget(QLabel(self.tr("Hann Chunk")), 3, 0)

        self._hann_chunk = _no_scroll_wheel(QSpinBox())

        self._hann_chunk.setRange(8, 240)

        self._hann_chunk.setValue(80)

        self._hann_chunk.setToolTip(self.tr("Frames per stage-1 temporal Hann window."))

        self._hann_chunk.valueChanged.connect(self._emit_changed)

        ck2_grid.addWidget(self._hann_chunk, 3, 1)



        ck2_grid.addWidget(QLabel(self.tr("Hann Stride")), 4, 0)

        self._hann_stride = _no_scroll_wheel(QSpinBox())

        self._hann_stride.setRange(4, 120)

        self._hann_stride.setValue(40)

        self._hann_stride.setToolTip(self.tr("Frame hop stride between stage-1 windows."))

        self._hann_stride.valueChanged.connect(self._emit_changed)

        ck2_grid.addWidget(self._hann_stride, 4, 1)



        # Stage 2 Batch

        ck2_grid.addWidget(QLabel(self.tr("Stage 2 Batch")), 5, 0)

        self._stage2_batch = _no_scroll_wheel(QSpinBox())

        self._stage2_batch.setRange(1, 16)

        self._stage2_batch.setValue(1)

        self._stage2_batch.setToolTip(self.tr("Stage 2 full-res refinement batch size."))

        self._stage2_batch.valueChanged.connect(self._emit_changed)

        ck2_grid.addWidget(self._stage2_batch, 5, 1)



        ck2_layout.addLayout(ck2_grid)

        # Use Prior Hints (Alpha / Mask) Checkbox
        self._use_hint = QCheckBox(self.tr("Use Prior Hints (Alpha / Mask)"))
        self._use_hint.setChecked(True)
        self._use_hint.setToolTip(
            self.tr("When enabled, CorridorKey guides inference with prior SAM 3.1, BiRefNet, Chroma Key, or imported hints.\n"
                    "When unchecked, prior hints are ignored and CorridorKey keys the video standalone on its own.")
        )
        self._use_hint.stateChanged.connect(self._on_use_hint_toggled)
        ck2_layout.addWidget(self._use_hint)

        # Hint Mask System Selector (BiRefNet vs SAM 3.1 vs Auto)
        hint_sys_row = QHBoxLayout()
        hint_sys_row.setContentsMargins(0, 2, 0, 2)
        hint_sys_row.setSpacing(6)
        self._hint_system_label = QLabel(self.tr("Hint System:"))
        self._hint_system_label.setStyleSheet("color: #C0C0A0; font-size: 11px;")
        hint_sys_row.addWidget(self._hint_system_label)

        self._hint_source = _no_scroll_wheel(QComboBox())
        self._hint_source.setFocusPolicy(Qt.ClickFocus)
        self._hint_source.addItem(self.tr("Auto (BiRefNet / SAM)"), "auto")
        self._hint_source.addItem(self.tr("BiRefNet (AI Matte)"), "birefnet")
        self._hint_source.addItem(self.tr("SAM 3.1 (Tracked Masks)"), "sam")
        self._hint_source.setToolTip(
            self.tr("Choose which mask generation system to use as the hint for CorridorKey:\n"
                    "- Auto: Automatically uses available tracked SAM 3.1 masks or BiRefNet alpha\n"
                    "- BiRefNet: Strictly uses BiRefNet AI matte hints\n"
                    "- SAM 3.1: Strictly uses SAM 3.1 interactive tracking masks")
        )
        self._hint_source.currentIndexChanged.connect(self._emit_changed)
        hint_sys_row.addWidget(self._hint_source, 1)
        ck2_layout.addLayout(hint_sys_row)

        # Hint Mode Selector (Auto vs One Frame Hint Only vs Full Sequence Track)
        hint_mode_row = QHBoxLayout()
        hint_mode_row.setContentsMargins(0, 2, 0, 2)
        hint_mode_row.setSpacing(6)
        self._hint_mode_label = QLabel(self.tr("Hint Mode:"))
        self._hint_mode_label.setStyleSheet("color: #C0C0A0; font-size: 11px;")
        hint_mode_row.addWidget(self._hint_mode_label)

        self._hint_mode = _no_scroll_wheel(QComboBox())
        self._hint_mode.setFocusPolicy(Qt.ClickFocus)
        self._hint_mode.addItem(self.tr("Auto (Match Available)"), "auto")
        self._hint_mode.addItem(self.tr("One Frame Hint Only"), "one_frame")
        self._hint_mode.addItem(self.tr("Full Sequence Track"), "full_sequence")
        self._hint_mode.setToolTip(
            self.tr("Choose how hints guide CorridorKey 2.0:\n"
                    "- Auto: Automatically uses One Frame Hint if single frame exists, or full sequence if tracked\n"
                    "- One Frame Hint Only: Enforces using only the chosen single-frame guide without tracking full video\n"
                    "- Full Sequence Track: Uses tracked masks across all frames")
        )
        self._hint_mode.currentIndexChanged.connect(self._emit_changed)
        hint_mode_row.addWidget(self._hint_mode, 1)
        ck2_layout.addLayout(hint_mode_row)

        # Hint Quality Sliders
        self._hint_quality_label = QLabel(self.tr("Hint Quality: 0.95"))
        ck2_layout.addWidget(self._hint_quality_label)

        self._hint_quality = _no_scroll_wheel(QSlider(Qt.Horizontal))
        self._hint_quality.setRange(20, 100)
        self._hint_quality.setValue(95)
        self._hint_quality.setToolTip(self.tr("Trust factor for alpha hints (0.20 - 1.00). Lower values treat masks as a soft guide without forcing errors."))
        self._hint_quality.valueChanged.connect(
            lambda v: self._hint_quality_label.setText(self.tr("Hint Quality: %s") % f"{v / 100:.2f}")
        )
        self._hint_quality.valueChanged.connect(self._emit_changed)
        ck2_layout.addWidget(self._hint_quality)



        # Carry Hint & Carry Quality

        carry_row = QHBoxLayout()

        carry_row.setSpacing(4)

        self._carry_hint = QCheckBox(self.tr("Carry Hint Across Windows"))

        self._carry_hint.setChecked(True)

        self._carry_hint.setToolTip(self.tr("Propagate blended carry hints across consecutive temporal windows."))

        self._carry_hint.stateChanged.connect(self._on_carry_hint_toggled)

        carry_row.addWidget(self._carry_hint, 1)



        self._carry_hint_quality = _no_scroll_wheel(QDoubleSpinBox())

        self._carry_hint_quality.setRange(0.5, 1.0)

        self._carry_hint_quality.setSingleStep(0.05)

        self._carry_hint_quality.setValue(0.95)

        self._carry_hint_quality.setToolTip(self.tr("Confidence weight applied to carried temporal hints."))

        self._carry_hint_quality.valueChanged.connect(self._emit_changed)

        carry_row.addWidget(self._carry_hint_quality)

        ck2_layout.addLayout(carry_row)



        # Despill Toggle + Strength Slider
        despill_hdr = QHBoxLayout()
        despill_hdr.setSpacing(6)
        self._despill_enabled_cb = QCheckBox(self.tr("Despill"))
        self._despill_enabled_cb.setChecked(True)
        self._despill_enabled_cb.setToolTip(
            self.tr("Enable or disable green/blue screen spill removal (preserves slider strength when off).")
        )
        despill_hdr.addWidget(self._despill_enabled_cb)

        self._despill_label = QLabel(self.tr("Strength: 0.50"))
        despill_hdr.addWidget(self._despill_label)
        despill_hdr.addStretch()
        ck2_layout.addLayout(despill_hdr)

        self._despill_slider = _no_scroll_wheel(QSlider(Qt.Horizontal))

        self._despill_slider.setRange(0, 100)

        self._despill_slider.setValue(50)

        self._despill_slider.setToolTip(self.tr("Native CorridorKey 2.0 despill algorithm strength (0.0 - 1.0)."))

        self._despill_slider.valueChanged.connect(

            lambda v: self._despill_label.setText(self.tr("Strength: %s") % f"{v / 100:.2f}")

        )

        self._despill_slider.valueChanged.connect(self._emit_changed)

        def _on_despill_toggled(checked: bool) -> None:
            self._despill_slider.setEnabled(bool(checked))
            self._despill_label.setEnabled(bool(checked))
            self._emit_changed()

        self._despill_enabled_cb.toggled.connect(_on_despill_toggled)

        ck2_layout.addWidget(self._despill_slider)



        # Linear Color Adjustments

        color_grid = QGridLayout()

        color_grid.setHorizontalSpacing(4)

        color_grid.setVerticalSpacing(4)



        color_grid.addWidget(QLabel(self.tr("Linear Brightness")), 0, 0)

        self._linear_brightness = _no_scroll_wheel(QDoubleSpinBox())

        self._linear_brightness.setRange(0.1, 5.0)

        self._linear_brightness.setSingleStep(0.1)

        self._linear_brightness.setValue(1.0)

        self._linear_brightness.valueChanged.connect(self._emit_changed)

        color_grid.addWidget(self._linear_brightness, 0, 1)



        color_grid.addWidget(QLabel(self.tr("Linear Contrast")), 1, 0)

        self._linear_contrast = _no_scroll_wheel(QDoubleSpinBox())

        self._linear_contrast.setRange(0.1, 5.0)

        self._linear_contrast.setSingleStep(0.1)

        self._linear_contrast.setValue(1.0)

        self._linear_contrast.valueChanged.connect(self._emit_changed)

        color_grid.addWidget(self._linear_contrast, 1, 1)



        color_grid.addWidget(QLabel(self.tr("Contrast Pivot")), 2, 0)

        self._linear_contrast_pivot = _no_scroll_wheel(QDoubleSpinBox())

        self._linear_contrast_pivot.setRange(0.01, 1.0)

        self._linear_contrast_pivot.setSingleStep(0.02)

        self._linear_contrast_pivot.setValue(0.18)

        self._linear_contrast_pivot.valueChanged.connect(self._emit_changed)

        color_grid.addWidget(self._linear_contrast_pivot, 2, 1)



        ck2_layout.addLayout(color_grid)



        # Scene Linear Checkboxes

        linear_row = QHBoxLayout()

        linear_row.setSpacing(4)

        self._frame_dir_linear = QCheckBox(self.tr("Scene-Linear Input"))

        self._frame_dir_linear.setChecked(False)

        self._frame_dir_linear.stateChanged.connect(self._emit_changed)

        linear_row.addWidget(self._frame_dir_linear)



        self._cutout_linear = QCheckBox(self.tr("Linear Cutout"))

        self._cutout_linear.setChecked(False)

        self._cutout_linear.stateChanged.connect(self._emit_changed)

        linear_row.addWidget(self._cutout_linear)

        ck2_layout.addLayout(linear_row)



        # Despeckle & Garbage Matte

        cleanup_grid = QGridLayout()

        cleanup_grid.setHorizontalSpacing(4)

        cleanup_grid.setVerticalSpacing(4)



        self._despeckle_check = QCheckBox(self.tr("Despeckle"))

        self._despeckle_check.setChecked(True)

        self._despeckle_check.stateChanged.connect(self._emit_changed)

        cleanup_grid.addWidget(self._despeckle_check, 0, 0)



        self._despeckle_size = _no_scroll_wheel(QSpinBox())

        self._despeckle_size.setRange(0, 999999)

        self._despeckle_size.setValue(400)

        self._despeckle_size.setSuffix("px")

        self._despeckle_size.valueChanged.connect(self._emit_changed)

        cleanup_grid.addWidget(self._despeckle_size, 0, 1)



        self._garbage_matte_check = QCheckBox(self.tr("Garbage Matte"))

        self._garbage_matte_check.setChecked(False)

        self._garbage_matte_check.stateChanged.connect(self._on_garbage_matte_toggled)

        cleanup_grid.addWidget(self._garbage_matte_check, 1, 0)



        self._garbage_matte_px = _no_scroll_wheel(QSpinBox())

        self._garbage_matte_px.setRange(1, 500)

        self._garbage_matte_px.setValue(20)

        self._garbage_matte_px.setSuffix("px")

        self._garbage_matte_px.setEnabled(False)

        self._garbage_matte_px.valueChanged.connect(self._emit_changed)

        cleanup_grid.addWidget(self._garbage_matte_px, 1, 1)



        cleanup_grid.addWidget(QLabel(self.tr("Edge Choke / Grow")), 2, 0)

        self._edge_erode_px = _no_scroll_wheel(QSpinBox())

        self._edge_erode_px.setRange(-10, 10)

        self._edge_erode_px.setValue(0)

        self._edge_erode_px.setSuffix("px")

        self._edge_erode_px.setToolTip(self.tr("Negative shrinks matte edges (choke), positive expands matte edges (grow)."))

        self._edge_erode_px.valueChanged.connect(self._emit_changed)

        cleanup_grid.addWidget(self._edge_erode_px, 2, 1)



        cleanup_grid.addWidget(QLabel(self.tr("Edge Feather / Blur")), 3, 0)

        self._edge_blur_px = _no_scroll_wheel(QSpinBox())

        self._edge_blur_px.setRange(0, 20)

        self._edge_blur_px.setValue(0)

        self._edge_blur_px.setSuffix("px")

        self._edge_blur_px.setToolTip(self.tr("Gaussian feather radius applied to alpha edges during post-processing."))

        self._edge_blur_px.valueChanged.connect(self._emit_changed)

        cleanup_grid.addWidget(self._edge_blur_px, 3, 1)

        cleanup_grid.addWidget(QLabel(self.tr("Alpha Black / White")), 4, 0)
        lvl_row = QHBoxLayout()
        lvl_row.setSpacing(4)
        self._alpha_clip_black = _no_scroll_wheel(QDoubleSpinBox())
        self._alpha_clip_black.setRange(0.0, 0.95)
        self._alpha_clip_black.setSingleStep(0.01)
        self._alpha_clip_black.setValue(0.0)
        self._alpha_clip_black.setToolTip(self.tr("Crush semi-transparent background noise below this alpha value."))
        self._alpha_clip_black.valueChanged.connect(self._emit_changed)
        lvl_row.addWidget(self._alpha_clip_black, 1)

        self._alpha_clip_white = _no_scroll_wheel(QDoubleSpinBox())
        self._alpha_clip_white.setRange(0.05, 1.0)
        self._alpha_clip_white.setSingleStep(0.01)
        self._alpha_clip_white.setValue(1.0)
        self._alpha_clip_white.setToolTip(self.tr("Clamp core foreground alpha above this value to solid 1.0."))
        self._alpha_clip_white.valueChanged.connect(self._emit_changed)
        lvl_row.addWidget(self._alpha_clip_white, 1)
        cleanup_grid.addLayout(lvl_row, 4, 1)

        cleanup_grid.addWidget(QLabel(self.tr("Alpha Gamma")), 5, 0)
        self._alpha_gamma = _no_scroll_wheel(QDoubleSpinBox())
        self._alpha_gamma.setRange(0.20, 3.00)
        self._alpha_gamma.setSingleStep(0.05)
        self._alpha_gamma.setValue(1.0)
        self._alpha_gamma.setToolTip(self.tr("Adjust midtone edge density ( > 1 lifts hair wisps, < 1 tightens )."))
        self._alpha_gamma.valueChanged.connect(self._emit_changed)
        cleanup_grid.addWidget(self._alpha_gamma, 5, 1)

        ck2_layout.addLayout(cleanup_grid)

        roi_row = QHBoxLayout()
        roi_row.setSpacing(4)
        self._auto_roi_crop = QCheckBox(self.tr("Auto-ROI Crop (Faster Live Inference)"))
        self._auto_roi_crop.setChecked(False)
        self._auto_roi_crop.setToolTip(self.tr("Automatically crop tightly around the subject hint during single-frame live inference."))
        self._auto_roi_crop.stateChanged.connect(self._emit_changed)
        roi_row.addWidget(self._auto_roi_crop, 1)
        self._auto_roi_pad_px = _no_scroll_wheel(QSpinBox())
        self._auto_roi_pad_px.setRange(16, 512)
        self._auto_roi_pad_px.setSingleStep(16)
        self._auto_roi_pad_px.setValue(96)
        self._auto_roi_pad_px.setSuffix("px")
        self._auto_roi_pad_px.setFixedWidth(64)
        self._auto_roi_pad_px.valueChanged.connect(self._emit_changed)
        roi_row.addWidget(self._auto_roi_pad_px)
        ck2_layout.addLayout(roi_row)



        self._live_preview = QCheckBox(self.tr("Live Render (Current Frame, In-Memory)"))

        self._live_preview.setChecked(False)

        self._live_preview.setToolTip(
            self.tr(
                "Preview changes live on the current frame in memory without saving to disk."
            )
        )

        ck2_layout.addWidget(self._live_preview)



        layout.addWidget(ck2_group)

        # ── Optional Toggleable Pro Engines (ALL disabled by default so existing projects behave identically) ──

        # A. Color Management (Log / Viewer LUT)
        self._color_mgmt_group = _ProToggleSection(self.tr("COLOR MANAGEMENT (LOG/LUT)"), checked=False)
        self._color_mgmt_group.toggled.connect(self._on_lut_controls_changed)
        self._color_mgmt_group.toggled.connect(self._emit_changed)
        cm_grid = self._color_mgmt_group.body_layout
        cm_grid.addWidget(QLabel(self.tr("Input Log")), 0, 0)
        self._log_profile_combo = _no_scroll_wheel(QComboBox())
        for llabel, lval in (
            ("None (Standard sRGB/Rec.709)", "none"),
            ("Sony S-Log3", "slog3"),
            ("ARRI LogC3", "logc3"),
            ("RED Log3G10", "log3g10"),
            ("ACEScg", "acescg"),
        ):
            self._log_profile_combo.addItem(llabel, lval)
        self._log_profile_combo.currentIndexChanged.connect(self._emit_changed)
        cm_grid.addWidget(self._log_profile_combo, 0, 1)

        self._display_lut_check = QCheckBox(self.tr("Viewer .cube LUT"))
        self._display_lut_check.setChecked(False)
        self._display_lut_check.toggled.connect(self._on_lut_controls_changed)
        self._display_lut_check.toggled.connect(self._emit_changed)
        cm_grid.addWidget(self._display_lut_check, 1, 0)

        lut_row = QHBoxLayout()
        lut_row.setSpacing(4)
        self._display_lut_path = ""
        self._lut_path_label = QLabel(self.tr("None"))
        self._lut_path_label.setStyleSheet("color: #808070; font-size: 10px;")
        lut_row.addWidget(self._lut_path_label, 1)
        self._lut_browse_btn = QPushButton(self.tr(".cube..."))
        self._lut_browse_btn.setFixedWidth(56)
        self._lut_browse_btn.clicked.connect(self._on_browse_lut)
        lut_row.addWidget(self._lut_browse_btn)
        cm_grid.addLayout(lut_row, 1, 1)
        layout.addWidget(self._color_mgmt_group)

        # B. Advanced Despill, Edge Extend & Light Wrap
        self._adv_despill_group = _ProToggleSection(self.tr("ADVANCED DESPILL & EDGE"), checked=False)
        self._adv_despill_group.toggled.connect(self._emit_changed)
        ad_grid = self._adv_despill_group.body_layout
        ad_grid.addWidget(QLabel(self.tr("Mode")), 0, 0)
        self._despill_mode = _no_scroll_wheel(QComboBox())
        for mlabel, mval in (("Average", "average"), ("Max", "max"), ("Red/Blue Bias", "red_blue")):
            self._despill_mode.addItem(mlabel, mval)
        self._despill_mode.currentIndexChanged.connect(self._emit_changed)
        ad_grid.addWidget(self._despill_mode, 0, 1)

        ad_grid.addWidget(QLabel(self.tr("Luma Restore")), 1, 0)
        self._despill_luma_restore = _no_scroll_wheel(QDoubleSpinBox())
        self._despill_luma_restore.setRange(0.0, 1.0)
        self._despill_luma_restore.setSingleStep(0.1)
        self._despill_luma_restore.setValue(0.0)
        self._despill_luma_restore.valueChanged.connect(self._emit_changed)
        ad_grid.addWidget(self._despill_luma_restore, 1, 1)

        ad_grid.addWidget(QLabel(self.tr("Edge Extend")), 2, 0)
        self._edge_extend_px = _no_scroll_wheel(QSpinBox())
        self._edge_extend_px.setRange(0, 32)
        self._edge_extend_px.setValue(0)
        self._edge_extend_px.setSuffix("px")
        self._edge_extend_px.valueChanged.connect(self._emit_changed)
        ad_grid.addWidget(self._edge_extend_px, 2, 1)

        ad_grid.addWidget(QLabel(self.tr("Light Wrap")), 3, 0)
        lw_row = QHBoxLayout()
        lw_row.setSpacing(4)
        self._light_wrap_strength = _no_scroll_wheel(QDoubleSpinBox())
        self._light_wrap_strength.setRange(0.0, 1.0)
        self._light_wrap_strength.setSingleStep(0.05)
        self._light_wrap_strength.setValue(0.0)
        self._light_wrap_strength.valueChanged.connect(self._emit_changed)
        lw_row.addWidget(self._light_wrap_strength, 1)
        self._light_wrap_radius = _no_scroll_wheel(QSpinBox())
        self._light_wrap_radius.setRange(2, 64)
        self._light_wrap_radius.setValue(12)
        self._light_wrap_radius.setSuffix("px")
        self._light_wrap_radius.valueChanged.connect(self._emit_changed)
        lw_row.addWidget(self._light_wrap_radius)
        ad_grid.addLayout(lw_row, 3, 1)
        layout.addWidget(self._adv_despill_group)

        # C. Temporal Alpha Deflicker
        self._deflicker_group = _ProToggleSection(self.tr("TEMPORAL ALPHA DEFLICKER"), checked=False)
        self._deflicker_group.toggled.connect(self._emit_changed)
        df_grid = self._deflicker_group.body_layout
        df_grid.addWidget(QLabel(self.tr("Strength")), 0, 0)
        df_row = QHBoxLayout()
        df_row.setSpacing(4)
        self._deflicker_strength = _no_scroll_wheel(QDoubleSpinBox())
        self._deflicker_strength.setRange(0.0, 1.0)
        self._deflicker_strength.setSingleStep(0.05)
        self._deflicker_strength.setValue(0.35)
        self._deflicker_strength.valueChanged.connect(self._emit_changed)
        df_row.addWidget(self._deflicker_strength, 1)
        self._deflicker_radius = _no_scroll_wheel(QSpinBox())
        self._deflicker_radius.setRange(1, 3)
        self._deflicker_radius.setValue(1)
        self._deflicker_radius.setSuffix("f")
        self._deflicker_radius.valueChanged.connect(self._emit_changed)
        df_row.addWidget(self._deflicker_radius)
        df_grid.addLayout(df_row, 0, 1)
        layout.addWidget(self._deflicker_group)

        # D. Clean Plate (IBK Difference)
        self._clean_plate_group = _ProToggleSection(self.tr("CLEAN PLATE (IBK)"), checked=False)
        self._clean_plate_group.toggled.connect(self._emit_changed)
        cp_grid = self._clean_plate_group.body_layout
        cp_row = QHBoxLayout()
        cp_row.setSpacing(4)
        self._clean_plate_path = ""
        self._clean_plate_label = QLabel(self.tr("None"))
        self._clean_plate_label.setStyleSheet("color: #808070; font-size: 10px;")
        cp_row.addWidget(self._clean_plate_label, 1)
        self._cp_browse_btn = QPushButton(self.tr("Pick..."))
        self._cp_browse_btn.clicked.connect(self._on_browse_clean_plate)
        cp_row.addWidget(self._cp_browse_btn)
        self._cp_auto_btn = QPushButton(self.tr("Auto-Gen"))
        self._cp_auto_btn.setToolTip(self.tr("Synthesize clean plate from current frame & mask."))
        self._cp_auto_btn.clicked.connect(self.auto_clean_plate_requested.emit)
        cp_row.addWidget(self._cp_auto_btn)
        cp_grid.addLayout(cp_row, 0, 0, 1, 2)
        cp_grid.addWidget(QLabel(self.tr("Diff Blend")), 1, 0)
        self._clean_plate_strength = _no_scroll_wheel(QDoubleSpinBox())
        self._clean_plate_strength.setRange(0.0, 1.0)
        self._clean_plate_strength.setSingleStep(0.05)
        self._clean_plate_strength.setValue(0.5)
        self._clean_plate_strength.valueChanged.connect(self._emit_changed)
        cp_grid.addWidget(self._clean_plate_strength, 1, 1)
        layout.addWidget(self._clean_plate_group)

        # ── Section 3: OUTPUT ENCODING & EXPORT ──

        out_group = QGroupBox(self.tr("OUTPUT & EXPORT"))

        out_layout = QVBoxLayout(out_group)

        out_layout.setSpacing(6)



        out_grid = QGridLayout()

        out_grid.setHorizontalSpacing(4)

        out_grid.setVerticalSpacing(4)



        out_grid.addWidget(QLabel(self.tr("Format")), 0, 0)

        self._output_format = _no_scroll_wheel(QComboBox())

        self._output_format.addItem(self.tr("Video (MP4/H.264)"), "h264")
        self._output_format.addItem(self.tr("OpenEXR Sequences"), "exr")
        self._output_format.addItem(self.tr("ProRes 4444 (.mov + Alpha)"), "prores_4444")
        self._output_format.addItem(self.tr("ProRes 422 HQ (.mov)"), "prores_422hq")
        self._output_format.addItem(self.tr("PNG Sequence (RGBA)"), "png_seq")

        self._output_format.currentIndexChanged.connect(self._emit_changed)

        out_grid.addWidget(self._output_format, 0, 1)



        out_grid.addWidget(QLabel(self.tr("EXR Precision")), 1, 0)

        self._exr_dtype = _no_scroll_wheel(QComboBox())

        self._exr_dtype.addItems([self.tr("Half (16-bit Float)"), self.tr("Float (32-bit Float)")])

        self._exr_dtype.currentIndexChanged.connect(self._emit_changed)

        out_grid.addWidget(self._exr_dtype, 1, 1)



        out_grid.addWidget(QLabel(self.tr("H.264 CRF / Preset")), 2, 0)

        enc_row = QHBoxLayout()

        enc_row.setSpacing(2)

        self._crf = _no_scroll_wheel(QSpinBox())

        self._crf.setRange(0, 51)

        self._crf.setValue(12)

        self._crf.valueChanged.connect(self._emit_changed)

        enc_row.addWidget(self._crf)

        self._preset = _no_scroll_wheel(QComboBox())

        self._preset.addItems(["fast", "medium", "slow"])

        self._preset.setCurrentText("medium")

        self._preset.currentIndexChanged.connect(self._emit_changed)

        enc_row.addWidget(self._preset)

        out_grid.addLayout(enc_row, 2, 1)



        out_layout.addLayout(out_grid)



        # Channel Toggles

        channels_row = QHBoxLayout()

        channels_row.setSpacing(4)

        self._out_alpha = QCheckBox(self.tr("Alpha"))

        self._out_alpha.setChecked(True)

        self._out_alpha.stateChanged.connect(self._emit_changed)

        channels_row.addWidget(self._out_alpha)

        self._out_fg = QCheckBox(self.tr("FG"))

        self._out_fg.setChecked(True)

        self._out_fg.stateChanged.connect(self._emit_changed)

        channels_row.addWidget(self._out_fg)

        self._out_checker = QCheckBox(self.tr("Checker"))

        self._out_checker.setChecked(True)

        self._out_checker.stateChanged.connect(self._emit_changed)

        channels_row.addWidget(self._out_checker)

        self._out_cutout = QCheckBox(self.tr("Cutout"))

        self._out_cutout.setChecked(True)

        self._out_cutout.stateChanged.connect(self._emit_changed)

        channels_row.addWidget(self._out_cutout)

        out_layout.addLayout(channels_row)



        layout.addWidget(out_group)

        layout.addStretch(1)



        scroll.setWidget(inner)

        outer.addWidget(scroll)



        for btn in self.findChildren(QPushButton):

            btn.setFocusPolicy(Qt.NoFocus)

        for spin in self.findChildren(QSpinBox) + self.findChildren(QDoubleSpinBox):

            spin.setLocale(QLocale.c())



    def _emit_changed(self) -> None:

        if not self._suppress_signals:

            self.params_changed.emit()



    def _emit_chroma_preview(self) -> None:

        if not self._suppress_signals:

            self.chroma_key_preview.emit()



    def _on_carry_hint_toggled(self, _state: int) -> None:

        self._carry_hint_quality.setEnabled(
            self._use_hint.isChecked() and self._carry_hint.isChecked()
        )

        self._emit_changed()



    def _on_use_hint_toggled(self, state: int) -> None:
        enabled = (state == Qt.Checked.value or state == 2 or bool(state) is True)
        self._hint_system_label.setEnabled(enabled)
        self._hint_source.setEnabled(enabled)
        if hasattr(self, "_hint_mode_label"):
            self._hint_mode_label.setEnabled(enabled)
            self._hint_mode.setEnabled(enabled)
        self._hint_quality.setEnabled(enabled)
        self._hint_quality_label.setEnabled(enabled)
        self._carry_hint.setEnabled(enabled)
        self._carry_hint_quality.setEnabled(enabled and self._carry_hint.isChecked())
        self._emit_changed()

    def _on_garbage_matte_toggled(self, state: int) -> None:

        self._garbage_matte_px.setEnabled(state == Qt.Checked.value)

        self._emit_changed()



    def _on_bg_color_changed(self, index: int) -> None:

        if self._suppress_signals:

            return

        color = ["auto", "green", "blue"][index]

        self.screen_color_changed.emit(color)

        self._emit_changed()



    def _on_birefnet_clicked(self) -> None:
        if not getattr(self, "_birefnet_has_installed", False):
            return
        self.birefnet_requested.emit(self._birefnet_model.currentText())

    def _on_birefnet_one_frame_clicked(self) -> None:
        if not getattr(self, "_birefnet_has_installed", False):
            return
        self.birefnet_one_frame_requested.emit(self._birefnet_model.currentText())



    @property

    def live_preview_enabled(self) -> bool:

        return self._live_preview.isChecked()

    def set_live_preview_checked(self, checked: bool) -> None:

        old = self._live_preview.blockSignals(True)

        try:

            self._live_preview.setChecked(bool(checked))

        finally:

            self._live_preview.blockSignals(old)



    def get_params(self) -> InferenceParams:

        bg_idx = self._bg_color.currentIndex()

        screen_color = ["auto", "green", "blue"][bg_idx]

        vram_idx = self._vram_mode.currentIndex()

        auto_vram = (vram_idx == 0)

        low_vram = (vram_idx == 1)



        hint_source_val = self._hint_source.currentData() or "auto"
        hint_mode_val = self._hint_mode.currentData() if hasattr(self, "_hint_mode") else "auto"
        return InferenceParams(
            use_hint=self._use_hint.isChecked(),
            hint_source=hint_source_val,
            hint_mode=hint_mode_val or "auto",
            input_is_linear=self._frame_dir_linear.isChecked(),

            despill_enabled=self._despill_enabled_cb.isChecked(),

            despill_strength=self._despill_slider.value() / 100.0,

            auto_despeckle=self._despeckle_check.isChecked(),

            despeckle_size=self._despeckle_size.value(),

            screen_color=screen_color,

            garbage_matte_px=(

                self._garbage_matte_px.value()

                if self._garbage_matte_check.isChecked()

                else 0

            ),

            hint_quality=self._hint_quality.value() / 100.0,

            carry_hint=self._carry_hint.isChecked(),

            carry_hint_quality=self._carry_hint_quality.value(),

            hann_chunk=self._hann_chunk.value(),

            hann_stride=self._hann_stride.value(),

            auto_vram=auto_vram,

            low_vram=low_vram,

            stage2_batch=self._stage2_batch.value(),

            model_scale=self._model_scale.value(),

            linear_brightness=self._linear_brightness.value(),

            linear_contrast=self._linear_contrast.value(),

            linear_contrast_pivot=self._linear_contrast_pivot.value(),

            frame_dir_linear=self._frame_dir_linear.isChecked(),

            cutout_linear=self._cutout_linear.isChecked(),

            output_format="exr" if (self._output_format.currentData() or ("exr" if self._output_format.currentIndex() == 1 else "h264")) == "exr" else "video",

            export_codec=str(self._output_format.currentData() or ("exr" if self._output_format.currentIndex() == 1 else "h264")),

            exr_dtype="half" if self._exr_dtype.currentIndex() == 0 else "float",

            crf=self._crf.value(),

            preset=self._preset.currentText(),

            pix_fmt="yuv444p10le",

            edge_erode_px=self._edge_erode_px.value(),

            edge_blur_px=self._edge_blur_px.value(),

            alpha_clip_black=self._alpha_clip_black.value(),

            alpha_clip_white=self._alpha_clip_white.value(),

            alpha_gamma=self._alpha_gamma.value(),

            auto_roi_crop=self._auto_roi_crop.isChecked(),

            auto_roi_pad_px=self._auto_roi_pad_px.value(),

            color_mgmt_enabled=self._color_mgmt_group.isChecked(),
            input_log_profile=str(self._log_profile_combo.currentData() or "none"),
            display_lut_enabled=self._display_lut_check.isChecked(),
            display_lut_path=str(self._display_lut_path or ""),
            temporal_deflicker_enabled=self._deflicker_group.isChecked(),
            temporal_deflicker_strength=self._deflicker_strength.value(),
            temporal_deflicker_radius=self._deflicker_radius.value(),
            advanced_despill_enabled=self._adv_despill_group.isChecked(),
            despill_mode=str(self._despill_mode.currentData() or "average"),
            despill_luma_restore=self._despill_luma_restore.value(),
            edge_extend_px=self._edge_extend_px.value(),
            light_wrap_strength=self._light_wrap_strength.value(),
            light_wrap_radius=self._light_wrap_radius.value(),
            clean_plate_enabled=self._clean_plate_group.isChecked(),
            clean_plate_path=str(self._clean_plate_path or ""),
            clean_plate_strength=self._clean_plate_strength.value(),

        )



    def get_output_config(self) -> OutputConfig:

        fmt_data = str(self._output_format.currentData() or ("exr" if self._output_format.currentIndex() == 1 else "h264"))

        is_exr = (fmt_data == "exr")

        img_fmt = "exr" if is_exr else "png"

        return OutputConfig(

            fg_enabled=self._out_fg.isChecked(),

            fg_format=img_fmt,

            matte_enabled=self._out_alpha.isChecked(),

            matte_format=img_fmt,

            comp_enabled=self._out_checker.isChecked(),

            comp_format=img_fmt,

            processed_enabled=self._out_cutout.isChecked(),

            processed_format=img_fmt,

            cutout_enabled=self._out_cutout.isChecked(),

            cutout_format="exr" if is_exr else "mov",

            output_format="exr" if is_exr else "video",

            export_codec=fmt_data,

            exr_dtype="half" if self._exr_dtype.currentIndex() == 0 else "float",

        )



    def set_input_is_linear(self, input_is_linear: bool) -> None:

        self._suppress_signals = True

        try:

            self._frame_dir_linear.setChecked(input_is_linear)

        finally:

            self._suppress_signals = False



    def set_params(self, params: InferenceParams) -> None:

        self._suppress_signals = True

        try:

            if hasattr(params, 'use_hint'):
                self._use_hint.setChecked(params.use_hint)
                self._on_use_hint_toggled(params.use_hint)
            if hasattr(params, 'hint_source'):
                idx = self._hint_source.findData(params.hint_source)
                if idx >= 0:
                    self._hint_source.setCurrentIndex(idx)
            if hasattr(params, 'hint_mode') and hasattr(self, "_hint_mode"):
                idx_mode = self._hint_mode.findData(params.hint_mode)
                if idx_mode >= 0:
                    self._hint_mode.setCurrentIndex(idx_mode)
            sc = getattr(params, "screen_color", "auto")
            sc_map = {"auto": 0, "green": 1, "blue": 2}
            if sc in sc_map:
                self._bg_color.setCurrentIndex(sc_map[sc])
            auto_v = bool(getattr(params, "auto_vram", True))
            low_v = bool(getattr(params, "low_vram", False))
            self._vram_mode.setCurrentIndex(0 if auto_v else (1 if low_v else 2))
            if hasattr(params, "stage2_batch"):
                self._stage2_batch.setValue(int(params.stage2_batch))
            if hasattr(params, "linear_brightness"):
                self._linear_brightness.setValue(float(params.linear_brightness))
            if hasattr(params, "linear_contrast"):
                self._linear_contrast.setValue(float(params.linear_contrast))
            if hasattr(params, "linear_contrast_pivot"):
                self._linear_contrast_pivot.setValue(float(params.linear_contrast_pivot))
            self._frame_dir_linear.setChecked(getattr(params, "frame_dir_linear", False))
            self._cutout_linear.setChecked(getattr(params, "cutout_linear", False))

            despill_on = bool(getattr(params, "despill_enabled", True))
            self._despill_enabled_cb.setChecked(despill_on)
            self._despill_slider.setEnabled(despill_on)
            self._despill_label.setEnabled(despill_on)
            self._despill_slider.setValue(int(params.despill_strength * 100))

            self._despeckle_check.setChecked(params.auto_despeckle)

            self._despeckle_size.setValue(params.despeckle_size)

            gm_px = int(getattr(params, "garbage_matte_px", 0) or 0)
            gm_on = gm_px > 0

            self._garbage_matte_check.setChecked(gm_on)

            if gm_on:

                self._garbage_matte_px.setValue(gm_px)

            self._garbage_matte_px.setEnabled(gm_on)

            self._edge_erode_px.setValue(int(getattr(params, "edge_erode_px", 0) or 0))
            self._edge_blur_px.setValue(int(getattr(params, "edge_blur_px", 0) or 0))
            self._alpha_clip_black.setValue(float(getattr(params, "alpha_clip_black", 0.0) or 0.0))
            w_val = getattr(params, "alpha_clip_white", 1.0)
            self._alpha_clip_white.setValue(float(1.0 if w_val is None else w_val))
            self._alpha_gamma.setValue(float(getattr(params, "alpha_gamma", 1.0) or 1.0))
            self._auto_roi_crop.setChecked(bool(getattr(params, "auto_roi_crop", False)))
            self._auto_roi_pad_px.setValue(int(getattr(params, "auto_roi_pad_px", 96) or 96))

            if hasattr(params, "hint_quality"):

                self._hint_quality.setValue(int(params.hint_quality * 100))

            if hasattr(params, "carry_hint"):

                self._carry_hint.setChecked(params.carry_hint)

            if hasattr(params, "carry_hint_quality"):

                self._carry_hint_quality.setValue(params.carry_hint_quality)

            self._carry_hint_quality.setEnabled(self._use_hint.isChecked() and self._carry_hint.isChecked())

            if hasattr(params, "hann_chunk"):

                self._hann_chunk.setValue(params.hann_chunk)

            if hasattr(params, "hann_stride"):

                self._hann_stride.setValue(params.hann_stride)

            if hasattr(params, "model_scale"):

                self._model_scale.setValue(params.model_scale)

            out_fmt = str(getattr(params, "output_format", "video") or "video").lower()
            exp_codec = str(getattr(params, "export_codec", "h264") or "h264").lower()
            if out_fmt == "exr":
                idx_fmt = self._output_format.findData("exr")
                self._output_format.setCurrentIndex(idx_fmt if idx_fmt >= 0 else 1)
            else:
                idx_fmt = self._output_format.findData(exp_codec)
                self._output_format.setCurrentIndex(idx_fmt if idx_fmt >= 0 else 0)
            exr_dt = str(getattr(params, "exr_dtype", "half") or "half").lower()
            self._exr_dtype.setCurrentIndex(1 if exr_dt == "float" else 0)
            if hasattr(params, "crf"):
                self._crf.setValue(int(params.crf))
            if hasattr(params, "preset"):
                idx_p = self._preset.findText(str(params.preset))
                if idx_p >= 0:
                    self._preset.setCurrentIndex(idx_p)

            self._color_mgmt_group.setChecked(bool(getattr(params, "color_mgmt_enabled", False)))
            idx_lp = self._log_profile_combo.findData(str(getattr(params, "input_log_profile", "none") or "none"))
            if idx_lp >= 0:
                self._log_profile_combo.setCurrentIndex(idx_lp)
            self._display_lut_check.setChecked(bool(getattr(params, "display_lut_enabled", False)))
            self.set_display_lut_path(str(getattr(params, "display_lut_path", "") or ""))

            self._adv_despill_group.setChecked(bool(getattr(params, "advanced_despill_enabled", False)))
            idx_dm = self._despill_mode.findData(str(getattr(params, "despill_mode", "average") or "average"))
            if idx_dm >= 0:
                self._despill_mode.setCurrentIndex(idx_dm)
            self._despill_luma_restore.setValue(float(getattr(params, "despill_luma_restore", 0.0) or 0.0))
            self._edge_extend_px.setValue(int(getattr(params, "edge_extend_px", 0) or 0))
            self._light_wrap_strength.setValue(float(getattr(params, "light_wrap_strength", 0.0) or 0.0))
            self._light_wrap_radius.setValue(int(getattr(params, "light_wrap_radius", 12) or 12))

            self._deflicker_group.setChecked(bool(getattr(params, "temporal_deflicker_enabled", False)))
            self._deflicker_strength.setValue(float(getattr(params, "temporal_deflicker_strength", 0.35) or 0.35))
            self._deflicker_radius.setValue(int(getattr(params, "temporal_deflicker_radius", 1) or 1))

            self._clean_plate_group.setChecked(bool(getattr(params, "clean_plate_enabled", False)))
            self.set_clean_plate_path(str(getattr(params, "clean_plate_path", "") or ""))
            self._clean_plate_strength.setValue(float(getattr(params, "clean_plate_strength", 0.5) or 0.5))

        finally:

            self._suppress_signals = False



    def set_track_masks_enabled(self, enabled: bool) -> None:
        self._track_masks_btn.setEnabled(enabled)
        if hasattr(self, "_sam3_one_frame_btn"):
            self._sam3_one_frame_btn.setEnabled(enabled)

    def set_clear_mask_enabled(self, enabled: bool) -> None:
        self._clear_mask_btn.setEnabled(enabled)

    def refresh_birefnet_models(self) -> None:
        """Populate BiRefNet dropdown with ONLY locally downloaded models."""
        from modules.BiRefNetModule.wrapper import (
            BIREFNET_MODELS,
            DEFAULT_MODEL,
            resolve_birefnet_snapshot,
        )
        prev = self._birefnet_model.currentText()
        self._birefnet_model.clear()
        seen_repos: set[str] = set()
        installed_names: list[str] = []
        for name, (repo_id, _res) in BIREFNET_MODELS.items():
            if repo_id in seen_repos:
                continue
            seen_repos.add(repo_id)
            if resolve_birefnet_snapshot(repo_id) is not None:
                installed_names.append(name)

        self._birefnet_has_installed = bool(installed_names)
        if installed_names:
            for name in installed_names:
                self._birefnet_model.addItem(name)
            self._birefnet_model.setEnabled(True)
            pick = prev if prev in installed_names else (DEFAULT_MODEL if DEFAULT_MODEL in installed_names else installed_names[0])
            idx = self._birefnet_model.findText(pick)
            if idx >= 0:
                self._birefnet_model.setCurrentIndex(idx)
        else:
            self._birefnet_model.addItem(self.tr("No BiRefNet installed (Edit -> Download Manager)"))
            self._birefnet_model.setEnabled(False)
        self.set_birefnet_enabled(getattr(self, "_birefnet_clip_enabled", False))

    def set_birefnet_enabled(self, enabled: bool) -> None:
        self._birefnet_clip_enabled = bool(enabled)
        can_run = self._birefnet_clip_enabled and getattr(self, "_birefnet_has_installed", False)
        self._birefnet_btn.setEnabled(can_run)
        if hasattr(self, "_birefnet_one_frame_btn"):
            self._birefnet_one_frame_btn.setEnabled(can_run)



    def set_matanyone2_enabled(self, enabled: bool) -> None:

        """No-op compatibility placeholder after MatAnyone2 removal."""

        pass



    def set_videomama_enabled(self, enabled: bool) -> None:

        """No-op compatibility placeholder after VideoMaMa removal."""

        pass



    def set_import_alpha_enabled(self, enabled: bool) -> None:

        self._import_alpha_btn.setEnabled(enabled)



    def set_chroma_key_enabled(self, enabled: bool) -> None:

        self._chroma_key_btn.setEnabled(enabled)

        if not enabled:

            self._chroma_key_btn.setChecked(False)



    def set_output_config(self, config: OutputConfig) -> None:

        self._suppress_signals = True

        try:

            self._out_fg.setChecked(config.fg_enabled)

            self._out_alpha.setChecked(config.matte_enabled)

            self._out_checker.setChecked(config.comp_enabled)

            self._out_cutout.setChecked(bool(config.cutout_enabled or config.processed_enabled))

        finally:

            self._suppress_signals = False



    def set_chroma_params(self, params: dict) -> None:

        if not params:

            return

        self._suppress_signals = True

        try:

            if "strength" in params:

                self._ck_strength.setValue(int(params["strength"] * 10))

            if "clip_black" in params:

                self._ck_clip_black.setValue(int(params["clip_black"] * 100))

            if "clip_white" in params:

                self._ck_clip_white.setValue(int(params["clip_white"] * 100))

            if "screen_color" in params and params["screen_color"]:

                c = params["screen_color"]

                self.set_sampled_screen_color(c[0], c[1], c[2])

            if "screen_samples" in params and params["screen_samples"]:

                self._ck_screen_samples = params["screen_samples"]

        finally:

            self._suppress_signals = False



    def reset_chroma_params(self) -> None:

        self._suppress_signals = True

        try:

            self._ck_strength.setValue(10)

            self._ck_clip_black.setValue(8)

            self._ck_clip_white.setValue(92)

            self._ck_screen_color = None

            self._ck_screen_samples = []

            self._color_swatch.setStyleSheet("background: #00B140; border: 1px solid #374151; border-radius: 3px;")

        finally:

            self._suppress_signals = False



    def set_annotation_info(self, annotated: int, total: int = 0) -> None:

        if annotated > 0 and total > 0:

            self._annotation_info.setText(self.tr("Prompts: %d / %d frames") % (annotated, total))

        else:

            self._annotation_info.setText("")



    def _on_chroma_key_toggled(self, checked: bool) -> None:

        self._chroma_params_widget.setVisible(checked)

        if not checked:

            self._eyedropper_btn.setChecked(False)



    def _on_eyedropper_toggled(self, checked: bool) -> None:

        self.eyedropper_requested.emit(checked)



    def preview_screen_color(self, r: int, g: int, b: int) -> None:

        self._color_swatch.setStyleSheet(f"background: rgb({r},{g},{b}); border: 2px solid #10B981;")



    def set_screen_samples(self, samples: list[tuple[int, int, int]]) -> None:

        if len(samples) > 200:

            step = len(samples) / 200

            samples = [samples[int(i * step)] for i in range(200)]

        self._ck_screen_samples = samples



    def set_sampled_screen_color(self, r: int, g: int, b: int) -> None:

        self._ck_screen_color = (r, g, b)

        self._color_swatch.setStyleSheet(f"background: rgb({r},{g},{b}); border: 1px solid #374151;")

        self._eyedropper_btn.setChecked(False)



    def get_chroma_params(self) -> dict:

        bg_idx = self._bg_color.currentIndex()

        screen_type = ["auto", "green", "blue"][bg_idx]

        return {

            "screen_color": self._ck_screen_color,

            "screen_samples": self._ck_screen_samples or None,

            "screen_type": screen_type,

            "strength": self._ck_strength.value() / 10.0,

            "clip_black": self._ck_clip_black.value() / 100.0,

            "clip_white": self._ck_clip_white.value() / 100.0,

            "shrink_grow": self._edge_erode_px.value(),

            "edge_blur": self._edge_blur_px.value(),

        }



    def _on_chroma_key_generate(self) -> None:

        self.chroma_key_requested.emit(self.get_chroma_params())

    def poly_invert_checked(self) -> bool:
        return bool(self._poly_invert_check.isChecked())

    def poly_feather_value(self) -> int:
        return int(self._poly_feather_spin.value())

    def poly_curve_checked(self) -> bool:
        return bool(self._poly_curve_check.isChecked())

    def set_display_lut_path(self, path: str) -> None:
        import os as _os
        self._display_lut_path = str(path or "").strip()
        self._lut_path_label.setText(
            _os.path.basename(self._display_lut_path) if self._display_lut_path else self.tr("None")
        )
        self._on_lut_controls_changed()

    def set_clean_plate_path(self, path: str) -> None:
        import os as _os
        self._clean_plate_path = str(path or "").strip()
        self._clean_plate_label.setText(
            _os.path.basename(self._clean_plate_path) if self._clean_plate_path else self.tr("None")
        )

    def _on_lut_controls_changed(self, *_args) -> None:
        if getattr(self, "_suppress_signals", False):
            return
        enabled = bool(self._color_mgmt_group.isChecked() and self._display_lut_check.isChecked())
        self.lut_changed.emit(enabled, str(self._display_lut_path or ""))

    def _on_browse_lut(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Display .cube LUT"),
            "",
            "CUBE LUT (*.cube);;All Files (*)",
        )
        if path:
            self.set_display_lut_path(path)
            self._color_mgmt_group.setChecked(True)
            self._display_lut_check.setChecked(True)
            self._on_lut_controls_changed()
            self._emit_changed()

    def _on_browse_clean_plate(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Empty Clean Plate Image"),
            "",
            "Images (*.png *.jpg *.jpeg *.exr *.tif *.tiff);;All Files (*)",
        )
        if path:
            self.set_clean_plate_path(path)
            self._clean_plate_group.setChecked(True)
            self._emit_changed()

    def _on_apply_preset(self) -> None:
        name = self._preset_combo.currentText()
        pdata = self._preset_mgr.get_preset(name)
        if not pdata:
            return
        merged = {**self.get_params().to_dict(), **pdata}
        self.set_params(InferenceParams.from_dict(merged))
        self.params_changed.emit()

    def _on_save_preset(self) -> None:
        name, ok = QInputDialog.getText(
            self,
            self.tr("Save Preset"),
            self.tr("Preset name:"),
            text="Custom Preset",
        )
        if not ok or not (name or "").strip():
            return
        clean = name.strip()
        self._preset_mgr.save_preset(clean, self.get_params().to_dict())
        if self._preset_combo.findText(clean) < 0:
            self._preset_combo.addItem(clean)
        self._preset_combo.setCurrentText(clean)

    def _on_copy_params(self) -> None:
        data = self.get_params().to_dict()
        self._copied_params_dict = data
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(json.dumps(data))

    def _on_paste_params(self) -> None:
        data = None
        cb = QApplication.clipboard()
        if cb is not None:
            raw = (cb.text() or "").strip()
            if raw.startswith("{"):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        data = parsed
                except Exception:
                    data = None
        if data is None:
            data = self._copied_params_dict
        if isinstance(data, dict):
            self.set_params(InferenceParams.from_dict(data))
            self.params_changed.emit()

