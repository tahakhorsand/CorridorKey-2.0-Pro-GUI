"""Right panel — alpha generation, tracking, and CorridorKey 2.0 native parameters."""

from __future__ import annotations



import sys

from PySide6.QtCore import QEvent, QLocale, QSettings, Qt, Signal

from PySide6.QtWidgets import (

    QCheckBox,

    QComboBox,

    QDoubleSpinBox,

    QGridLayout,

    QGroupBox,

    QHBoxLayout,

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

        self._ck_strength.valueChanged.connect(lambda: self.chroma_key_preview.emit())

        ck_layout.addWidget(self._ck_strength)



        self._ck_clip_black_label = QLabel(self.tr("Clip Black: 0.08"))

        ck_layout.addWidget(self._ck_clip_black_label)

        self._ck_clip_black = _no_scroll_wheel(QSlider(Qt.Horizontal))

        self._ck_clip_black.setRange(0, 100)

        self._ck_clip_black.setValue(8)

        self._ck_clip_black.valueChanged.connect(

            lambda v: self._ck_clip_black_label.setText(self.tr("Clip Black: %s") % f"{v / 100:.2f}")

        )

        self._ck_clip_black.valueChanged.connect(lambda: self.chroma_key_preview.emit())

        ck_layout.addWidget(self._ck_clip_black)



        self._ck_clip_white_label = QLabel(self.tr("Clip White: 0.92"))

        ck_layout.addWidget(self._ck_clip_white_label)

        self._ck_clip_white = _no_scroll_wheel(QSlider(Qt.Horizontal))

        self._ck_clip_white.setRange(0, 100)

        self._ck_clip_white.setValue(92)

        self._ck_clip_white.valueChanged.connect(

            lambda v: self._ck_clip_white_label.setText(self.tr("Clip White: %s") % f"{v / 100:.2f}")

        )

        self._ck_clip_white.valueChanged.connect(lambda: self.chroma_key_preview.emit())

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
        from modules.BiRefNetModule.wrapper import BIREFNET_MODELS, DEFAULT_MODEL
        for m in BIREFNET_MODELS:
            self._birefnet_model.addItem(m)
        idx = self._birefnet_model.findText(DEFAULT_MODEL)
        if idx >= 0:
            self._birefnet_model.setCurrentIndex(idx)
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

        self._carry_hint.stateChanged.connect(self._emit_changed)

        carry_row.addWidget(self._carry_hint, 1)



        self._carry_hint_quality = _no_scroll_wheel(QDoubleSpinBox())

        self._carry_hint_quality.setRange(0.5, 1.0)

        self._carry_hint_quality.setSingleStep(0.05)

        self._carry_hint_quality.setValue(0.95)

        self._carry_hint_quality.setToolTip(self.tr("Confidence weight applied to carried temporal hints."))

        self._carry_hint_quality.valueChanged.connect(self._emit_changed)

        carry_row.addWidget(self._carry_hint_quality)

        ck2_layout.addLayout(carry_row)



        # Despill Strength Slider

        self._despill_label = QLabel(self.tr("Despill Strength: 0.50"))

        ck2_layout.addWidget(self._despill_label)

        self._despill_slider = _no_scroll_wheel(QSlider(Qt.Horizontal))

        self._despill_slider.setRange(0, 100)

        self._despill_slider.setValue(50)

        self._despill_slider.setToolTip(self.tr("Native CorridorKey 2.0 despill algorithm strength (0.0 - 1.0)."))

        self._despill_slider.valueChanged.connect(

            lambda v: self._despill_label.setText(self.tr("Despill Strength: %s") % f"{v / 100:.2f}")

        )

        self._despill_slider.valueChanged.connect(self._emit_changed)

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



        ck2_layout.addLayout(cleanup_grid)



        self._live_preview = QCheckBox(self.tr("Live Preview (Current Frame)"))

        self._live_preview.setChecked(True)

        ck2_layout.addWidget(self._live_preview)



        layout.addWidget(ck2_group)



        # ── Section 3: OUTPUT ENCODING & EXPORT ──

        out_group = QGroupBox(self.tr("OUTPUT & EXPORT"))

        out_layout = QVBoxLayout(out_group)

        out_layout.setSpacing(6)



        out_grid = QGridLayout()

        out_grid.setHorizontalSpacing(4)

        out_grid.setVerticalSpacing(4)



        out_grid.addWidget(QLabel(self.tr("Format")), 0, 0)

        self._output_format = _no_scroll_wheel(QComboBox())

        self._output_format.addItems([self.tr("Video (MP4/H.264)"), self.tr("OpenEXR Sequences")])

        self._output_format.currentIndexChanged.connect(self._emit_changed)

        out_grid.addWidget(self._output_format, 0, 1)



        out_grid.addWidget(QLabel(self.tr("EXR Precision")), 1, 0)

        self._exr_dtype = _no_scroll_wheel(QComboBox())

        self._exr_dtype.addItems([self.tr("Half (16-bit Float)"), self.tr("Float (32-bit Float)")])

        out_grid.addWidget(self._exr_dtype, 1, 1)



        out_grid.addWidget(QLabel(self.tr("H.264 CRF / Preset")), 2, 0)

        enc_row = QHBoxLayout()

        enc_row.setSpacing(2)

        self._crf = _no_scroll_wheel(QSpinBox())

        self._crf.setRange(0, 51)

        self._crf.setValue(12)

        enc_row.addWidget(self._crf)

        self._preset = _no_scroll_wheel(QComboBox())

        self._preset.addItems(["fast", "medium", "slow"])

        self._preset.setCurrentText("medium")

        enc_row.addWidget(self._preset)

        out_grid.addLayout(enc_row, 2, 1)



        out_layout.addLayout(out_grid)



        # Channel Toggles

        channels_row = QHBoxLayout()

        channels_row.setSpacing(4)

        self._out_alpha = QCheckBox(self.tr("Alpha"))

        self._out_alpha.setChecked(True)

        channels_row.addWidget(self._out_alpha)

        self._out_fg = QCheckBox(self.tr("FG"))

        self._out_fg.setChecked(True)

        channels_row.addWidget(self._out_fg)

        self._out_checker = QCheckBox(self.tr("Checker"))

        self._out_checker.setChecked(True)

        channels_row.addWidget(self._out_checker)

        self._out_cutout = QCheckBox(self.tr("Cutout"))

        self._out_cutout.setChecked(True)

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

        color = ["auto", "green", "blue"][index]

        self.screen_color_changed.emit(color)

        self._emit_changed()



    def _on_birefnet_clicked(self) -> None:
        self.birefnet_requested.emit(self._birefnet_model.currentText())

    def _on_birefnet_one_frame_clicked(self) -> None:
        self.birefnet_one_frame_requested.emit(self._birefnet_model.currentText())



    @property

    def live_preview_enabled(self) -> bool:

        return self._live_preview.isChecked()



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

            output_format="video" if self._output_format.currentIndex() == 0 else "exr",

            exr_dtype="half" if self._exr_dtype.currentIndex() == 0 else "float",

            crf=self._crf.value(),

            preset=self._preset.currentText(),

            pix_fmt="yuv444p10le",

        )



    def get_output_config(self) -> OutputConfig:

        is_exr = self._output_format.currentIndex() == 1

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
            self._frame_dir_linear.setChecked(getattr(params, "frame_dir_linear", False))

            self._despill_slider.setValue(int(params.despill_strength * 100))

            self._despeckle_check.setChecked(params.auto_despeckle)

            self._despeckle_size.setValue(params.despeckle_size)

            gm_on = getattr(params, "garbage_matte_px", 0) > 0

            self._garbage_matte_check.setChecked(gm_on)

            if gm_on:

                self._garbage_matte_px.setValue(params.garbage_matte_px)

            self._garbage_matte_px.setEnabled(gm_on)

            if hasattr(params, "hint_quality"):

                self._hint_quality.setValue(int(params.hint_quality * 100))

            if hasattr(params, "carry_hint"):

                self._carry_hint.setChecked(params.carry_hint)

            if hasattr(params, "carry_hint_quality"):

                self._carry_hint_quality.setValue(params.carry_hint_quality)

            if hasattr(params, "hann_chunk"):

                self._hann_chunk.setValue(params.hann_chunk)

            if hasattr(params, "hann_stride"):

                self._hann_stride.setValue(params.hann_stride)

            if hasattr(params, "model_scale"):

                self._model_scale.setValue(params.model_scale)

        finally:

            self._suppress_signals = False



    def set_track_masks_enabled(self, enabled: bool) -> None:
        self._track_masks_btn.setEnabled(enabled)
        if hasattr(self, "_sam3_one_frame_btn"):
            self._sam3_one_frame_btn.setEnabled(enabled)

    def set_clear_mask_enabled(self, enabled: bool) -> None:
        self._clear_mask_btn.setEnabled(enabled)

    def set_birefnet_enabled(self, enabled: bool) -> None:
        self._birefnet_btn.setEnabled(enabled)
        if hasattr(self, "_birefnet_one_frame_btn"):
            self._birefnet_one_frame_btn.setEnabled(enabled)



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

            self._out_cutout.setChecked(config.processed_enabled)

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

            "shrink_grow": 0,

            "edge_blur": 0,

        }



    def _on_chroma_key_generate(self) -> None:

        self.chroma_key_requested.emit(self.get_chroma_params())

