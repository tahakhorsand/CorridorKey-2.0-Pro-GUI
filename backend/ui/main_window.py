"""Main window — dual viewer layout with I/O tray and menu bar.

Layout:
    +--[CORRIDORKEY]------------------[GPU | VRAM ## X/YGB]-+
    +----------------------+--------------------------------+
    |  INPUT    | OUTPUT   |  Parameters                    |
    |  Viewer   | Viewer   |    Panel                       |
    |  (fills)  | (fills)  |  (280px)                       |
    +-----------+----------+--------------------------------+
    |  INPUT (N)  [+ADD]       |  EXPORTS (N)               |
    +----------------------------------------------------------+
    |  Queue Panel (collapsible, per-job progress)         |
    +----------------------------------------------------------+
    |  [progress]  frame counter  warnings  [RUN/STOP]     |
    +----------------------------------------------------------+
"""
from __future__ import annotations

import glob as glob_module
import json
import logging
import os
import re
import shutil
import sys

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QVBoxLayout, QWidget,
    QLabel, QHBoxLayout, QMessageBox, QStackedWidget,
    QProgressBar, QFileDialog, QInputDialog, QGraphicsOpacityEffect,
    QPushButton,
)
from PySide6.QtCore import Qt, Slot, QTimer, QPropertyAnimation, QEasingCurve, QSettings, QThread, Signal
from PySide6.QtGui import QKeySequence, QAction, QImage, QPainter

from backend import (
    CorridorKeyService, ClipAsset, ClipEntry, ClipState, InferenceParams,
    InOutRange, OutputConfig, JobType,
    PipelineRoute, classify_pipeline_route, mask_sequence_is_videomama_ready,
)
from backend.project import VIDEO_FILE_FILTER, is_video_file
from backend.frame_io import open_video, imwrite_unicode

from ui.models.clip_model import ClipListModel
from ui.preview.frame_index import ViewMode
from ui.preview.display_transform import processed_rgba_to_qimage
from ui.widgets.dual_viewer import DualViewerPanel
from ui.widgets.parameter_panel import ParameterPanel
from ui.widgets.status_bar import StatusBar
from ui.widgets.queue_panel import QueuePanel
from ui.widgets.io_tray_panel import IOTrayPanel
from ui.widgets.welcome_screen import WelcomeScreen
from ui.widgets.preferences_dialog import (
    PreferencesDialog, KEY_SHOW_TOOLTIPS, DEFAULT_SHOW_TOOLTIPS,
    KEY_COPY_SOURCE, DEFAULT_COPY_SOURCE,
    KEY_COPY_SEQUENCES, DEFAULT_COPY_SEQUENCES,
    KEY_TRACKER_MODEL, DEFAULT_TRACKER_MODEL,
    KEY_MODEL_RESOLUTION, DEFAULT_MODEL_RESOLUTION,
    KEY_PARALLEL_CLIPS, DEFAULT_PARALLEL_CLIPS,
    get_setting_bool, get_setting_str, get_setting_int,
)
from ui.workers.gpu_job_worker import GPUJobWorker, create_job_snapshot
from ui.workers.gpu_monitor import GPUMonitor
from ui.workers.thumbnail_worker import ThumbnailGenerator
from ui.workers.extract_worker import ExtractWorker
from ui.recent_sessions import RecentSessionsStore
from ui.shortcut_registry import ShortcutRegistry

from ui.main_window_mixins import (
    MenuMixin, ShortcutsMixin, ClipMixin, ImportMixin,
    InferenceMixin, AlphaImportMixin, ModelRunMixin, CancelMixin,
    WorkerMixin, AnnotationMixin,
    ExportMixin, SessionMixin, SettingsMixin,
)
from ui.main_window_mixins.batch_pipeline_mixin import BatchPipelineMixin
from ui.main_window_mixins.chroma_key_mixin import ChromaKeyMixin

logger = logging.getLogger(__name__)


class _StartupProbeWorker(QThread):
    """Run device probe and optional startup diagnostics off the main GUI thread."""

    ready = Signal(str, list)

    def __init__(self, service: CorridorKeyService, run_diagnostics: bool = True, parent=None):
        super().__init__(parent)
        self._service = service
        self._run_diagnostics = run_diagnostics

    def run(self) -> None:
        device = "cpu"
        issues: list = []
        try:
            device = self._service.detect_device()
        except Exception as exc:
            logger.warning("Startup device probe failed: %s", exc)
        if self._run_diagnostics and not self.isInterruptionRequested():
            try:
                from ui.widgets.diagnostic_dialog import run_startup_diagnostics
                issues = run_startup_diagnostics(device) or []
            except Exception as exc:
                logger.warning("Startup diagnostics failed: %s", exc)
        if not self.isInterruptionRequested():
            self.ready.emit(device, issues)



class _Toast(QLabel):
    """Non-blocking notification that auto-fades after a duration. Click to dismiss."""

    def __init__(self, parent: QWidget, text: str, duration_ms: int = 4000,
                 center: bool = False):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background: #1A1900; color: #E0E0D0; border: 1px solid #454430;"
            "border-radius: 6px; padding: 12px 20px; font-size: 13px;"
        )
        self.setFixedWidth(380)
        self.adjustSize()
        if center:
            # Position center of window
            self.move(
                (parent.width() - self.width()) // 2,
                (parent.height() - self.height()) // 2,
            )
        else:
            # Position bottom-center, above the status bar
            self.move(
                (parent.width() - self.width()) // 2,
                parent.height() - self.height() - 60,
            )
        # Fade-out animation
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity)
        self._fade = QPropertyAnimation(self._opacity, b"opacity")
        self._fade.setDuration(800)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.setEasingCurve(QEasingCurve.InQuad)
        self._fade.finished.connect(self.deleteLater)
        # Start fade after hold duration
        QTimer.singleShot(duration_ms, self._fade.start)
        self.show()
        self.raise_()

    def mousePressEvent(self, event):
        self.deleteLater()
        super().mousePressEvent(event)


def _remove_alpha_hint_assets(root_path: str) -> None:
    """Delete sequence or video alpha-hint assets for a clip."""
    alpha_dir = os.path.join(root_path, "AlphaHint")
    if os.path.isdir(alpha_dir):
        shutil.rmtree(alpha_dir, ignore_errors=True)

    for candidate in glob_module.glob(os.path.join(root_path, "AlphaHint.*")):
        if os.path.isfile(candidate) and is_video_file(candidate):
            try:
                os.remove(candidate)
            except OSError:
                logger.warning("Failed to remove alpha video asset: %s", candidate)


def _import_alpha_video_as_sequence(
    video_path: str,
    alpha_dir: str,
    input_files: list[str],
) -> int:
    """Decode an alpha video into AlphaHint/*.png named to match input stems."""
    os.makedirs(alpha_dir, exist_ok=True)
    cap = open_video(video_path)
    imported_count = 0
    try:
        for input_name in input_files:
            ret, frame = cap.read()
            if not ret:
                break
            input_stem = os.path.splitext(input_name)[0]
            dst_path = os.path.join(alpha_dir, f"{input_stem}.png")
            if frame.ndim == 2:
                mask_u8 = frame
            elif frame.ndim == 3 and frame.shape[2] == 4:
                mask_u8 = frame[:, :, 3]
            else:
                mask_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if mask_u8.dtype != np.uint8:
                mask_u8 = (np.clip(mask_u8, 0.0, 1.0) * 255.0).astype(np.uint8)
            if imwrite_unicode(dst_path, mask_u8):
                imported_count += 1
            else:
                logger.warning("Failed to write imported alpha video frame: %s", dst_path)
    finally:
        cap.release()
    return imported_count


class _MuteOverlay(QLabel):
    """Brief overlay inside the brand bar strip, fades after 1.5s."""

    _FIXED_W = 160
    _FIXED_H = 22

    def __init__(self, parent: QWidget, text: str):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(self._FIXED_W, self._FIXED_H)
        self.setStyleSheet(
            "background: rgba(26, 25, 0, 220); color: #E0E0D0;"
            "border: 1px solid #454430; border-radius: 4px;"
            "font-family: 'Open Sans'; font-size: 11px; font-weight: 600;"
        )
        # Position inside the brand bar (top strip, ~24px tall)
        menu_h = parent.menuBar().height() if hasattr(parent, 'menuBar') else 22
        self.move((parent.width() - self._FIXED_W) // 2, menu_h + 2)
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity)
        self._fade = QPropertyAnimation(self._opacity, b"opacity")
        self._fade.setDuration(600)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.setEasingCurve(QEasingCurve.InQuad)
        self._fade.finished.connect(self.deleteLater)
        QTimer.singleShot(1500, self._fade.start)
        self.raise_()


class _UpdateChecker(QThread):
    """Background thread that checks GitHub for a newer version."""
    update_available = Signal(str)  # emits the remote version string

    def __init__(self, local_version: str):
        super().__init__()
        self._local = local_version

    def run(self):
        try:
            import sys as _sys
            import urllib.request
            # Read the published GitHub release, the SAME source the updater
            # downloads from (_run_frozen_update). Reading main's pyproject.toml
            # instead would nag for a version that has no downloadable release
            # yet whenever main is bumped ahead of a cut release.
            from backend.update_verify import UPDATE_ORIGIN
            url = (
                f"https://api.github.com/repos/{UPDATE_ORIGIN}"
                "/releases/latest"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "CorridorKey"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                release = json.loads(resp.read().decode("utf-8"))
            tag = release.get("tag_name", "")
            remote = tag.lstrip("v") if tag else ""
            if not remote or not self._is_newer(remote, self._local):
                return
            # Frozen builds self-update by downloading a platform asset. Do not
            # nag if that asset is missing from the release (the click would
            # fail in _run_frozen_update). Source installs update via git pull,
            # so they only need the version to be newer.
            if getattr(_sys, "frozen", False) and not self._has_platform_asset(release, remote):
                return
            self.update_available.emit(remote)
        except Exception:
            pass  # silently fail — no internet / 404 (no releases) / no button

    @staticmethod
    def _has_platform_asset(release: dict, version: str) -> bool:
        """Whether the release carries the frozen-update asset for this platform.

        Mirrors the asset name built in _run_frozen_update.
        """
        import sys as _sys
        if _sys.platform == "darwin":
            asset_platform = "macOS-arm64"
        elif _sys.platform == "win32":
            asset_platform = "Windows-x64"
        else:
            return False
        asset_name = f"CorridorKey-{version}-{asset_platform}.zip"
        return any(a.get("name") == asset_name for a in release.get("assets", []))

    @staticmethod
    def _is_newer(remote: str, local: str) -> bool:
        """Simple semver comparison: 1.4.0 > 1.3.1."""
        try:
            r = tuple(int(x) for x in remote.split("."))
            l = tuple(int(x) for x in local.split("."))
            return r > l
        except (ValueError, AttributeError):
            return False


class MainWindow(
    QMainWindow,
    MenuMixin, ShortcutsMixin, ClipMixin, ImportMixin,
    InferenceMixin, AlphaImportMixin, ModelRunMixin, CancelMixin,
    WorkerMixin, AnnotationMixin, ChromaKeyMixin, BatchPipelineMixin,
    ExportMixin, SessionMixin, SettingsMixin,
):
    """CorridorKey main application window."""

    _mps_warning_acknowledged = False

    def _warn_mps_slow(self, feature_name: str) -> bool:
        """Show a one-time performance warning on MPS. Returns False if user cancels."""
        if getattr(self._service, '_device', '') != 'mps':
            return True
        if MainWindow._mps_warning_acknowledged:
            return True
        reply = QMessageBox.warning(
            self,
            self.tr("%s — Mac Performance Warning") % feature_name,
            self.tr(
                "GPU-intensive features (SAM 3.1, BiRefNet) "
                "are very slow on Mac (Apple Silicon MPS).\n\n"
                "This may take hours for longer clips and could freeze your system.\n\n"
                "Recommendation: Import pre-made alpha mattes from After Effects, "
                "DaVinci Resolve, or Nuke instead.\n\n"
                "Continue anyway? (This warning won't appear again this session.)"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            MainWindow._mps_warning_acknowledged = True
            return True
        return False

    def __init__(self, service: CorridorKeyService | None = None,
                 store: RecentSessionsStore | None = None):
        super().__init__()
        self.setWindowTitle(self.tr("CorridorKey 2.0"))
        self.setMinimumSize(1100, 650)

        container_env = str(os.getenv("CORRIDORKEY_CONTAINER_MODE", "0")).strip().lower()
        self._container_mode = container_env in {"1", "true", "yes", "on", "y"}
        logger.info(
            "MainWindow init: CORRIDORKEY_CONTAINER_MODE=%r -> container_mode=%s",
            container_env,
            self._container_mode,
        )

        if self._container_mode:
            self.setMinimumSize(1100, 650)
            logger.info("MainWindow init: fullscreen enabled for container mode")
        else:
            self._container_mode = False
            self.resize(1400, 800)


        self.setAcceptDrops(True)

        self._service = service or CorridorKeyService()
        self._recent_store = store or RecentSessionsStore()
        self._current_clip: ClipEntry | None = None
        self._last_clip_index: int = 0  # track position for left-neighbor on delete
        self._clips_dir: str | None = None
        self._clip_input_is_linear: dict[str, bool] = {}
        # Track active job ID — set only once when job starts, not on every progress
        self._active_job_id: str | None = None
        self._cancel_requested_job_id: str | None = None
        self._force_stop_armed = False
        self._skip_shutdown_cleanup = False
        self._bg_cache: QImage | None = None
        # Batch pipeline: clip_name -> remaining queued steps after the current one.
        self._pipeline_steps: dict[str, list[JobType]] = {}
        # Batch Pipeline dialog state
        self._init_batch_pipeline()
        # Debug console — created eagerly so log handler captures from startup
        from ui.widgets.debug_console import DebugConsoleWidget
        self._debug_console = DebugConsoleWidget()
        self._prefs_dialog = None

        # Data model
        self._clip_model = ClipListModel()

        # Thumbnail generator (background, Codex: no QPixmap off main thread)
        self._thumb_gen = ThumbnailGenerator(self)
        self._thumb_gen.thumbnail_ready.connect(self._clip_model.set_thumbnail)

        # Reprocess debounce timer (200ms, Codex: coalesce stale requests)
        self._reprocess_timer = QTimer(self)
        self._reprocess_timer.setSingleShot(True)
        self._reprocess_timer.setInterval(200)
        self._reprocess_timer.timeout.connect(self._do_reprocess)

        # Live selected-clip refresh: coalesce worker progress into a cheap UI-only
        # asset rescan so the scrubber and mode buttons update while frames are written.
        self._live_asset_refresh_timer = QTimer(self)
        self._live_asset_refresh_timer.setSingleShot(True)
        self._live_asset_refresh_timer.setInterval(150)
        self._live_asset_refresh_timer.timeout.connect(self._refresh_selected_clip_live_assets)
        self._pending_live_asset_refresh_clip: str | None = None

        # Shortcut registry — single source of truth for key bindings
        self._shortcut_registry = ShortcutRegistry()

        # Build UI
        self._build_menu_bar()
        self._build_central()
        self._build_status_bar()
        self._setup_shortcuts()

        # Workers
        self._gpu_worker = GPUJobWorker(
            self._service,
            max_workers=get_setting_int(KEY_PARALLEL_CLIPS, DEFAULT_PARALLEL_CLIPS),
            parent=self,
        )
        self._gpu_monitor = GPUMonitor(interval_ms=2000, parent=self)
        self._extract_worker = ExtractWorker(parent=self)
        self._extract_progress: dict[str, tuple[int, int]] = {}  # clip_name -> (current, total)

        # Connect signals
        self._connect_signals()

        # Start GPU monitoring
        self._gpu_monitor.start()

        # Periodic auto-save for crash recovery (every 60s)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(60_000)
        self._autosave_timer.timeout.connect(self._auto_save_session)
        self._autosave_timer.start()

        # Probe device + run startup diagnostics asynchronously after first show
        self._startup_probe_worker: _StartupProbeWorker | None = None
        QTimer.singleShot(50, self._start_startup_probe)

        # Always start on welcome screen — user picks a project from recents or imports
        # Deferred sync of IO tray divider with viewer splitter
        QTimer.singleShot(0, self._sync_io_divider)

        # Apply persisted preferences (e.g. tooltip visibility, sound mute)
        self._apply_tooltip_setting()
        self._apply_sound_setting()
        self._apply_tracker_model_setting()
        self._apply_model_resolution_setting()
        self._apply_inference_backend_setting()

        # Check for updates (non-blocking background thread)
        if os.environ.get("CORRIDORKEY_SKIP_UPDATE_CHECK") != "1":
            self._check_for_updates()

        # Re-apply configured window mode after first layout pass.
        QTimer.singleShot(0, self._ensure_window_mode)

    def _ensure_window_mode(self) -> None:
        """Keep the top-level window in configured startup mode."""
        if getattr(self, "_container_mode", False):
            if not self.isFullScreen():
                self.showFullScreen()
        else:
            return
        self.raise_()
        self.activateWindow()

    def _start_startup_probe(self) -> None:
        """Launch deferred background hardware probe so MainWindow paints immediately."""
        if getattr(self, "_startup_probe_worker", None) is not None:
            return
        run_diag = os.environ.get("CORRIDORKEY_SKIP_STARTUP_DIAGNOSTICS") != "1"
        worker = _StartupProbeWorker(self._service, run_diagnostics=run_diag, parent=self)
        worker.ready.connect(self._on_startup_probe_ready)
        worker.finished.connect(self._on_startup_probe_finished)
        self._startup_probe_worker = worker
        worker.start()

    @Slot(str, list)
    def _on_startup_probe_ready(self, device: str, issues: list) -> None:
        try:
            import shiboken6
            if not shiboken6.isValid(self):
                return
        except Exception:
            pass
        logger.info("Compute device ready: %s", device)
        if issues and os.environ.get("CORRIDORKEY_SKIP_STARTUP_DIAGNOSTICS") != "1":
            from ui.widgets.diagnostic_dialog import StartupDiagnosticDialog
            dlg = StartupDiagnosticDialog(issues, parent=self)
            dlg.exec()

    @Slot()
    def _on_startup_probe_finished(self) -> None:
        worker = self._startup_probe_worker
        self._startup_probe_worker = None
        if worker is not None:
            worker.deleteLater()

    def _build_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Top bar with brand mark (left) + GPU/VRAM info (right)
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(12, 6, 12, 6)

        self._current_accent = "#2CC350"
        self._brand_label = QLabel(
            '<span style="color:#FFF203;">CORRIDOR</span>'
            '<span style="color:#2CC350;">KEY</span>'
            '<span style="color:#FFF203;font-size:11px;"> 2.0</span>'
        )
        self._brand_label.setObjectName("brandMark")
        top_bar.addWidget(self._brand_label)
        top_bar.addStretch()

        # GPU info (right side of brand bar)
        self._gpu_label = QLabel("")
        self._gpu_label.setObjectName("gpuName")
        self._gpu_label.setToolTip(self.tr("Detected GPU used for inference"))
        top_bar.addWidget(self._gpu_label)

        self._vram_label = QLabel(self.tr("VRAM"))
        self._vram_label.setStyleSheet("color: #808070; font-size: 10px;")
        top_bar.addWidget(self._vram_label)

        self._vram_bar = QProgressBar()
        self._vram_bar.setObjectName("vramMeter")
        self._vram_bar.setFixedWidth(80)
        self._vram_bar.setFixedHeight(8)
        self._vram_bar.setTextVisible(False)
        self._vram_bar.setRange(0, 100)
        self._vram_bar.setValue(0)
        self._vram_bar.setToolTip(self.tr("GPU video memory usage — updates during inference"))
        top_bar.addWidget(self._vram_bar)

        self._vram_text = QLabel("")
        self._vram_text.setObjectName("vramText")
        self._vram_text.setMinimumWidth(70)
        self._vram_text.setToolTip(self.tr("Current VRAM used / total available"))
        top_bar.addWidget(self._vram_text)

        main_layout.addLayout(top_bar)

        # Stacked widget: page 0 = welcome, page 1 = workspace
        self._stack = QStackedWidget()

        # Page 0 — Welcome/drop screen
        self._welcome = WelcomeScreen(self._recent_store)
        self._welcome.folder_selected.connect(self._on_welcome_folder)
        self._welcome.files_selected.connect(self._on_welcome_files)
        self._welcome.recent_project_opened.connect(self._on_recent_project_opened)
        self._stack.addWidget(self._welcome)

        # Page 1 — Workspace (vertical splitter: top panels + I/O tray)
        workspace = QWidget()
        ws_layout = QVBoxLayout(workspace)
        ws_layout.setContentsMargins(0, 0, 0, 0)
        ws_layout.setSpacing(0)

        # Vertical splitter: top = viewer+params, bottom = I/O tray
        self._vsplitter = QSplitter(Qt.Vertical)

        # Horizontal splitter: dual viewer | param panel
        self._splitter = QSplitter(Qt.Horizontal)

        # Left — Dual Viewer (input + output side by side)
        self._dual_viewer = DualViewerPanel()
        self._splitter.addWidget(self._dual_viewer)

        # Right — Parameter Panel
        self._param_panel = ParameterPanel()
        self._splitter.addWidget(self._param_panel)

        # Viewer fills, param panel starts at its full content width so
        # nothing is clipped on the right
        self._splitter.setSizes([920, max(280, self._param_panel.minimumWidth())])
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, False)

        self._vsplitter.addWidget(self._splitter)

        # Bottom — I/O Tray Panel
        self._io_tray = IOTrayPanel(self._clip_model)
        self._vsplitter.addWidget(self._io_tray)

        # Top fills by default, tray can be dragged up freely
        self._vsplitter.setStretchFactor(0, 1)
        self._vsplitter.setStretchFactor(1, 0)
        self._vsplitter.setCollapsible(0, False)
        self._vsplitter.setCollapsible(1, False)
        self._vsplitter.setSizes([600, 140])

        ws_layout.addWidget(self._vsplitter, 1)

        # Queue panel — floating overlay on the left edge of the workspace
        self._queue_panel = QueuePanel(self._service.job_queue, parent=workspace)
        self._queue_panel.raise_()

        self._stack.addWidget(workspace)

        # Keep the queue panel sized to the workspace height
        self._workspace = workspace

        main_layout.addWidget(self._stack, 1)

    def _build_status_bar(self) -> None:
        self._status_bar = StatusBar()
        self.centralWidget().layout().addWidget(self._status_bar)
        # Hidden until user opens a project (welcome screen has no use for it)
        self._status_bar.hide()

    def _connect_signals(self) -> None:
        # Clip model — detect clip removal
        self._clip_model.clip_count_changed.connect(self._on_clip_count_changed)

        # I/O tray — clip selection, import, drag-and-drop
        self._io_tray.clip_clicked.connect(self._on_tray_clip_clicked)
        self._io_tray.selection_changed.connect(self._on_selection_changed)
        self._io_tray.clips_dir_changed.connect(self._on_tray_folder_imported)
        self._io_tray.files_imported.connect(self._on_tray_files_imported)
        self._io_tray.sequence_folder_imported.connect(self._on_sequence_folder_imported)
        self._io_tray.image_files_dropped.connect(self._on_image_files_dropped)
        self._io_tray.extract_requested.connect(self._on_extract_requested)
        self._io_tray.export_video_requested.connect(self._on_export_video)
        self._io_tray.export_nuke_requested.connect(self._on_export_nuke_script)
        self._io_tray.export_fcpxml_requested.connect(self._on_export_fcpxml)
        self._io_tray.export_ae_requested.connect(self._on_export_ae_jsx)
        self._io_tray.export_fusion_requested.connect(self._on_export_fusion_comp)
        self._io_tray.reset_in_out_requested.connect(self._on_reset_all_in_out)

        # Status bar buttons
        self._status_bar.run_clicked.connect(self._on_run_inference)
        self._status_bar.extract_clicked.connect(self._on_extract_current_clip)
        self._status_bar.resume_clicked.connect(self._on_resume_inference)
        self._status_bar.stop_clicked.connect(self._on_stop_inference)
        self._status_bar.live_render_toggled.connect(self._on_live_preview_toggled)
        self._status_bar.ram_preview_requested.connect(self._on_ram_preview_requested)

        # GPU worker signals
        self._gpu_worker.progress.connect(self._on_worker_progress)
        self._gpu_worker.preview_ready.connect(self._on_worker_preview)
        self._gpu_worker.clip_finished.connect(self._on_worker_clip_finished)
        self._gpu_worker.warning.connect(self._on_worker_warning)
        self._gpu_worker.status_update.connect(self._on_worker_status)
        self._gpu_worker.error.connect(self._on_worker_error)
        self._gpu_worker.queue_empty.connect(self._on_queue_empty)
        self._gpu_worker.reprocess_result.connect(self._on_reprocess_result)

        # GPU monitor -> top bar widgets (not status bar)
        self._gpu_monitor.vram_updated.connect(self._update_vram)
        self._gpu_monitor.gpu_name.connect(self._set_gpu_name)

        # Queue panel cancel & pause signals
        self._queue_panel.cancel_job_requested.connect(self._on_cancel_job)
        self._queue_panel.pause_toggled.connect(self._on_queue_pause_toggled)

        # Parameter panel — wire BiRefNet / SAM 3.1 Track Mask / Chroma Key
        self._param_panel.birefnet_requested.connect(self._on_run_birefnet)
        self._param_panel.birefnet_one_frame_requested.connect(self._on_run_birefnet_one_frame)
        self._param_panel.track_masks_requested.connect(self._on_track_masks)
        self._param_panel.sam3_one_frame_requested.connect(self._on_run_sam3_one_frame)
        self._param_panel.clear_mask_requested.connect(self._on_clear_mask_and_hints)
        self._param_panel.import_alpha_requested.connect(self._on_import_alpha)
        self._param_panel.chroma_key_requested.connect(self._on_run_chroma_key)
        self._param_panel.chroma_key_preview.connect(self._on_chroma_key_param_changed)
        self._param_panel.eyedropper_requested.connect(self._on_eyedropper_toggle)
        self._param_panel.screen_color_changed.connect(self._on_screen_color_changed)

        # Multi-subject SAM 3.1, Polygon Matte authoring, and Post-Track Mask Fix tools
        self._param_panel.active_object_changed.connect(self._set_active_object_id)
        self._param_panel.poly_garbage_requested.connect(self._toggle_poly_garbage)
        self._param_panel.poly_core_requested.connect(self._toggle_poly_core)
        self._param_panel.poly_invert_changed.connect(self._on_poly_invert_changed)
        self._param_panel.poly_feather_changed.connect(self._on_poly_feather_changed)
        self._param_panel.poly_curve_changed.connect(self._on_poly_curve_changed)
        self._param_panel.poly_clear_requested.connect(self._clear_polygon_mattes)
        self._param_panel.mask_paint_fg_requested.connect(self._toggle_mask_paint_fg)
        self._param_panel.mask_paint_bg_requested.connect(self._toggle_mask_paint_bg)
        self._param_panel.mask_interp_range_requested.connect(self._interpolate_mask_range)
        self._param_panel.lut_changed.connect(self._on_lut_changed)
        self._param_panel.auto_clean_plate_requested.connect(self._on_auto_clean_plate_requested)

        self._dual_viewer.pixel_hovered.connect(self._status_bar.update_pixel_info)
        self._dual_viewer.custom_plate_requested.connect(self._on_custom_plate_requested)
        self._dual_viewer.polygon_completed.connect(self._on_polygon_completed)
        self._dual_viewer.polygon_vertex_moved.connect(self._on_polygon_vertex_moved)
        self._dual_viewer.mask_paint_stroke.connect(self._on_mask_paint_stroke)

        # Eyedropper color sampling from either viewport (including wipe overlay)
        self._dual_viewer.input_viewer._split_view.color_sampled.connect(self._on_color_sampled)
        self._dual_viewer.output_viewer._split_view.color_sampled.connect(self._on_color_sampled)
        self._dual_viewer._wipe_overlay.color_sampled.connect(self._on_color_sampled)
        # Live preview during eyedropper drag
        self._dual_viewer.input_viewer._split_view.color_preview.connect(self._on_color_preview)
        self._dual_viewer.output_viewer._split_view.color_preview.connect(self._on_color_preview)
        self._dual_viewer._wipe_overlay.color_preview.connect(self._on_color_preview)
        # Full sample list for multi-reference keying
        self._dual_viewer.input_viewer._split_view.screen_samples_ready.connect(self._on_screen_samples)
        self._dual_viewer.output_viewer._split_view.screen_samples_ready.connect(self._on_screen_samples)
        self._dual_viewer._wipe_overlay.screen_samples_ready.connect(self._on_screen_samples)

        # Share annotation and holdout models between viewports
        self._dual_viewer.setup_shared_annotations()
        self._dual_viewer.setup_shared_holdout()

        # Stroke finished -> context-aware save (holdout vs SAM 3.1)
        def _on_stroke_finished():
            mode = self._dual_viewer.input_viewer.annotation_mode
            if mode == "holdout":
                self._auto_save_holdout()
                self._schedule_chroma_key_preview()
            else:
                self._auto_save_annotations()
                self._update_annotation_info()
                clip = self._current_clip
                iv = self._dual_viewer.input_viewer
                if clip is not None and iv.current_stem_index >= 0:
                    self._invalidate_sam3_preview_cache_for_frame(
                        clip.root_path, iv.current_stem_index
                    )
                self._refresh_live_sam3_preview(force_recompute=True)
            if hasattr(self._param_panel, "set_clear_mask_enabled"):
                self._param_panel.set_clear_mask_enabled(True)

        def _on_timeline_frame_changed(frame_idx: int):
            self._refresh_live_sam3_preview(frame_idx)
            if self._param_panel.live_preview_enabled and self._current_clip is not None:
                cur_root = self._current_clip.root_path
                lr = getattr(self, "_last_live_render", None)
                if lr is None or lr[0] != cur_root or lr[1] != frame_idx:
                    self._last_live_render = None
                    self._reprocess_timer.start()

        self._dual_viewer.frame_changed.connect(_on_timeline_frame_changed)
        self._dual_viewer.output_backdrop_changed.connect(self._on_output_backdrop_changed)

        self._dual_viewer.input_viewer._split_view.stroke_finished.connect(
            _on_stroke_finished
        )
        self._dual_viewer.output_viewer._split_view.stroke_finished.connect(
            _on_stroke_finished
        )

        # Parameter panel — live reprocess (debounced, Codex: coalesce stale)
        self._param_panel.params_changed.connect(self._on_params_changed)
        self._param_panel.parallel_frames_changed.connect(
            lambda n: self._gpu_worker.set_max_workers(n)
        )
        self._param_panel._live_preview.toggled.connect(self._on_live_preview_toggled)

        # Sync IO tray divider with dual viewer splitter
        self._dual_viewer._viewer_splitter.splitterMoved.connect(self._sync_io_divider)
        self._dual_viewer.output_mode_changed.connect(self._on_output_mode_changed)

        # Reposition queue overlay when vertical splitter is dragged
        self._vsplitter.splitterMoved.connect(self._position_queue_panel)

        # Scrubber in/out marker drags -> persist + refresh button state
        scrubber = self._dual_viewer._scrubber
        scrubber.in_point_changed.connect(lambda _: self._persist_in_out())
        scrubber.out_point_changed.connect(lambda _: self._persist_in_out())
        scrubber.range_cleared.connect(self._clear_in_out)

        # Extract worker signals
        self._extract_worker.progress.connect(self._on_extract_progress)
        self._extract_worker.finished.connect(self._on_extract_finished)
        self._extract_worker.error.connect(self._on_extract_error)

    # ── GPU Header ──

    @Slot()
    def _on_custom_plate_requested(self) -> None:
        """Open file picker for a Custom Backdrop Plate (image or video)."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select Custom Background Plate"),
            self._clips_dir or "",
            self.tr("Images & Videos (*.png *.jpg *.jpeg *.exr *.tif *.tiff *.mp4 *.mov *.mkv)"),
        )
        if not path:
            return
        from ui.preview.display_transform import set_custom_backdrop_plate
        if set_custom_backdrop_plate(path):
            self._dual_viewer.output_viewer.refresh_current_frame()
            self._on_output_backdrop_changed("custom")
            self._status_bar.set_message(self.tr("Loaded custom backdrop plate: %s") % os.path.basename(path))

    @Slot(bool, str)
    def _on_lut_changed(self, enabled: bool, lut_path: str) -> None:
        from backend.color_management import set_display_lut, set_display_lut_enabled
        if lut_path:
            set_display_lut(lut_path)
        set_display_lut_enabled(bool(enabled and lut_path))
        self._dual_viewer.output_viewer._split_view.update()
        self._dual_viewer._wipe_overlay.update()

    @Slot()
    def _on_auto_clean_plate_requested(self) -> None:
        clip = getattr(self, "_current_clip", None)
        if clip is None or clip.input_asset is None:
            return
        import cv2
        from backend.frame_io import imread_unicode, imwrite_unicode, read_video_frame_at
        from backend.keying_post import generate_auto_clean_plate

        fidx = max(0, int(self._dual_viewer.current_stem_index))
        frame_rgb = None
        stem = f"frame_{fidx:06d}"
        if clip.input_asset.asset_type == "sequence":
            files = clip.input_asset.get_frame_files()
            if 0 <= fidx < len(files):
                stem = os.path.splitext(files[fidx])[0]
                bgr = imread_unicode(os.path.join(clip.input_asset.path, files[fidx]), cv2.IMREAD_COLOR)
                if bgr is not None:
                    frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = read_video_frame_at(clip.input_asset.path, fidx)
        if frame_rgb is None:
            return

        mask_u8 = None
        for mfolder in ("SamMaskHint", "AlphaHint"):
            cand = os.path.join(clip.root_path, mfolder, f"{stem}.png")
            if os.path.isfile(cand):
                mask_u8 = imread_unicode(cand, cv2.IMREAD_GRAYSCALE)
                if mask_u8 is not None:
                    break
        if mask_u8 is None:
            self._status_bar.set_message(self.tr("Generate or paint a mask on the current frame first."))
            return

        plate_rgb = generate_auto_clean_plate(frame_rgb, mask_u8)
        out_cp = os.path.join(clip.root_path, "clean_plate_auto.png")
        imwrite_unicode(out_cp, cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2BGR))
        self._param_panel.set_clean_plate_path(out_cp)
        self._param_panel._clean_plate_group.setChecked(True)
        self._param_panel._emit_changed()
        self._status_bar.set_message(self.tr("Generated auto clean plate: clean_plate_auto.png"))

    @Slot(dict)
    def _update_vram(self, info: dict) -> None:
        """Update VRAM/Memory meter in the top bar."""
        self._last_vram_info = info  # stash for Report Issue dialog
        if not info.get("available"):
            self._vram_text.setText(self.tr("No GPU"))
            self._vram_bar.setValue(0)
            return

        # Apple Silicon uses unified memory, not dedicated VRAM
        name = info.get("name", "")
        if name.startswith("Apple"):
            self._vram_label.setText(self.tr("Memory"))
            self._vram_bar.setToolTip(self.tr("Unified memory usage — CPU and GPU share the same pool"))
            self._vram_text.setToolTip(self.tr("Current unified memory used / total available"))
        pct = info.get("usage_pct", 0)
        used = info.get("used_gb", 0)
        total = info.get("total_gb", 0)
        self._vram_bar.setValue(int(pct))
        self._vram_text.setText(f"{used:.1f}/{total:.1f}GB")

    @Slot(str)
    def _set_gpu_name(self, name: str) -> None:
        """Display GPU name in the top bar."""
        short = name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
        self._gpu_label.setText(short)

    def _show_toast(self, text: str, duration_ms: int = 4000) -> None:
        """Show a non-blocking notification toast at the bottom of the window."""
        _Toast(self, text, duration_ms)

    def _refresh_input_thumbnail(
        self,
        clip: ClipEntry,
        *,
        input_is_linear: bool | None = None,
    ) -> None:
        """Regenerate the input-strip thumbnail using the viewer's input decode."""
        if clip.input_asset is None:
            return
        self._thumb_gen.generate(
            clip.name,
            clip.root_path,
            kind="input",
            input_path=clip.input_asset.path,
            asset_type=clip.input_asset.asset_type,
            input_exr_is_linear=(
                clip.should_default_input_linear()
                if input_is_linear is None
                else input_is_linear
            ),
        )

    def _refresh_export_thumbnail(self, clip: ClipEntry) -> None:
        """Regenerate the export-strip thumbnail using the current output mode."""
        if clip.state != ClipState.COMPLETE and not getattr(clip, "has_exports", False):
            return
        self._thumb_gen.generate(
            clip.name,
            clip.root_path,
            kind="export",
            export_mode=self._dual_viewer.current_output_mode,
        )

    def paintEvent(self, event) -> None:
        """Paint dithered diagonal gradient background.

        Uses a cached QImage with per-pixel noise dithering to eliminate
        banding on subtle dark gradients. Diagonal: lower-left (darker)
        to upper-right (lighter).
        """
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return

        # Cache the gradient image, regenerate only on resize
        if (self._bg_cache is None
                or self._bg_cache.width() != w
                or self._bg_cache.height() != h):
            self._bg_cache = self._render_dithered_gradient(w, h)

        painter = QPainter(self)
        painter.drawImage(0, 0, self._bg_cache)
        painter.end()

    def _render_dithered_gradient(self, w: int, h: int) -> QImage:
        """Render a dithered diagonal gradient to a QImage."""
        img = np.empty((h, w, 3), dtype=np.float32)

        # Diagonal parameter: 0 at lower-left, 1 at upper-right
        ys = np.linspace(1, 0, h).reshape(-1, 1)  # top=0 -> 1, bottom=1 -> 0
        xs = np.linspace(0, 1, w).reshape(1, -1)
        diag = (xs + ys) * 0.5  # 0..1 diagonal

        # Color range: dark edge (10, 9, 0) to lighter center (22, 21, 2)
        r = 10.0 + diag * 12.0
        g = 9.0 + diag * 12.0
        b = 0.0 + diag * 2.0

        img[..., 0] = r
        img[..., 1] = g
        img[..., 2] = b

        # Add triangular dithering noise (+/-0.5) to break banding
        rng = np.random.default_rng(42)  # fixed seed for stability
        noise = rng.uniform(-0.5, 0.5, (h, w, 3)).astype(np.float32)
        img += noise

        # Clamp and convert to uint8
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)

        # numpy RGB -> QImage
        bytes_per_line = w * 3
        qimg = QImage(img_u8.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return qimg.copy()  # deep copy so numpy buffer can be freed


    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_queue_panel()
        self._position_update_btn()

    @Slot()
    def _position_queue_panel(self, *_args) -> None:
        """Keep the floating queue panel sized to the viewer area height.

        The panel must never cover the coverage bar / scrubber strip at the
        bottom of the viewer area, so its bottom edge is clamped to the
        coverage bar's top.
        """
        if hasattr(self, '_workspace') and hasattr(self, '_queue_panel'):
            # Use the top section height (viewer+params) not the full workspace
            sizes = self._vsplitter.sizes()
            h = sizes[0] if sizes else self._workspace.height()
            try:
                # Clamp to the scrubber center strip (coverage bar + slider +
                # marker overlay). The in/out bracket tips paint from the very
                # top of this strip, so using the coverage bar itself would
                # still clip them.
                strip = self._dual_viewer._scrubber._center
                if strip.isVisible():
                    strip_top = strip.mapTo(self._workspace, strip.rect().topLeft()).y()
                    if 0 < strip_top < h:
                        h = strip_top
            except (AttributeError, RuntimeError):
                pass  # before first layout or scrubber not built yet
            self._queue_panel.setFixedHeight(h)
            self._queue_panel.move(0, 0)
            self._queue_panel.raise_()

    # ── Global drag-and-drop (accepts drops anywhere on window) ──

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            from backend.project import is_video_file, is_image_file
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isdir(path) or is_video_file(path) or is_image_file(path):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event) -> None:
        from backend.project import is_video_file, is_image_file, folder_has_image_sequence
        folders = []
        video_files = []
        image_files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                folders.append(path)
            elif os.path.isfile(path):
                if is_video_file(path):
                    video_files.append(path)
                elif is_image_file(path):
                    image_files.append(path)

        if folders:
            folder = folders[0]
            if folder_has_image_sequence(folder):
                self._on_sequence_folder_imported(folder)
            else:
                self._on_tray_folder_imported(folder)
        elif video_files and not image_files:
            self._on_tray_files_imported(video_files)
        elif image_files:
            self._on_image_files_dropped(image_files)
        elif video_files:
            self._on_tray_files_imported(video_files)

    def closeEvent(self, event) -> None:
        """Clean shutdown — auto-save session, stop workers, unload engines."""
        if self._skip_shutdown_cleanup:
            super().closeEvent(event)
            return
        # Save annotation and holdout strokes before closing
        self._auto_save_annotations()
        self._auto_save_holdout()
        # Auto-save session on close
        if self._clips_dir:
            try:
                self._on_save_session()
            except Exception:
                pass

        self._gpu_monitor.stop()
        for tmr_attr in (
            "_reprocess_timer",
            "_sam3_preview_timer",
            "_live_asset_refresh_timer",
            "_autosave_timer",
            "_ram_play_timer",
        ):
            tmr = getattr(self, tmr_attr, None)
            if tmr is not None:
                try:
                    tmr.stop()
                except Exception:
                    pass
        try:
            self._service.job_queue.cancel_all()
        except Exception:
            pass
        ram_worker = getattr(self, "_ram_worker", None)
        if ram_worker is not None and ram_worker.isRunning():
            ram_worker.requestInterruption()
            ram_worker.wait(2000)
        probe_worker = getattr(self, "_startup_probe_worker", None)
        if probe_worker is not None and probe_worker.isRunning():
            probe_worker.requestInterruption()
            probe_worker.wait(2000)
        sam3_worker = getattr(self, "_sam3_preview_worker", None)
        if sam3_worker is not None and sam3_worker.isRunning():
            sam3_worker.requestInterruption()
            sam3_worker.wait(2000)
        update_thread = getattr(self, "_update_thread", None)
        if update_thread is not None and update_thread.isRunning():
            update_thread.requestInterruption()
            update_thread.wait(2000)
        export_worker = getattr(self, "_export_worker", None)
        if export_worker is not None and export_worker.isRunning():
            export_worker.request_cancel()
            export_worker.wait(3000)
        if self._extract_worker.isRunning():
            self._extract_worker.stop()
            self._extract_worker.wait(5000)
        if self._gpu_worker.isRunning():
            self._gpu_worker.stop()
            self._gpu_worker.wait(5000)
        self._service.unload_engines()
        if self._debug_console is not None:
            self._debug_console.close_permanently()
        super().closeEvent(event)
