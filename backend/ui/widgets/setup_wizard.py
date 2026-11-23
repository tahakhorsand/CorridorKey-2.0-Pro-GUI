"""First-launch setup wizard — download model checkpoints with progress.



Shown automatically when the required CorridorKey checkpoint is missing.

Provides the same model selection as install.bat / setup scripts in

a PySide6 GUI with checkboxes and progress bars.

"""

from __future__ import annotations



import importlib.util

import logging

import os

import platform

import sys

import traceback

from pathlib import Path



from PySide6.QtWidgets import (

    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,

    QPushButton, QProgressBar, QWidget,

    QFrame, QLineEdit, QFileDialog,

)

from PySide6.QtCore import Qt, Signal, QThread, QObject

from PySide6.QtGui import QFont



logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------

# Path helpers

# ---------------------------------------------------------------------------



def _project_root() -> Path:

    if getattr(sys, "frozen", False):

        return Path(sys._MEIPASS)

    return Path(__file__).resolve().parent.parent.parent





def _default_install_dir() -> Path:

    """Default model location: where the app is running from.



    Works the same for Inno installs, portable zips, git clones, and the

    source release zip. macOS frozen builds are the one exception since

    .app bundles often sit in read-only /Applications and writing next to

    them is a permissions fight, so we keep the platform user-data path.

    """

    if sys.platform == "darwin" and getattr(sys, "frozen", False):

        preferred = Path.home() / "Library" / "Application Support" / "CorridorKey 2.0"

        legacy = Path.home() / "Library" / "Application Support" / "EZ-CorridorKey"

        return legacy if (not preferred.exists() and legacy.exists()) else preferred

    if getattr(sys, "frozen", False):

        return Path(sys.executable).resolve().parent

    # Source install (git clone or GitHub source zip): the repo root.

    return Path(__file__).resolve().parent.parent.parent





def _data_root() -> Path:

    """Writable data root where models are stored.



    Dev mode: project root. Frozen: delegates to backend.project.get_data_dir().

    """

    if not getattr(sys, "frozen", False):

        return _project_root()

    try:

        from backend.project import get_data_dir

        return Path(get_data_dir())

    except ImportError:

        return _default_install_dir()





def _checkpoint_dir() -> Path:
    return _data_root() / "weights"





def _hf_hub_cache_dir() -> Path:

    """Hugging Face hub cache location, respecting HF_HOME / HF_HUB_CACHE env vars.



    SAM 3.1 checkpoints are downloaded into this shared cache regardless of the

    user's chosen install path, so the wizard detects them here — same location

    across fresh installs, reinstalls, and folder changes.

    """

    override = os.environ.get("HF_HUB_CACHE")

    if override:

        return Path(override)

    hf_home = os.environ.get("HF_HOME")

    if hf_home:

        return Path(hf_home) / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"





def _saved_install_path() -> Path:

    """Return the user's last-saved install path, falling back to default."""

    default_dir = _default_install_dir()

    # If the local project directory already contains CorridorKey 2.0 weights, always prefer it over stale paths

    if (default_dir / "weights" / "corridorkey_v2.safetensors").is_file():

        try:

            from PySide6.QtCore import QSettings

            saved = QSettings().value("app/install_path", "", type=str)

            if saved:

                saved_p = Path(saved)

                if (saved_p / "weights" / "corridorkey_v2.safetensors").is_file():

                    return saved_p

        except Exception:

            pass

        return default_dir

    try:

        from PySide6.QtCore import QSettings

        saved = QSettings().value("app/install_path", "", type=str)

        if saved and Path(saved).is_dir():

            return Path(saved)

    except Exception:

        pass

    return default_dir





def _current_version() -> str:

    """Return central app version (with pyproject.toml fallback)."""

    try:

        from backend import APP_VERSION

        return APP_VERSION

    except Exception:

        pass

    try:

        import tomllib

    except Exception:

        return "2.0.0"

    candidates: list[Path] = []

    if getattr(sys, "frozen", False):

        candidates.append(Path(sys._MEIPASS) / "pyproject.toml")

    candidates.append(_project_root() / "pyproject.toml")

    for path in candidates:

        try:

            with open(path, "rb") as fh:

                return tomllib.load(fh)["project"]["version"]

        except Exception:

            continue

    return "2.0.0"





def detect_installed_models(install_path: Path | None = None) -> dict[str, bool]:

    """Return {model_key: present} for every model in the wizard catalogue.



    Models that live under the install path (CorridorKey, MLX, GVM, VideoMaMa,

    MatAnyone2) are resolved against ``install_path``. Models that live in the

    Hugging Face hub cache (SAM 3.1) are resolved against the HF cache regardless

    of the install path — that's how hf_hub_download stores them without a

    local_dir override.



    Detection is intentionally decoupled from scripts/setup_models.py because

    that module bakes ``PROJECT_ROOT`` at import time, so re-scanning after a

    path change would see stale paths.

    """

    root = Path(install_path) if install_path else _saved_install_path()

    hf = _hf_hub_cache_dir()



    results: dict[str, bool] = {}



    # CorridorKey core/blue/MLX weights are bundled into the app by the

    # installers, but they live in the read-only bundle dir (_MEIPASS in a

    # frozen build), not the writable install path. Check both, mirroring the

    # loader's search order in weights, so the wizard does

    # not offer to download core models that already ship in the bundle.

    ck_dir = root / "weights"
    ck_bundled = _project_root() / "weights"



    def _ck_has(*names: str) -> bool:

        for d in (ck_dir, ck_bundled):

            if any((d / n).is_file() for n in names):

                return True

        return False



    v2_safetensors = (root / "weights" / "corridorkey_v2.safetensors").is_file()

    foundation_stats = (root / "weights" / "foundation_stats.safetensors").is_file()

    cradio_dir = root / "vfms" / "cradio" / "C-RADIOv4-SO400M"
    cradio_present = cradio_dir.is_dir() and any(
        (cradio_dir / f).is_file() for f in ("model.safetensors", "config.json")
    )
    vfms_present = (
        (root / "vfms" / "rvm" / "pretrain_rvm_small16_256_204031069.npz").is_file()
        and (root / "vfms" / "moge" / "model.pt").is_file()
        and cradio_present
    )



    results["corridorkey"] = v2_safetensors and foundation_stats

    results["foundation_stats"] = foundation_stats

    results["vfms"] = vfms_present



    # CorridorKey Blue — specific filename check

    results["corridorkey-blue"] = _ck_has("CorridorKeyBlue_1.0.pth")



    # CorridorKey MLX — built-in engine needs both green and blue weights

    results["corridorkey-mlx"] = _ck_has("CorridorKey.mlx.safetensors") and _ck_has(

        "CorridorKeyBlue_1.0.mlx.safetensors"

    )



    # SAM 3.1 Native INT8 ConvRot + tracker head stored under weights/sam3/
    sam_dir = root / "weights" / "sam3"
    sam_bundled = _project_root() / "weights" / "sam3"
    results["sam3"] = any(
        (d / "sam3.1-INT8-ConvRot.safetensors").is_file()
        and (d / "sam3.1-tracker-head.safetensors").is_file()
        for d in (sam_dir, sam_bundled)
    )



    # GVM & MatAnyone2 decommissioned

    results["gvm"] = True

    results["matanyone2"] = True



    # All 9 BiRefNet variants — use same snapshot check as runtime (model.safetensors + config.json)
    from modules.BiRefNetModule.wrapper import resolve_birefnet_snapshot
    brn_dir = root / "modules" / "BiRefNetModule" / "checkpoints" / "BiRefNet-matting"
    try:
        brn_local = (
            brn_dir.is_dir()
            and (brn_dir / "config.json").is_file()
            and any(p.suffix == ".safetensors" for p in brn_dir.iterdir() if p.is_file())
        )
    except OSError:
        brn_local = False
    for bkey, (_, brepo, _, _) in _BIREFNET_DOWNLOAD_SPECS.items():
        present = resolve_birefnet_snapshot(brepo, hf) is not None
        if bkey == "birefnet":
            present = present or brn_local
        results[bkey] = present



    # VideoMaMa — needs BOTH the VideoMaMa weights and the SVD base model

    vm_root = root / "VideoMaMaInferenceModule" / "checkpoints"

    vm_main = vm_root / "VideoMaMa" / "diffusion_pytorch_model.safetensors"

    vm_base = vm_root / "stable-video-diffusion-img2vid-xt" / "model_index.json"

    results["videomama"] = False



    return results





def _sam3_package_available() -> bool:

    """True when the sam3 Python package is importable in this environment.



    Probes sam3.build_sam because that is the exact module the tracker

    imports at runtime. A partial install can leave the bare top-level

    package importable while the submodule the runtime needs is missing.

    """

    try:

        return importlib.util.find_spec("sam3.build_sam") is not None

    except Exception:

        return False





def _hf_cache_has(repo_id: str, cache_dir: Path) -> bool:

    """True when any snapshot for ``repo_id`` exists under the HF hub cache.



    huggingface_hub stores repos as ``models--<org>--<name>/snapshots/<rev>/``.

    We don't probe for a specific filename because users may have downloaded

    different SAM 3.1 variants (small / base-plus / large) and we just want to

    know whether the shared cache already has a usable checkpoint.

    """

    try:

        if not cache_dir.is_dir():

            return False

        repo_folder = "models--" + repo_id.replace("/", "--")

        snapshots = cache_dir / repo_folder / "snapshots"

        if not snapshots.is_dir():

            return False

        for snap in snapshots.iterdir():

            if snap.is_dir() and any(snap.iterdir()):

                return True

    except OSError:

        pass

    return False





# ---------------------------------------------------------------------------

# Model catalogue

# ---------------------------------------------------------------------------



_IS_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"



_BIREFNET_DOWNLOAD_SPECS: dict[str, tuple[str, str, str, str]] = {
    "birefnet": ("BiRefNet Matting (Default)", "ZhengPeng7/BiRefNet-matting", "885 MB", "Best for hair, fur & transparent edges (1024px)"),
    "birefnet-general": ("BiRefNet General", "ZhengPeng7/BiRefNet", "885 MB", "General-purpose foreground segmentation (1024px)"),
    "birefnet-hr-matting": ("BiRefNet HR Matting", "ZhengPeng7/BiRefNet_HR-Matting", "885 MB", "High-resolution fine matting (2048px)"),
    "birefnet-hr": ("BiRefNet HR General", "ZhengPeng7/BiRefNet_HR", "885 MB", "High-resolution general segmentation (2048px)"),
    "birefnet-dynamic-matting": ("BiRefNet Dynamic Matting", "ZhengPeng7/BiRefNet_dynamic-matting", "885 MB", "Dynamic resolution matting (1024px)"),
    "birefnet-dynamic": ("BiRefNet Dynamic General", "ZhengPeng7/BiRefNet_dynamic", "885 MB", "Dynamic resolution general segmentation (1024px)"),
    "birefnet-lite-matting": ("BiRefNet Lite Matting", "ZhengPeng7/BiRefNet_lite-matting", "178 MB", "Lightweight matting for low-VRAM GPUs (1024px)"),
    "birefnet-lite-2k": ("BiRefNet Lite 2K", "ZhengPeng7/BiRefNet_lite-2K", "178 MB", "Lightweight high-res segmentation (2048px)"),
    "birefnet-lite": ("BiRefNet Lite General", "ZhengPeng7/BiRefNet_lite", "178 MB", "Fastest lightweight general segmentation (1024px)"),
}


MODELS: list[dict] = [

    {

        "key": "corridorkey",

        "label": "CorridorKey 2.0 Foundation Model",

        "size": "2.66 GB",

        "required": True,

        "description": "Unified deep visual foundation model for native-resolution keying (+ normalization stats)",

        "default_checked": True,

    },

    {

        "key": "vfms",

        "label": "Vision Foundation Models (RVM, MoGe, C-RADIO)",

        "size": "1.8 GB",

        "required": True,

        "description": "Geometric, matting, and visual feature extractors",

        "default_checked": True,

    },

    {

        "key": "sam3",

        "label": "SAM 3.1 Tracker (INT8 ConvRot)",

        "size": "1.11 GB",

        "required": False,

        "description": "Interactive point/box mask tracking & multi-frame propagation",

        "default_checked": False,

    },

    *[
        {
            "key": bkey,
            "label": blabel,
            "size": bsize,
            "required": False,
            "description": bdesc,
            "default_checked": False,
        }
        for bkey, (blabel, _brepo, bsize, bdesc) in _BIREFNET_DOWNLOAD_SPECS.items()
    ],

]





def _required_model_keys() -> list[str]:

    return [m["key"] for m in MODELS if m.get("required")]





def needs_setup() -> bool:

    """True only when a required model is missing at the current install path."""

    try:

        install_path = _saved_install_path()

        present = detect_installed_models(install_path)

        for key in _required_model_keys():

            if not present.get(key, False):

                return True

    except Exception:

        logger.exception("detect_installed_models failed during needs_setup()")

        return False

    return False





def has_required_models() -> bool:

    """True when every required model is present at the current install path.



    Used by main.py to decide whether to exit after the wizard closes: if the

    user cancelled without installing a required model, we still can't launch.

    A version-bump-only trigger with all requireds present lets the user X out

    and keep going.

    """

    try:

        present = detect_installed_models(_saved_install_path())

    except Exception:

        logger.exception("detect_installed_models failed during has_required_models()")

        return False

    return all(present.get(k, False) for k in _required_model_keys())





def mark_setup_seen() -> None:

    """Persist the current version so the version-bump trigger doesn't fire again.



    Called from main.py after the wizard runs (or is skipped) so that the next

    launch on the same version does not re-show the wizard. A later version

    upgrade bumps the key and re-triggers it once.

    """

    try:

        from PySide6.QtCore import QSettings

        QSettings().setValue("app/version_last_seen", _current_version())

    except Exception:

        logger.exception("Failed to persist app/version_last_seen")





# ---------------------------------------------------------------------------

# Import setup_models

# ---------------------------------------------------------------------------



def _load_setup_models():

    script_path = _project_root() / "scripts" / "download_weights.py"

    if not script_path.is_file():

        raise ImportError(f"download_weights.py not found at {script_path}")

    spec = importlib.util.spec_from_file_location("download_weights", str(script_path))

    if not spec or not spec.loader:

        raise ImportError(f"Cannot create import spec for {script_path}")

    mod = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(mod)

    return mod





# ---------------------------------------------------------------------------

# Download worker

# ---------------------------------------------------------------------------



class _DownloadWorker(QThread):

    """Download selected models sequentially in a background thread.



    IMPORTANT: This thread must be properly waited on before the parent

    dialog is destroyed, otherwise Qt will abort() on ~QThread.

    """

    model_started = Signal(str, str)          # key, label

    model_progress = Signal(str, int, int)    # key, downloaded_mb, total_mb

    model_finished = Signal(str, bool, str)   # key, success, message

    all_finished = Signal(bool)               # overall success



    def __init__(self, selected_keys: list[str], install_root: Path | None = None, parent=None):

        super().__init__(parent)

        self._selected = selected_keys

        self._install_root = install_root or _data_root()

        self._cancelled = False

        self._current_key = ""



    def cancel(self):

        self._cancelled = True



    def run(self):

        try:

            dw = _load_setup_models()

        except Exception:

            logger.exception("Failed to load download_weights")

            self.all_finished.emit(False)

            return

        # main.py enables HF_HUB_OFFLINE=1 at module import time; temporarily
        # allow online Hugging Face downloads during explicit SetupWizard runs.
        prev_hf_offline = os.environ.pop("HF_HUB_OFFLINE", None)
        prev_tf_offline = os.environ.pop("TRANSFORMERS_OFFLINE", None)
        orig_hf_download = None
        orig_snapshot_download = None
        orig_dw_hf_download = None
        orig_dw_snapshot_download = None
        hf_mod = None
        try:
            try:
                import huggingface_hub as hf_mod
                orig_hf_download = hf_mod.hf_hub_download
                orig_snapshot_download = getattr(hf_mod, "snapshot_download", None)
                worker = self

                def _progress_hf_download(*a, **kw):
                    res = orig_hf_download(*a, **kw)
                    if not worker._cancelled:
                        worker.model_progress.emit(worker._current_key, 100, 100)
                    return res

                def _progress_snapshot_download(*a, **kw):
                    res = orig_snapshot_download(*a, **kw)
                    if not worker._cancelled:
                        worker.model_progress.emit(worker._current_key, 100, 100)
                    return res

                hf_mod.hf_hub_download = _progress_hf_download
                if hasattr(dw, "hf_hub_download"):
                    orig_dw_hf_download = dw.hf_hub_download
                    dw.hf_hub_download = _progress_hf_download
                if orig_snapshot_download is not None:
                    hf_mod.snapshot_download = _progress_snapshot_download
                    if hasattr(dw, "snapshot_download"):
                        orig_dw_snapshot_download = dw.snapshot_download
                        dw.snapshot_download = _progress_snapshot_download
            except ImportError:
                hf_mod = None

            all_ok = True

            for key in self._selected:

                if self._cancelled:

                    break

                self._current_key = key

                label = next((m["label"] for m in MODELS if m["key"] == key), key)

                self.model_started.emit(key, label)

                try:

                    ok = self._download_one(dw, key)

                except Exception:

                    logger.exception("Download failed for %s", key)

                    ok = False

                if self._cancelled:

                    break

                self.model_finished.emit(key, ok, "OK" if ok else "Failed")

                if not ok:

                    all_ok = False

            self.all_finished.emit(all_ok)
        finally:
            if hf_mod is not None and orig_hf_download is not None:
                hf_mod.hf_hub_download = orig_hf_download
            if hf_mod is not None and orig_snapshot_download is not None:
                hf_mod.snapshot_download = orig_snapshot_download
            if orig_dw_hf_download is not None:
                dw.hf_hub_download = orig_dw_hf_download
            if orig_dw_snapshot_download is not None:
                dw.snapshot_download = orig_dw_snapshot_download
            if prev_hf_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = prev_hf_offline
            if prev_tf_offline is not None:
                os.environ["TRANSFORMERS_OFFLINE"] = prev_tf_offline



    def _download_one(self, dw, key: str) -> bool:
        root = Path(self._install_root)
        root.mkdir(parents=True, exist_ok=True)

        if key == "corridorkey":
            from huggingface_hub import hf_hub_download
            hf_hub_download(dw.CORRIDORKEY_REPO, "corridorkey_v2.safetensors", local_dir=root / "weights")
            hf_hub_download(dw.CORRIDORKEY_REPO, "foundation_stats.safetensors", local_dir=root / "weights")
            return (
                (root / "weights" / "corridorkey_v2.safetensors").is_file()
                and (root / "weights" / "foundation_stats.safetensors").is_file()
            )

        elif key == "vfms":
            dw.download_cradio(root)
            dw.download_moge(root)
            dw.download_rvm(root)
            return (
                (root / "vfms" / "rvm" / "pretrain_rvm_small16_256_204031069.npz").is_file()
                and (root / "vfms" / "moge" / "model.pt").is_file()
                and (root / "vfms" / "cradio" / "C-RADIOv4-SO400M").is_dir()
            )

        elif key in _BIREFNET_DOWNLOAD_SPECS:
            from huggingface_hub import snapshot_download
            _, repo_id, _, _ = _BIREFNET_DOWNLOAD_SPECS[key]
            hf_dir = _hf_hub_cache_dir()
            snapshot_download(repo_id, cache_dir=hf_dir)
            return _hf_cache_has(repo_id, hf_dir)

        elif key == "sam3":
            sam_dir = root / "weights" / "sam3"
            sam_dir.mkdir(parents=True, exist_ok=True)
            sam_file = sam_dir / "sam3.1-INT8-ConvRot.safetensors"
            if not sam_file.is_file():
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    "unsloth/sam3.1",
                    "sam3.1-INT8-ConvRot.safetensors",
                    local_dir=sam_dir,
                )
            head_file = sam_dir / "sam3.1-tracker-head.safetensors"
            if not head_file.is_file():
                from scripts.extract_sam3_tracker_head import extract_sam3_tracker_head
                extract_sam3_tracker_head(head_file)
            return sam_file.is_file() and head_file.is_file()

        return False





# ---------------------------------------------------------------------------

# Model row widget

# ---------------------------------------------------------------------------



class _ModelRow(QWidget):

    def __init__(self, model: dict, parent=None):

        super().__init__(parent)

        self.key = model["key"]

        self._required = bool(model["required"])

        self._default_checked = bool(model["default_checked"])

        self._base_desc = f"{model['description']}  ({model['size']})"



        layout = QHBoxLayout(self)

        layout.setContentsMargins(12, 6, 12, 6)

        layout.setSpacing(12)



        self.checkbox = QCheckBox()

        self.checkbox.setChecked(self._default_checked)

        if self._required:

            self.checkbox.setChecked(True)

            self.checkbox.setEnabled(False)

        layout.addWidget(self.checkbox)



        info = QVBoxLayout()

        info.setSpacing(2)

        label = QLabel(model["label"])

        label_font = QFont()

        label_font.setBold(True)

        label_font.setPointSize(13)

        label.setFont(label_font)

        label.setStyleSheet("color: #FFFFFF; background: transparent;")

        info.addWidget(label)

        self._desc_label = QLabel(self._base_desc)

        desc_font = QFont()

        desc_font.setPointSize(11)

        self._desc_label.setFont(desc_font)

        self._desc_label.setStyleSheet("color: #999999; background: transparent;")

        info.addWidget(self._desc_label)

        layout.addLayout(info, 1)



        self.progress = QProgressBar()

        self.progress.setFixedWidth(220)

        self.progress.setFixedHeight(18)

        self.progress.setRange(0, 100)

        self.progress.setValue(0)

        self.progress.setVisible(False)

        self.progress.setFormat("%v / %m MB")

        self.progress.setStyleSheet("""

            QProgressBar {

                border: 1px solid #444; border-radius: 4px;

                background: #1a1a18; text-align: center;

                color: #ccc; font-size: 11px;

            }

            QProgressBar::chunk { background: #FFF203; border-radius: 3px; }

        """)

        layout.addWidget(self.progress)



        self.status_icon = QLabel("")

        self.status_icon.setFixedWidth(32)

        self.status_icon.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        layout.addWidget(self.status_icon)



    def is_selected(self) -> bool:

        return self.checkbox.isChecked()



    def set_installed_state(self, installed: bool) -> None:

        """Reflect whether this model is already present at the current install path.



        Installed: green check, checkbox disabled+unchecked (nothing to do), label

        shows "Installed". Missing: restore default checked state and the model

        description. Required models stay checked+disabled in both states so the

        user can't accidentally deselect them.

        """

        self.progress.setVisible(False)

        if installed:

            self.checkbox.blockSignals(True)

            self.checkbox.setChecked(False)

            self.checkbox.setEnabled(False)

            self.checkbox.blockSignals(False)

            self._desc_label.setText(self._base_desc + self.tr("  \u2014 Installed"))

            self._desc_label.setStyleSheet(

                "color: #4CAF50; background: transparent;"

            )

            self.status_icon.setText("\u2714")

            self.status_icon.setStyleSheet("color: #4CAF50; font-size: 20px;")

        else:

            self.checkbox.blockSignals(True)

            self.checkbox.setChecked(self._default_checked or self._required)

            self.checkbox.setEnabled(not self._required)

            self.checkbox.blockSignals(False)

            self._desc_label.setText(self._base_desc)

            self._desc_label.setStyleSheet(

                "color: #999999; background: transparent;"

            )

            self.status_icon.setText("")

            self.status_icon.setStyleSheet("")



    def set_downloading(self):

        self.checkbox.setEnabled(False)

        self.progress.setVisible(True)

        self.progress.setRange(0, 0)

        self.progress.setFormat(self.tr("Downloading..."))

        self.status_icon.setText("")



    def set_progress(self, downloaded_mb: int, total_mb: int):

        if total_mb > 0:

            self.progress.setRange(0, total_mb)

            self.progress.setValue(min(downloaded_mb, total_mb))

            self.progress.setFormat(self.tr("%d / %d MB") % (downloaded_mb, total_mb))



    def set_finished(self, ok: bool):

        self.progress.setVisible(False)

        if ok:

            self.status_icon.setText("\u2714")

            self.status_icon.setStyleSheet("color: #4CAF50; font-size: 20px;")

        else:

            self.status_icon.setText("\u2718")

            self.status_icon.setStyleSheet("color: #F44336; font-size: 20px;")





# ---------------------------------------------------------------------------

# Setup Wizard Dialog

# ---------------------------------------------------------------------------



class SetupWizard(QDialog):

    def __init__(self, parent=None, preselected: list[str] | None = None):

        super().__init__(parent)

        self._preselected = preselected

        self.setWindowTitle(self.tr("CorridorKey 2.0 Setup"))

        self.setMinimumSize(700, 500)

        self.setModal(True)

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)



        self._worker: _DownloadWorker | None = None

        self._downloading = False

        self._cancelling = False

        self._rows: dict[str, _ModelRow] = {}



        layout = QVBoxLayout(self)

        layout.setSpacing(16)

        layout.setContentsMargins(28, 28, 28, 28)



        title = QLabel('Welcome to <span style="color:#FFF203;">EZ</span><span style="color:#4CAF50;">-</span><span style="color:#FFF203;">Corridor</span><span style="color:#4CAF50;">Key</span>')

        title.setFont(QFont("Open Sans", 18, QFont.Bold))

        title.setAlignment(Qt.AlignCenter)

        title.setStyleSheet("color: #FFFFFF;")

        layout.addWidget(title)



        subtitle = QLabel(

            self.tr(

                "Select which models to download. The core CorridorKey model is required.\n"

                "Optional models can be downloaded later from Edit \u2192 Download Manager."

            )

        )

        subtitle.setAlignment(Qt.AlignCenter)

        subtitle.setWordWrap(True)

        subtitle.setStyleSheet("color: #CCCCCC;")

        layout.addWidget(subtitle)



        layout.addSpacing(8)



        # Data directory picker (models, projects, frame cache)

        loc_label = QLabel(self.tr("Data directory (models, projects, frame cache):"))

        loc_label.setStyleSheet("color: #CCCCCC;")

        layout.addWidget(loc_label)



        loc_row = QHBoxLayout()

        loc_row.setSpacing(8)

        # Pre-fill with the user's previously saved install path so a re-open

        # of the wizard scans the folder the user is actually installed to —

        # not a fresh default that would report everything as missing.

        self._loc_edit = QLineEdit(str(_saved_install_path()))

        self._loc_edit.setStyleSheet(

            "QLineEdit { background: #1a1a18; color: #fff; border: 1px solid #444; "

            "border-radius: 4px; padding: 6px; }"

        )

        self._loc_edit.setReadOnly(True)

        self._loc_edit.textChanged.connect(self._refresh_installed_state)

        loc_row.addWidget(self._loc_edit, 1)



        browse_btn = QPushButton(self.tr("Browse..."))
        self._browse_btn = browse_btn

        browse_btn.setStyleSheet(

            "QPushButton { color: #ccc; border: 1px solid #555; padding: 6px 14px; "

            "border-radius: 4px; }"

            "QPushButton:hover { color: #fff; border-color: #888; }"

        )

        browse_btn.clicked.connect(self._on_browse_location)

        loc_row.addWidget(browse_btn)



        default_btn = QPushButton(self.tr("Default Location"))
        self._default_btn = default_btn

        default_btn.setToolTip(

            self.tr(

                "Reset the data directory to the platform default "

                "(in case you changed it and want to return)."

            )

        )

        default_btn.setStyleSheet(

            "QPushButton { color: #ccc; border: 1px solid #555; padding: 6px 14px; "

            "border-radius: 4px; }"

            "QPushButton:hover { color: #fff; border-color: #888; }"

        )

        default_btn.clicked.connect(self._on_default_location)

        loc_row.addWidget(default_btn)

        layout.addLayout(loc_row)



        layout.addSpacing(8)



        # Scrollable model list — labels always display full width

        from PySide6.QtWidgets import QScrollArea

        model_container = QWidget()

        model_layout = QVBoxLayout(model_container)

        model_layout.setContentsMargins(0, 0, 0, 0)

        model_layout.setSpacing(4)

        for model in MODELS:

            row = _ModelRow(model)

            self._rows[model["key"]] = row

            model_layout.addWidget(row)

        model_layout.addStretch()



        scroll = QScrollArea()

        scroll.setWidget(model_container)

        scroll.setWidgetResizable(True)

        scroll.setFrameShape(QScrollArea.NoFrame)

        scroll.setStyleSheet("QScrollArea { background: transparent; }"

                             "QWidget { background: transparent; }")

        layout.addWidget(scroll, 1)



        # Divider + desktop shortcut

        divider = QFrame()

        divider.setFrameShape(QFrame.HLine)

        divider.setStyleSheet("color: #333;")

        layout.addWidget(divider)



        self._shortcut_check = QCheckBox(self.tr("Create Desktop shortcut"))

        self._shortcut_check.setChecked(True)

        self._shortcut_check.setStyleSheet(

            "QCheckBox { color: #4CAF50; }"

            "QCheckBox::indicator:checked { background: #4CAF50; border: 1px solid #4CAF50; border-radius: 2px; }"

            "QCheckBox::indicator { border: 1px solid #555; border-radius: 2px; width: 14px; height: 14px; }"

        )

        layout.addWidget(self._shortcut_check)



        # One-time crash reporting ask for new users. Off unless they opt

        # in here; never shown again once answered. Same setting as

        # Preferences > Privacy.

        self._crash_check: QCheckBox | None = None

        from PySide6.QtCore import QSettings

        if not QSettings().value("privacy/crash_reports_prompted",

                                 False, type=bool):

            self._crash_check = QCheckBox(

                self.tr("Send anonymous crash reports to help "

                        "troubleshooting")

            )

            self._crash_check.setChecked(False)

            self._crash_check.setToolTip(

                self.tr(

                    "Crash details, GPU/driver info, app version. Never "

                    "your media, files, or personal info. Change anytime "

                    "in Preferences > Privacy."

                )

            )

            self._crash_check.setStyleSheet(

                "QCheckBox { color: #CCCCCC; }"

                "QCheckBox::indicator:checked { background: #FFF203; "

                "border: 1px solid #FFF203; border-radius: 2px; }"

                "QCheckBox::indicator { border: 1px solid #555; "

                "border-radius: 2px; width: 14px; height: 14px; }"

            )

            layout.addWidget(self._crash_check)



        self._overall_label = QLabel("")

        self._overall_label.setAlignment(Qt.AlignCenter)

        self._overall_label.setStyleSheet("color: #999;")

        layout.addWidget(self._overall_label)



        # Buttons — centered

        btn_layout = QHBoxLayout()

        btn_layout.addStretch()



        self._cancel_btn = QPushButton(self.tr("Cancel && Exit"))

        self._cancel_btn.setStyleSheet(

            "QPushButton { color: #F44336; border: 1px solid #F44336; padding: 8px 16px; "

            "border-radius: 4px; }"

            "QPushButton:hover { background: #F44336; color: #fff; }"

        )

        self._cancel_btn.clicked.connect(self._on_cancel)

        btn_layout.addWidget(self._cancel_btn)



        btn_layout.addSpacing(12)



        self._install_btn = QPushButton(self.tr("Download && Install"))

        self._install_btn.setStyleSheet(

            "QPushButton { background: #FFF203; color: #000; font-weight: bold; "

            "padding: 8px 24px; border-radius: 4px; }"

            "QPushButton:hover { background: #E6D900; }"

            "QPushButton:disabled { background: #555; color: #999; }"

        )

        self._install_btn.clicked.connect(self._on_install)

        btn_layout.addWidget(self._install_btn)



        btn_layout.addStretch()



        layout.addLayout(btn_layout)



        # Initial scan of the pre-filled install path so present/missing

        # state is visible the moment the wizard opens — not only after the

        # user interacts with the path picker.

        self._refresh_installed_state()



        # If opened with preselected keys, check only those and uncheck others

        if self._preselected is not None:
            for key, row in self._rows.items():
                if row._required or not row.checkbox.isEnabled():
                    continue
                row.checkbox.setChecked(key in self._preselected)
            if hasattr(self, "_install_btn") and any(r.is_selected() for r in self._rows.values()):
                self._install_btn.setText(self.tr("Download && Install"))



    def _refresh_installed_state(self) -> None:

        """Scan the current install path and update each _ModelRow accordingly.



        Called on wizard open and again every time ``_loc_edit`` changes

        (Browse / Default Location). Downloads in progress must not be

        clobbered — we skip the refresh if the worker is running.

        """

        if self._downloading:

            return

        install_path = Path(self._loc_edit.text()) if self._loc_edit.text() else None

        try:

            present = detect_installed_models(install_path)

        except Exception:

            logger.exception("Failed to detect installed models")

            return

        for key, row in self._rows.items():

            row.set_installed_state(present.get(key, False))

        any_selected = any(r.is_selected() for r in self._rows.values())

        all_req_ok = all(present.get(k, False) for k in _required_model_keys())

        if hasattr(self, "_install_btn"):

            if all_req_ok and not any_selected:

                self._install_btn.setText(self.tr("Continue"))

            else:

                self._install_btn.setText(self.tr("Download && Install"))



    def _on_browse_location(self):

        path = QFileDialog.getExistingDirectory(

            self, self.tr("Choose Install Location"), self._loc_edit.text()

        )

        if path:

            self._loc_edit.setText(path)



    def _on_default_location(self):

        """Reset the install path field to the platform default and re-scan."""

        self._loc_edit.setText(str(_default_install_dir()))



    def _on_cancel(self):

        if self._downloading:

            # Signal cancel but let thread finish gracefully

            self._cancelling = True

            self._cancel_btn.setEnabled(False)

            self._cancel_btn.setText(self.tr("Cancelling..."))

            if self._worker:

                self._worker.cancel()

            # Don't reject yet — wait for all_finished signal

        else:

            self.reject()



    def _persist_crash_consent(self) -> None:

        """Record the one-time crash reporting answer, if it was shown."""

        if self._crash_check is None:

            return

        from PySide6.QtCore import QSettings

        s = QSettings()

        s.setValue("privacy/crash_reports_enabled",

                   self._crash_check.isChecked())

        s.setValue("privacy/crash_reports_prompted", True)



    def _on_install(self):

        self._persist_crash_consent()

        # Always persist the chosen install directory first (even when all models are already installed!)

        install_dir = self._loc_edit.text().strip()

        if install_dir:

            from PySide6.QtCore import QSettings

            QSettings().setValue("app/install_path", install_dir)

            try:

                from backend.project import set_app_dir

                set_app_dir(install_dir)

            except Exception:

                pass

        mark_setup_seen()

        selected = [key for key, row in self._rows.items() if row.is_selected()]

        if not selected:

            if self._shortcut_check.isChecked():

                self._create_desktop_shortcut()

            self.accept()

            return



        self._downloading = True

        self._install_btn.setEnabled(False)

        self._loc_edit.setEnabled(False)
        if hasattr(self, "_browse_btn"):
            self._browse_btn.setEnabled(False)
        if hasattr(self, "_default_btn"):
            self._default_btn.setEnabled(False)

        for row in self._rows.values():

            row.checkbox.setEnabled(False)



        self._completed_count = 0

        self._total_count = len(selected)

        self._overall_label.setText(self.tr("Preparing downloads (0/%d)...") % self._total_count)

        self._overall_label.setStyleSheet("color: #FFF203;")



        # Create worker as child of this dialog so it stays alive

        install_root = Path(install_dir) if install_dir else None
        self._worker = _DownloadWorker(selected, install_root=install_root, parent=self)

        self._worker.model_started.connect(self._on_model_started)

        self._worker.model_progress.connect(self._on_model_progress)

        self._worker.model_finished.connect(self._on_model_finished)

        self._worker.all_finished.connect(self._on_all_finished)

        self._worker.start()



    def _on_model_started(self, key: str, label: str):

        if key in self._rows:

            self._rows[key].set_downloading()

        self._overall_label.setText(

            self.tr("Downloading %d/%d: %s...") % (self._completed_count + 1, self._total_count, label)

        )



    def _on_model_progress(self, key: str, downloaded_mb: int, total_mb: int):

        if key in self._rows:

            self._rows[key].set_progress(downloaded_mb, total_mb)



    def _on_model_finished(self, key: str, ok: bool, msg: str):

        if key in self._rows:

            self._rows[key].set_finished(ok)

        self._completed_count += 1



    def _on_all_finished(self, success: bool):

        # Wait for thread to fully exit before allowing dialog destruction

        if self._worker:

            self._worker.wait()

            self._worker = None



        self._downloading = False
        if hasattr(self, "_browse_btn"):
            self._browse_btn.setEnabled(True)
        if hasattr(self, "_default_btn"):
            self._default_btn.setEnabled(True)



        # If user pressed cancel, reject now that thread is safely done

        if self._cancelling:

            self.reject()

            return



        if success:

            self._overall_label.setText(self.tr("All %d downloads complete!") % self._total_count)

            self._overall_label.setStyleSheet("color: #4CAF50;")

        else:

            self._overall_label.setText(

                self.tr("Some downloads failed. You can retry from Edit \u2192 Download Manager.")

            )

            self._overall_label.setStyleSheet("color: #F44336;")



        # Create desktop shortcut if requested

        if self._shortcut_check.isChecked():

            self._create_desktop_shortcut()



        self._install_btn.setText(self.tr("Continue"))

        self._install_btn.setEnabled(True)

        try:

            self._install_btn.clicked.disconnect()

        except RuntimeError:

            pass

        self._install_btn.clicked.connect(self.accept)



        self._cancel_btn.setVisible(False)



    def _create_desktop_shortcut(self):

        """Create a desktop shortcut/alias to the .app bundle."""

        try:

            if not getattr(sys, "frozen", False):

                return

            desktop = Path.home() / "Desktop"

            if not desktop.is_dir():

                return



            if sys.platform == "darwin":

                # macOS: create a symlink to the .app on the Desktop

                app_path = Path(sys.executable).resolve().parent.parent.parent

                link = desktop / app_path.name

                if link.exists() or link.is_symlink():

                    link.unlink()

                link.symlink_to(app_path)

                logger.info("Desktop shortcut created: %s -> %s", link, app_path)

            elif sys.platform == "win32":

                # Windows: create a .lnk shortcut via PowerShell using env vars (no string injection)

                import subprocess

                app_exe = sys.executable

                icon = Path(sys._MEIPASS) / "ui" / "theme" / "corridorkey.ico"

                lnk = desktop / "CorridorKey.lnk"

                ps_cmd = (
                    "$ws = New-Object -ComObject WScript.Shell; "
                    "$s = $ws.CreateShortcut($env:CK_LNK); "
                    "$s.TargetPath = $env:CK_EXE; "
                    "$s.IconLocation = \"$($env:CK_ICON),0\"; "
                    "$s.Description = 'CorridorKey 2.0 - AI Green Screen'; "
                    "$s.Save()"
                )
                env = os.environ.copy()
                env["CK_LNK"] = str(lnk)
                env["CK_EXE"] = str(app_exe)
                env["CK_ICON"] = str(icon)

                subprocess.run(

                    ["powershell", "-NoProfile", "-Command", ps_cmd],

                    env=env, capture_output=True, timeout=10,

                )

                logger.info("Desktop shortcut created: %s", lnk)

        except Exception as e:

            logger.warning("Failed to create desktop shortcut: %s", e)



    def closeEvent(self, event):

        """Block close while downloading — user must use Cancel button."""

        if self._downloading:

            event.ignore()

            return

        super().closeEvent(event)



    def reject(self):

        """Block reject (Escape key / close) while downloading."""

        if self._downloading:

            return

        super().reject()

