"""CorridorKey entry point — GUI (default) or CLI mode.

Usage:
    python main.py              # Launch GUI (default)
    python main.py --gui        # Launch GUI explicitly
    python main.py --cli        # Run CLI wizard (original clip_manager.py)
"""
from __future__ import annotations

import os
# Enable OpenEXR support in OpenCV — must be set before cv2 is imported anywhere
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import json
import logging
import logging.handlers
import platform
import sys
from datetime import datetime
from pathlib import Path


class _NullTextStream:
    """Minimal text stream used when GUI launches without a console."""

    encoding = "utf-8"
    errors = "replace"

    def write(self, data) -> int:
        if data is None:
            return 0
        if isinstance(data, bytes):
            return len(data)
        return len(str(data))

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


def ensure_standard_streams() -> None:
    """Provide harmless stdout/stderr sinks for pythonw/PyInstaller GUI launches."""
    if sys.stdout is None:
        sys.stdout = _NullTextStream()
    if sys.stderr is None:
        sys.stderr = _NullTextStream()


def get_base_dir() -> str:
    """Get the project base directory, handling both dev and frozen (PyInstaller) modes.

    In development: returns the directory containing this file.
    In frozen build: returns sys._MEIPASS (PyInstaller temp dir) for bundled
    resources, or the executable's directory for user files.
    """
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller bundle
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))


def is_portable() -> bool:
    """True when a 'portable.txt' marker exists next to the executable."""
    if getattr(sys, 'frozen', False):
        return os.path.isfile(os.path.join(os.path.dirname(sys.executable), 'portable.txt'))
    return False


def get_app_dir() -> str:
    """Get the application directory for user-facing paths (logs, sessions, etc.).

    Portable mode: all data stays next to the .exe (USB-stick friendly).
    macOS frozen: /Applications is read-only → ~/Library/Application Support/EZ-CorridorKey.
    Windows frozen: returns the .exe directory (user-writable).
    Use get_base_dir() for bundled resources (checkpoints, QSS, fonts).
    """
    if getattr(sys, 'frozen', False):
        if is_portable():
            return os.path.dirname(sys.executable)
        if sys.platform == 'darwin':
            support = os.path.join(
                os.path.expanduser('~'), 'Library', 'Application Support', 'EZ-CorridorKey'
            )
            os.makedirs(support, exist_ok=True)
            return support
        return os.path.dirname(sys.executable)
    # In development mode: repo root (one level above backend/)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Ensure project root is on path
_base = get_base_dir()
sys.path.insert(0, _base)

# Add bundled ffmpeg to PATH if present
_ffmpeg = os.path.join(_base, "ffmpeg")
if os.path.isdir(_ffmpeg):
    os.environ["PATH"] = _ffmpeg + os.pathsep + os.environ.get("PATH", "")

# Fully offline and portable runtime
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with dual output: console + per-session file.

    Console respects --log-level flag. File always captures DEBUG.
    All timestamps use the system's local time.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"

    # Console handler — respects --log-level
    # Force UTF-8 so non-ASCII paths (e.g. CJK characters) don't crash
    # on Windows cp1252 consoles.
    import io
    utf8_stream = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True,
    ) if hasattr(sys.stderr, "buffer") else sys.stderr
    console = logging.StreamHandler(utf8_stream)
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))

    # File handler — session-named log, always DEBUG
    log_dir = os.path.join(get_app_dir(), "logs", "backend")
    os.makedirs(log_dir, exist_ok=True)

    session_ts = datetime.now().strftime("%y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{session_ts}_corridorkey.log")

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=50 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    # Root logger — let handlers filter
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)


def run_gui() -> int:
    """Launch the PySide6 desktop application."""
    ensure_standard_streams()
    # The GUI has its own progress UI; external library progress bars just create
    # stderr/console issues in pythonw and frozen launches.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    from ui.app import create_app
    from ui.main_window import MainWindow
    from ui.recent_sessions import RecentSessionsStore
    from backend import CorridorKeyService

    app = create_app()

    # First-launch / version-upgrade setup: show the Download Manager when a
    # required model is missing OR the installed version changed since the
    # last successful launch. The user can dismiss the version-upgrade case
    # and continue; missing-required still blocks launch.
    from ui.widgets.setup_wizard import (
        SetupWizard,
        needs_setup,
        has_required_models,
        mark_setup_seen,
    )
    if needs_setup():
        wizard = SetupWizard()
        wizard.exec()
        if not has_required_models():
            return 0
    mark_setup_seen()

    # Frozen builds: point projects at the user-chosen install directory
    # (same location as model checkpoints) instead of the exe directory.
    if getattr(sys, 'frozen', False):
        from backend.project import get_data_dir, set_app_dir
        set_app_dir(get_data_dir())

    service = CorridorKeyService()
    store = RecentSessionsStore()
    store.prune_missing()
    window = MainWindow(service, store)
    window.show()
    rc = app.exec()
    # Force-exit: torch.compile spawns Triton background threads that prevent
    # clean shutdown, leaving a zombie process holding VRAM.
    os._exit(rc)


def run_cli() -> int:
    """Run CorridorKey CLI inference via infer.py as a subprocess.

    Forwards CLI arguments directly to infer.py.
    """
    script = os.path.join(get_base_dir(), "infer.py")
    if not os.path.isfile(script):
        logging.error(
            "CLI mode requires infer.py in the project root. "
            "Use --gui (default) for the graphical interface."
        )
        return 1

    import subprocess

    # Forward everything after --cli to clip_manager.py.
    # sys.argv looks like: ['main.py', '--cli', '--action', 'wizard', ...]
    # We strip our own flags (--cli, --gui, --log-level <val>) and pass the rest.
    forwarded: list[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--cli", "--gui"):
            continue
        if arg == "--log-level":
            skip_next = True  # skip the value that follows
            continue
        forwarded.append(arg)

    cmd = [sys.executable, script] + forwarded
    logging.info("CLI passthrough: %s", " ".join(cmd))
    return subprocess.call(cmd)


def _checkpoint_info(path: str | os.PathLike[str]) -> dict[str, object]:
    p = Path(path)
    return {
        "path": str(p),
        "size": p.stat().st_size,
    }


def _verify_torch_checkpoint_load(path: str | os.PathLike[str]) -> dict[str, object]:
    import gc
    import torch

    p = Path(path)
    checkpoint = torch.load(str(p), map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    key_count = len(state_dict) if hasattr(state_dict, "keys") else None
    info = _checkpoint_info(p)
    info["state_dict_keys"] = key_count
    del state_dict
    del checkpoint
    gc.collect()
    return info


def _verify_safetensors_checkpoint_load(path: str | os.PathLike[str]) -> dict[str, object]:
    from safetensors import safe_open

    p = Path(path)
    with safe_open(str(p), framework="np") as handle:
        keys = list(handle.keys())
    info = _checkpoint_info(p)
    info["tensor_keys"] = len(keys)
    info["first_keys"] = keys[:5]
    return info


def _verify_birefnet_model_load(model_dir: str | os.PathLike[str]) -> dict[str, object]:
    import gc
    import torch
    from transformers import AutoConfig, AutoModelForImageSegmentation

    model_path = Path(model_dir)
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForImageSegmentation.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    model.to("cpu")
    model.eval()
    info = {
        "path": str(model_path),
        "model_type": getattr(config, "model_type", None),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    del model
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    return info


def run_model_path_check(
    data_dir: str | None = None,
    *,
    verify_model_loads: bool = False,
    verify_optional_model_downloads: bool = False,
) -> int:
    """Hidden smoke check to verify CorridorKey 2.0 models and foundation assets."""
    from backend.project import get_data_dir, set_app_dir

    resolved_data_dir = os.path.abspath(data_dir or get_data_dir())
    set_app_dir(resolved_data_dir)

    result: dict[str, object] = {
        "frozen": bool(getattr(sys, "frozen", False)),
        "executable": sys.executable,
        "base_dir": get_base_dir(),
        "data_dir": resolved_data_dir,
    }
    errors: list[str] = []

    # Check CorridorKey 2.0 core weights
    v2_weights = Path(resolved_data_dir) / "weights" / "corridorkey_v2.safetensors"
    if v2_weights.is_file():
        result["corridorkey_v2"] = str(v2_weights)
        if verify_model_loads:
            result["v2_load"] = _verify_safetensors_checkpoint_load(v2_weights)
    else:
        errors.append(f"corridorkey_v2.safetensors missing at {v2_weights}")

    stats_weights = Path(resolved_data_dir) / "weights" / "foundation_stats.safetensors"
    if stats_weights.is_file():
        result["foundation_stats"] = str(stats_weights)
    else:
        errors.append(f"foundation_stats.safetensors missing at {stats_weights}")

    # Check VFMs
    for vfm_name in ("rvm", "moge", "cradio"):
        vfm_p = Path(resolved_data_dir) / "vfms" / vfm_name
        if not vfm_p.is_dir():
            errors.append(f"VFM directory missing: {vfm_p}")

    # Check BiRefNet
    try:
        from modules.BiRefNetModule import wrapper as birefnet_wrapper
        repo_id, _ = birefnet_wrapper.BIREFNET_MODELS.get("Matting", ("ZhengPeng7/BiRefNet-matting", 1024))
        snap = birefnet_wrapper.resolve_birefnet_snapshot(repo_id)
        if snap:
            result["birefnet_snapshot"] = str(snap)
            if verify_model_loads:
                result["birefnet_load"] = _verify_birefnet_model_load(snap)
    except Exception as exc:
        errors.append(f"BiRefNet check error: {exc}")

    ok = not errors
    result["ok"] = ok
    if errors:
        result["errors"] = errors
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if ok else 1


def main() -> int:
    ensure_standard_streams()
    parser = argparse.ArgumentParser(
        description="CorridorKey — AI Green Screen Keyer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run the original CLI wizard instead of the GUI",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Launch the GUI (default)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--opt-mode",
        default=None,
        choices=["auto", "speed", "lowvram"],
        help="GPU optimization mode: auto (detect VRAM), speed (torch.compile), "
             "lowvram (tiled refiner for 8GB cards). Overrides CORRIDORKEY_OPT_MODE env var.",
    )
    parser.add_argument(
        "--verify-model-paths",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verify-model-data-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verify-model-loads",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verify-optional-model-downloads",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    # parse_known_args so CLI-mode flags (--action, --win_path, etc.)
    # pass through to clip_manager.py without error.
    args, _unknown = parser.parse_known_args()

    # CLI flag takes priority over env var for optimization mode
    if args.opt_mode:
        os.environ['CORRIDORKEY_OPT_MODE'] = args.opt_mode

    if args.verify_model_paths:
        return run_model_path_check(
            args.verify_model_data_dir,
            verify_model_loads=args.verify_model_loads,
            verify_optional_model_downloads=args.verify_optional_model_downloads,
        )

    setup_logging(args.log_level)

    # Configure backend with the application directory
    from backend.project import set_app_dir
    set_app_dir(get_app_dir())

    if args.cli:
        return run_cli()
    return run_gui()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
