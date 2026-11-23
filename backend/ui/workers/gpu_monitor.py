"""GPU monitor — non-blocking background thread VRAM/memory poller.

Moves NVML/CUDA probing and periodic telemetry off the main Qt event loop
so window creation stays responsive and never hitches on driver queries.
"""
from __future__ import annotations

import logging
import platform
import subprocess
import sys
import threading

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class GPUMonitor(QObject):
    """Polls GPU VRAM / unified memory periodically on a daemon thread."""

    vram_updated = Signal(dict)  # GPU info dict
    gpu_name = Signal(str)       # emitted once on first successful poll

    def __init__(self, interval_ms: int = 2000, parent=None):
        super().__init__(parent)
        self._interval_sec = max(0.25, float(interval_ms) / 1000.0)
        self._nvml_available = False
        self._nvml_handle = None
        self._torch_fallback = False
        self._apple_silicon = False
        self._apple_chip_name: str | None = None
        self._name_emitted = False
        self._initialized = False

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _init_nvml(self) -> None:
        """Initialize telemetry backend (called lazily inside background thread)."""
        self._initialized = True
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_available = True
            logger.info("GPU monitor: using NVML")
            return
        except Exception:
            self._nvml_available = False

        # Apple Silicon: unified memory reported via psutil without touching torch
        if sys.platform == "darwin" and platform.machine() == "arm64":
            self._apple_silicon = True
            self._apple_chip_name = self._detect_apple_chip()
            logger.info(
                "GPU monitor: Apple Silicon (%s), reporting unified memory",
                self._apple_chip_name or "unknown chip",
            )
            return

        # Delay any torch fallback slightly so UI boot finishes before torch loads
        if self._stop_event.wait(0.25):
            return
        try:
            import torch
            if torch.cuda.is_available():
                self._torch_fallback = True
                logger.info("GPU monitor: NVML unavailable, using torch.cuda fallback")
                return
        except ImportError:
            pass

        logger.info("GPU monitor: no GPU monitoring available")

    def start(self) -> None:
        """Start background GPU monitoring thread (non-blocking)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="CorridorKey-GPUMonitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop background polling cleanly."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None

    def _worker_loop(self) -> None:
        if not self._initialized:
            self._init_nvml()
        while not self._stop_event.is_set():
            self._poll()
            if self._stop_event.wait(self._interval_sec):
                break

    def _poll(self) -> None:
        """Query GPU status and emit signal if still running."""
        if self._stop_event.is_set():
            return
        if not self._initialized:
            self._init_nvml()
        if self._stop_event.is_set():
            return
        info = self._query_gpu()
        if self._stop_event.is_set():
            return
        self.vram_updated.emit(info)
        if info.get("available") and not self._name_emitted:
            self._name_emitted = True
            self.gpu_name.emit(info.get("name", "GPU"))

    def _query_gpu(self) -> dict:
        if self._nvml_available:
            return self._query_nvml()
        if self._torch_fallback:
            return self._query_torch()
        if self._apple_silicon:
            return self._query_apple_silicon()
        return {"available": False}

    def _query_nvml(self) -> dict:
        try:
            import pynvml
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            name = pynvml.nvmlDeviceGetName(self._nvml_handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            total_gb = mem.total / (1024 ** 3)
            used_gb = mem.used / (1024 ** 3)
            free_gb = mem.free / (1024 ** 3)
            return {
                "available": True,
                "name": name,
                "total_gb": round(total_gb, 1),
                "used_gb": round(used_gb, 1),
                "free_gb": round(free_gb, 1),
                "usage_pct": round(used_gb / total_gb * 100, 1) if total_gb > 0 else 0,
            }
        except Exception as e:
            logger.debug(f"NVML query failed: {e}")
            return {"available": False}

    def _query_torch(self) -> dict:
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory
            reserved = torch.cuda.memory_reserved(0)
            free = total - reserved
            total_gb = total / (1024 ** 3)
            used_gb = reserved / (1024 ** 3)
            free_gb = free / (1024 ** 3)
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "total_gb": round(total_gb, 1),
                "used_gb": round(used_gb, 1),
                "free_gb": round(free_gb, 1),
                "usage_pct": round(used_gb / total_gb * 100, 1) if total_gb > 0 else 0,
            }
        except Exception as e:
            logger.debug(f"Torch GPU query failed: {e}")
            return {"available": False}

    @staticmethod
    def _detect_apple_chip() -> str | None:
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    def _query_apple_silicon(self) -> dict:
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_bytes = int(mem.total)
            free_bytes = int(mem.available)
            used_bytes = max(0, total_bytes - free_bytes)
            total_gb = total_bytes / (1024 ** 3)
            used_gb = used_bytes / (1024 ** 3)
            free_gb = free_bytes / (1024 ** 3)
            usage_pct = (used_bytes / total_bytes * 100.0) if total_bytes > 0 else 0.0
            return {
                "available": True,
                "name": self._apple_chip_name or "Apple Silicon",
                "total_gb": round(total_gb, 1),
                "used_gb": round(used_gb, 1),
                "free_gb": round(free_gb, 1),
                "usage_pct": round(usage_pct, 1),
            }
        except ImportError:
            logger.debug("psutil not available for Apple Silicon memory monitoring")
            return {"available": False}
        except Exception as e:
            logger.debug(f"Apple Silicon memory query failed: {e}")
            return {"available": False}
