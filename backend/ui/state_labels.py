"""Translated display labels for backend enum values."""
from __future__ import annotations

from PySide6.QtCore import QCoreApplication

from backend.clip_state import ClipState


def state_display_name(state: ClipState) -> str:
    """Return the translated UI label for a clip state badge or info line."""
    labels = {
        ClipState.EXTRACTING: QCoreApplication.translate("ClipState", "EXTRACTING"),
        ClipState.RAW: QCoreApplication.translate("ClipState", "RAW"),
        ClipState.MASKED: QCoreApplication.translate("ClipState", "MASKED"),
        ClipState.READY: QCoreApplication.translate("ClipState", "READY"),
        ClipState.COMPLETE: QCoreApplication.translate("ClipState", "COMPLETE"),
        ClipState.ERROR: QCoreApplication.translate("ClipState", "ERROR"),
    }
    return labels.get(state, state.value)


def job_type_display_name(job_type) -> str:
    """Return the translated UI label for a GPU job type."""
    from backend.job_queue import JobType
    labels = {
        JobType.INFERENCE: QCoreApplication.translate("JobType", "CorridorKey 2.0"),
        JobType.BIREFNET_ALPHA: "BiRefNet",
        JobType.SAM3_PREVIEW: QCoreApplication.translate("JobType", "SAM 3.1 Preview"),
        JobType.SAM3_TRACK: QCoreApplication.translate("JobType", "SAM 3.1 Track"),
        JobType.MATANYONE2_ALPHA: "MatAnyone2",
        JobType.PREVIEW_REPROCESS: QCoreApplication.translate("JobType", "Preview"),
    }
    return labels.get(job_type, QCoreApplication.translate("JobType", "Pipeline"))


def backend_status_text(message: str) -> str:
    """Translate a backend status/phase message for the status bar."""
    if not message:
        return message
    known = {
        "Loading model...": QCoreApplication.translate("BackendStatus", "Loading model..."),
        "Loading frames...": QCoreApplication.translate("BackendStatus", "Loading frames..."),
        "Loading masks...": QCoreApplication.translate("BackendStatus", "Loading masks..."),
        "Loading preview frame...": QCoreApplication.translate("BackendStatus", "Loading preview frame..."),
        "Loading first-frame mask...": QCoreApplication.translate("BackendStatus", "Loading first-frame mask..."),
        "Loading state dict...": QCoreApplication.translate("BackendStatus", "Loading state dict..."),
        "Loading checkpoint weights...": QCoreApplication.translate("BackendStatus", "Loading checkpoint weights..."),
        "Loading MatAnyone2 checkpoint...": QCoreApplication.translate("BackendStatus", "Loading MatAnyone2 checkpoint..."),
        "Loading MatAnyone2 model...": QCoreApplication.translate("BackendStatus", "Loading MatAnyone2 model..."),
        "Initializing model backbone...": QCoreApplication.translate("BackendStatus", "Initializing model backbone..."),
        "Moving model to GPU...": QCoreApplication.translate("BackendStatus", "Moving model to GPU..."),
        "Model ready": QCoreApplication.translate("BackendStatus", "Model ready"),
        "BiRefNet model ready": QCoreApplication.translate("BackendStatus", "BiRefNet model ready"),
        "MatAnyone2 model ready": QCoreApplication.translate("BackendStatus", "MatAnyone2 model ready"),
        "Running SAM 3.1 tracker...": QCoreApplication.translate("BackendStatus", "Running SAM 3.1 tracker..."),
        "Running BiRefNet inference...": QCoreApplication.translate("BackendStatus", "Running BiRefNet inference..."),
        "Running MatAnyone2 inference...": QCoreApplication.translate("BackendStatus", "Running MatAnyone2 inference..."),
        "Previewing SAM 3.1 on annotated frame...": QCoreApplication.translate("BackendStatus", "Previewing SAM 3.1 on annotated frame..."),
        "Finalizing alpha hints...": QCoreApplication.translate("BackendStatus", "Finalizing alpha hints..."),
        "Clearing CUDA cache...": QCoreApplication.translate("BackendStatus", "Clearing CUDA cache..."),
    }
    return known.get(message, message)
