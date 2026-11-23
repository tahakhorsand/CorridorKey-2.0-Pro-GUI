"""Named InferenceParams Preset Manager for CorridorKey 2.0."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_DEFAULT_PRESETS_FILE = str(
    Path(__file__).resolve().parent.parent / "config" / "presets.json"
)

BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    "Default Balanced": {
        "screen_color": "auto",
        "despill_strength": 0.50,
        "hint_quality": 0.95,
        "model_scale": 1.0,
        "alpha_clip_black": 0.0,
        "alpha_clip_white": 1.0,
        "alpha_gamma": 1.0,
        "edge_erode_px": 0,
        "edge_blur_px": 0,
        "auto_despeckle": False,
        "despeckle_size": 20,
        "garbage_matte_px": 0,
        "auto_roi_crop": False,
    },
    "Clean Studio Green": {
        "screen_color": "green",
        "despill_strength": 0.65,
        "hint_quality": 0.95,
        "model_scale": 1.0,
        "alpha_clip_black": 0.02,
        "alpha_clip_white": 0.97,
        "alpha_gamma": 1.05,
        "edge_erode_px": 0,
        "edge_blur_px": 0,
        "auto_despeckle": True,
        "despeckle_size": 40,
        "garbage_matte_px": 16,
        "auto_roi_crop": False,
    },
    "Fine Hair Detail": {
        "screen_color": "auto",
        "despill_strength": 0.45,
        "hint_quality": 0.90,
        "model_scale": 1.0,
        "alpha_clip_black": 0.0,
        "alpha_clip_white": 0.98,
        "alpha_gamma": 1.18,
        "edge_erode_px": 0,
        "edge_blur_px": 1,
        "auto_despeckle": False,
        "despeckle_size": 10,
        "garbage_matte_px": 0,
        "auto_roi_crop": True,
    },
    "Hard Core & Spill Choke": {
        "screen_color": "green",
        "despill_strength": 0.85,
        "hint_quality": 0.98,
        "model_scale": 1.0,
        "alpha_clip_black": 0.06,
        "alpha_clip_white": 0.90,
        "alpha_gamma": 0.92,
        "edge_erode_px": -1,
        "edge_blur_px": 1,
        "auto_despeckle": True,
        "despeckle_size": 120,
        "garbage_matte_px": 20,
        "auto_roi_crop": False,
    },
}


class PresetManager:
    """Manage built-in and user-defined InferenceParams presets."""

    def __init__(self, presets_path: str | None = None) -> None:
        self._path = presets_path or _DEFAULT_PRESETS_FILE
        self._user_presets: dict[str, dict[str, Any]] = {}
        self.load()

    @property
    def presets_path(self) -> str:
        return self._path

    def load(self) -> None:
        """Load user presets from disk."""
        self._user_presets = {}
        if not os.path.isfile(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, dict) and k.strip():
                        self._user_presets[k.strip()] = dict(v)
        except Exception:
            self._user_presets = {}

    def _flush(self) -> None:
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._user_presets, f, indent=2, sort_keys=True)
        os.replace(tmp, self._path)

    def list_names(self) -> list[str]:
        """Return ordered preset names: built-ins followed by sorted user presets."""
        names = list(BUILTIN_PRESETS.keys())
        for k in sorted(self._user_presets.keys(), key=str.lower):
            if k not in BUILTIN_PRESETS:
                names.append(k)
        return names

    def is_builtin(self, name: str) -> bool:
        return name in BUILTIN_PRESETS and name not in self._user_presets

    def get_preset(self, name: str) -> dict[str, Any] | None:
        if name in self._user_presets:
            return dict(self._user_presets[name])
        if name in BUILTIN_PRESETS:
            return dict(BUILTIN_PRESETS[name])
        return None

    def save_preset(self, name: str, params_dict: dict[str, Any]) -> None:
        clean = (name or "").strip()
        if not clean:
            raise ValueError("Preset name cannot be empty")
        self._user_presets[clean] = dict(params_dict)
        self._flush()

    def delete_preset(self, name: str) -> bool:
        """Delete a user preset (returns False for pure built-ins)."""
        if name in self._user_presets:
            del self._user_presets[name]
            self._flush()
            return True
        return False
