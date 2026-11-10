"""Shortcut registry — single source of truth for keyboard bindings.

Stores default shortcuts, loads/saves user overrides via QSettings,
creates QShortcut objects, and detects conflicts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QWidget

logger = logging.getLogger(__name__)

_QSETTINGS_GROUP = "shortcuts"


@dataclass
class ShortcutDef:
    """One shortcut definition."""
    action_id: str          # Unique internal ID, e.g. "run_inference"
    display_name: str       # Human-readable label shown in the dialog
    category: str           # Grouping: "Global", "Timeline", etc.
    default_key: str        # Default key sequence string, e.g. "Ctrl+R"
    callback_name: str      # Name of the method on MainWindow to call
    menu_action: bool = False  # True for shortcuts managed by QAction (menu bar)
    app_shortcut: bool = False  # True to use ApplicationShortcut context (fires globally)


# Authoritative list — order here = display order in dialog
SHORTCUT_DEFAULTS: list[ShortcutDef] = [
    # Global
    ShortcutDef("escape",             "Stop / Cancel",                "Global",     "Esc",          "_on_escape",                 app_shortcut=True),
    ShortcutDef("run_inference",      "Run Inference",                "Global",     "Ctrl+R",       "_on_run_inference",          app_shortcut=True),
    ShortcutDef("run_all",            "Run All Clips",                "Global",     "Ctrl+Shift+R", "_on_run_all_ready",          app_shortcut=True),
    ShortcutDef("save_session",       "Save Session",                 "Global",     "Ctrl+S",       "_on_save_session",           menu_action=True),
    ShortcutDef("open_project",       "Open Project",                 "Global",     "Ctrl+O",       "_on_open_project",           menu_action=True),
    ShortcutDef("toggle_mute",        "Toggle Mute",                  "Global",     "Ctrl+M",       "_toggle_mute",               app_shortcut=True),
    ShortcutDef("delete_clips",       "Remove Selected Clips",        "Global",     "Del",          "_on_delete_selected_clips",  app_shortcut=True),
    ShortcutDef("toggle_queue",       "Toggle Queue",                 "Global",     "Q",            "_toggle_queue_panel",        app_shortcut=True),
    ShortcutDef("debug_console",      "Debug Console",                "Global",     "F12",          "_toggle_debug_console",      app_shortcut=True),
    ShortcutDef("preferences",        "Preferences",                  "Global",     "Ctrl+,",       "_show_preferences",          menu_action=True),
    # Timeline & Navigation
    ShortcutDef("set_in",             "Set In-Point",                 "Timeline",   "I",            "_set_in_point",              app_shortcut=True),
    ShortcutDef("set_out",            "Set Out-Point",                "Timeline",   "O",            "_set_out_point",             app_shortcut=True),
    ShortcutDef("clear_in_out",       "Clear In/Out",                 "Timeline",   "Alt+I",        "_clear_in_out",              app_shortcut=True),
    ShortcutDef("step_prev",          "Step Frame Backward",          "Timeline",   "Left",         "_step_frame_back",           app_shortcut=True),
    ShortcutDef("step_next",          "Step Frame Forward",           "Timeline",   "Right",        "_step_frame_forward",        app_shortcut=True),
    ShortcutDef("jump_start",         "Jump to First Frame",          "Timeline",   "Home",         "_jump_to_first_frame",       app_shortcut=True),
    ShortcutDef("jump_end",           "Jump to Last Frame",           "Timeline",   "End",          "_jump_to_last_frame",        app_shortcut=True),
    # Playback
    ShortcutDef("play_pause",         "Play / Pause",                 "Playback",   "Space",        "_toggle_playback",           app_shortcut=True),
    # Paint & Prompts
    ShortcutDef("annotation_brush",   "Brush Tool (SAM 3.1)",         "Paint",      "B",            "_toggle_annotation_brush",   app_shortcut=True),
    ShortcutDef("annotation_point",   "Point Tool (SAM 3.1)",         "Paint",      "P",            "_toggle_annotation_point",   app_shortcut=True),
    ShortcutDef("annotation_hair",    "Hair & Detail Paint (SAM 3.1)", "Paint",      "H",            "_toggle_annotation_hair",    app_shortcut=True),
    ShortcutDef("annotation_eraser",  "Eraser Tool (SAM 3.1)",        "Paint",      "E",            "_toggle_annotation_eraser",  app_shortcut=True),
    ShortcutDef("box_prompt",         "SAM 3.1 Box Prompt",           "Paint",      "R",            "_toggle_box_prompt",         app_shortcut=True),
    ShortcutDef("annotation_holdout", "Holdout Matte Paint",          "Paint",      "3",            "_toggle_annotation_holdout", app_shortcut=True),
    ShortcutDef("eyedropper",         "Eyedropper (Pick Color)",      "Paint",      "C",            "_toggle_eyedropper",         app_shortcut=True),
    ShortcutDef("cycle_fg_color",     "Cycle Foreground Color",       "Paint",      "Shift+C",      "_cycle_fg_color",            app_shortcut=True),
    ShortcutDef("undo_annotation",    "Undo Paint Stroke",            "Paint",      "Ctrl+Z",       "_undo_annotation",           app_shortcut=True),
    ShortcutDef("clear_annotations",  "Clear Paint Strokes",          "Paint",      "Ctrl+Shift+X", "_confirm_clear_annotations", app_shortcut=True),
    ShortcutDef("chroma_key",         "Toggle Chroma Key",            "Paint",      "`",            "_toggle_chroma_key",         app_shortcut=True),
    # Viewer
    ShortcutDef("fit_viewer",         "Fit Viewer",                   "Viewer",     "F",            "_fit_viewer",                app_shortcut=True),
    ShortcutDef("toggle_ab_wipe",     "Toggle A/B Wipe",              "Viewer",     "A",            "_toggle_ab_wipe",            app_shortcut=True),
    ShortcutDef("view_input",         "View: INPUT",                  "Viewer",     "F1",           "_view_mode_input",           app_shortcut=True),
    ShortcutDef("view_mask",          "View: MASK",                   "Viewer",     "F2",           "_view_mode_mask",            app_shortcut=True),
    ShortcutDef("view_alpha",         "View: ALPHA",                  "Viewer",     "F3",           "_view_mode_alpha",           app_shortcut=True),
    ShortcutDef("view_fg",            "View: FG",                     "Viewer",     "F4",           "_view_mode_fg",              app_shortcut=True),
    ShortcutDef("view_matte",         "View: MATTE",                  "Viewer",     "F5",           "_view_mode_matte",           app_shortcut=True),
    ShortcutDef("view_comp",          "View: COMP",                   "Viewer",     "F6",           "_view_mode_comp",            app_shortcut=True),
    ShortcutDef("view_proc",          "View: PROC",                   "Viewer",     "F7",           "_view_mode_proc",            app_shortcut=True),
]

# Category display order
CATEGORY_ORDER = ["Global", "Timeline", "Playback", "Viewer", "Paint"]


class ShortcutRegistry:
    """Manages shortcut definitions, user overrides, QShortcut objects, and conflicts."""

    def __init__(self) -> None:
        self._defs: dict[str, ShortcutDef] = {d.action_id: d for d in SHORTCUT_DEFAULTS}
        self._overrides: dict[str, str] = {}
        self._shortcuts: dict[str, QShortcut] = {}
        self._load_overrides()

    def _load_overrides(self) -> None:
        """Load user-customized key bindings from QSettings."""
        s = QSettings()
        s.beginGroup(_QSETTINGS_GROUP)
        for key in s.childKeys():
            if key in self._defs:
                self._overrides[key] = s.value(key, type=str)
        s.endGroup()
        if self._overrides:
            logger.info(f"Loaded {len(self._overrides)} custom shortcut(s)")

    def save_overrides(self) -> None:
        """Persist current overrides to QSettings. Only stores non-default keys."""
        s = QSettings()
        s.beginGroup(_QSETTINGS_GROUP)
        for key in s.childKeys():
            s.remove(key)
        for action_id, key_str in self._overrides.items():
            if key_str != self._defs[action_id].default_key:
                s.setValue(action_id, key_str)
        s.endGroup()

    def get_key(self, action_id: str) -> str:
        """Return the effective key sequence string for an action."""
        return self._overrides.get(action_id, self._defs[action_id].default_key)

    def set_key(self, action_id: str, key_str: str) -> None:
        """Set a custom key binding (in-memory only until save_overrides)."""
        self._overrides[action_id] = key_str

    def reset_key(self, action_id: str) -> None:
        """Reset a single action to its default key."""
        self._overrides.pop(action_id, None)

    def reset_all(self) -> None:
        """Reset all actions to defaults."""
        self._overrides.clear()

    def is_default(self, action_id: str) -> bool:
        """True if the action uses its default key (no custom override)."""
        return action_id not in self._overrides or \
            self._overrides[action_id] == self._defs[action_id].default_key

    def get_def(self, action_id: str) -> ShortcutDef | None:
        """Return the definition for an action."""
        return self._defs.get(action_id)

    def definitions(self) -> list[ShortcutDef]:
        """Return all definitions in their original order."""
        return list(SHORTCUT_DEFAULTS)

    def snapshot_overrides(self) -> dict[str, str]:
        """Return a copy of current overrides (for cancel/revert in dialogs)."""
        return dict(self._overrides)

    def restore_overrides(self, snapshot: dict[str, str]) -> None:
        """Restore overrides from a previous snapshot."""
        self._overrides = dict(snapshot)

    def find_conflicts(self, action_id: str, key_str: str) -> list[str]:
        """Return action_ids that already use key_str (excluding action_id itself)."""
        if not key_str:
            return []
        target = QKeySequence(key_str)
        conflicts = []
        for aid in self._defs:
            if aid == action_id:
                continue
            effective = self.get_key(aid)
            if effective and QKeySequence(effective) == target:
                conflicts.append(aid)
        return conflicts

    # ── Live QShortcut management ──

    def create_shortcuts(self, owner: QWidget) -> None:
        """Create QShortcut objects for all non-menu-action shortcuts."""
        self.destroy_shortcuts()
        for defn in SHORTCUT_DEFAULTS:
            if defn.menu_action:
                continue  # handled by QAction in menu bar
            key_str = self.get_key(defn.action_id)
            if not key_str:
                continue  # unbound
            callback = getattr(owner, defn.callback_name, None)
            if callback is None:
                logger.warning(f"Shortcut callback not found: {defn.callback_name}")
                continue
            sc = QShortcut(QKeySequence(key_str), owner)
            if defn.app_shortcut:
                sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(callback)
            self._shortcuts[defn.action_id] = sc

    def destroy_shortcuts(self) -> None:
        """Delete all existing QShortcut objects."""
        for sc in self._shortcuts.values():
            sc.deleteLater()
        self._shortcuts.clear()
