"""CorridorKey 2.0 Studio Dark Theme.

Calibrated professional VFX / keying dark palette with high-contrast semantics:
- Background Canvas: #0B0C0E (deep slate/obsidian)
- Surface Panels:    #121418 (panel background)
- Card / Elevated:   #181B20 (elevated controls and headers)
- Subtle Borders:    #22262E (crisp 1px borders)
- Brand Accent:      #6366F1 (Indigo primary)
- Emerald Key:       #10B981 (Keying active / Green screen)
- Blue Key:          #3B82F6 (Blue screen)
- Text High:         #F3F4F6 (95% brightness)
- Text Medium:       #9CA3AF (gray-400)
- Text Muted:        #6B7280 (gray-500)
"""
from __future__ import annotations

ACCENT_COLORS = {
    "green": {
        "accent": "#10B981",        # Emerald 500
        "accent_hover": "#059669",  # Emerald 600
        "accent_dim": "rgba(16, 185, 129, 0.15)",
        "accent_text": "#ECFDF5",
    },
    "blue": {
        "accent": "#3B82F6",        # Blue 500
        "accent_hover": "#2563EB",  # Blue 600
        "accent_dim": "rgba(59, 130, 246, 0.15)",
        "accent_text": "#EFF6FF",
    },
}


def load_stylesheet(screen_color: str = "green") -> str:
    palette = ACCENT_COLORS.get(screen_color, ACCENT_COLORS["green"])
    accent = palette["accent"]
    accent_hover = palette["accent_hover"]
    accent_dim = palette["accent_dim"]
    accent_text = palette["accent_text"]

    return f"""
/* Global Reset & Base Fonts */
QWidget {{
    background-color: #0B0C0E;
    color: #F3F4F6;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
    font-size: 12px;
    selection-background-color: {accent};
    selection-color: #0B0C0E;
}}

QMainWindow, QDialog {{
    background-color: #0B0C0E;
}}

/* Scrollbars */
QScrollBar:vertical {{
    background: #0B0C0E;
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #22262E;
    min-height: 24px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{
    background: #374151;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background: #0B0C0E;
    height: 8px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: #22262E;
    min-width: 24px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal:hover {{
    background: #374151;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* Group Boxes / Panels */
QGroupBox {{
    background-color: #121418;
    border: 1px solid #22262E;
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 10px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #9CA3AF;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 10px;
    background-color: #0B0C0E;
    color: {accent};
}}

/* PushButtons */
QPushButton {{
    background-color: #181B20;
    color: #F3F4F6;
    border: 1px solid #282D37;
    border-radius: 4px;
    padding: 5px 12px;
    font-weight: 600;
    font-size: 11px;
}}
QPushButton:hover {{
    background-color: #22262E;
    border-color: #374151;
    color: #FFFFFF;
}}
QPushButton:pressed {{
    background-color: #14161A;
    border-color: {accent};
}}
QPushButton:checked {{
    background-color: {accent_dim};
    border-color: {accent};
    color: {accent_text};
}}
QPushButton:disabled {{
    background-color: #101215;
    border-color: #1B1E24;
    color: #4B5563;
}}

/* Special Primary Run Button */
QPushButton#primaryRunBtn {{
    background-color: {accent};
    color: #0B0C0E;
    font-weight: 700;
    border: 1px solid {accent_hover};
}}
QPushButton#primaryRunBtn:hover {{
    background-color: {accent_hover};
    color: #FFFFFF;
}}

/* Input Controls */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: #181B20;
    color: #F3F4F6;
    border: 1px solid #282D37;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {accent};
    background-color: #1C2026;
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #9CA3AF;
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: #181B20;
    border: 1px solid #282D37;
    selection-background-color: {accent_dim};
    selection-color: #FFFFFF;
}}

/* Checkboxes */
QCheckBox {{
    color: #D1D5DB;
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid #374151;
    border-radius: 3px;
    background-color: #181B20;
}}
QCheckBox::indicator:hover {{
    border-color: {accent};
}}
QCheckBox::indicator:checked {{
    background-color: {accent};
    border-color: {accent};
}}

/* Sliders */
QSlider::groove:horizontal {{
    height: 4px;
    background: #22262E;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {accent};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: #F3F4F6;
    border: 1px solid #374151;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{
    background: #FFFFFF;
    border-color: {accent};
}}

/* Toolbars / Status Bar */
QStatusBar {{
    background-color: #121418;
    border-top: 1px solid #22262E;
    color: #9CA3AF;
}}
QProgressBar {{
    background-color: #181B20;
    border: 1px solid #282D37;
    border-radius: 3px;
    text-align: center;
    color: #F3F4F6;
    font-size: 10px;
    font-weight: 600;
}}
QProgressBar::chunk {{
    background-color: {accent};
    border-radius: 2px;
}}

/* Splitter */
QSplitter::handle {{
    background-color: #181B20;
}}
QSplitter::handle:hover {{
    background-color: {accent};
}}

/* Table Widget */
QTableWidget {{
    background-color: #121418;
    border: 1px solid #22262E;
    gridline-color: #1F232B;
    color: #F3F4F6;
}}
QHeaderView::section {{
    background-color: #181B20;
    color: #9CA3AF;
    font-weight: 600;
    padding: 4px;
    border: none;
    border-right: 1px solid #22262E;
    border-bottom: 1px solid #22262E;
}}
"""
