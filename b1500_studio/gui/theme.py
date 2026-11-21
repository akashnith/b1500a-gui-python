"""Light visual theme: colour tokens and the Qt style sheet.

One palette drives both halves of the window.  The widget colours here and
the figure colours in :mod:`b1500_studio.plotting` are stepped from the same
light-mode reference set, so a curve on the canvas and the chip beside it in
the navigator are the same blue rather than two blues that drift apart.

Status colours are never used alone: every place one appears it sits beside a
word, because a reader who cannot separate red from green still has to be
able to tell a failed verification from a passed one.
"""

from __future__ import annotations

__all__ = ["stylesheet", "FONT", "MONO"] + [
    "APP_BG", "SURFACE", "SURFACE_ALT", "SURFACE_SUNK",
    "BORDER", "BORDER_SOFT", "INK", "INK_SOFT", "INK_FAINT",
    "ACCENT", "ACCENT_DEEP", "ACCENT_WASH",
    "OK", "WARN", "CRIT", "OK_INK", "WARN_INK", "CRIT_INK", "IDLE",
]

# -- surfaces and ink -------------------------------------------------------

APP_BG = "#f1f1ef"        # window chrome, behind the cards
SURFACE = "#ffffff"       # cards, inputs, the plot canvas
SURFACE_ALT = "#f7f7f5"   # headers, alternating rows
SURFACE_SUNK = "#ebebe8"  # pressed / hovered chrome

BORDER = "#d8d7d2"
BORDER_SOFT = "#e7e6e2"

INK = "#17171a"           # primary text
INK_SOFT = "#52514e"      # labels, captions
INK_FAINT = "#8a8984"     # timestamps, placeholders, disabled

# -- accent -----------------------------------------------------------------

ACCENT = "#2a78d6"        # categorical slot 1
ACCENT_DEEP = "#1c5cab"
ACCENT_WASH = "#e9f1fc"

# -- status (fixed; always paired with a word) ------------------------------

OK = "#0ca30c"
WARN = "#fab219"
CRIT = "#d03b3b"
IDLE = "#a9a8a3"

# Darker steps for small text, where the dot colours alone are too light.
OK_INK = "#06690b"
WARN_INK = "#8a5a00"
CRIT_INK = "#b02a2a"

FONT = '"Segoe UI", "Inter", system-ui, sans-serif'
MONO = '"Cascadia Mono", "Consolas", monospace'


def stylesheet() -> str:
    """The application-wide Qt style sheet."""
    return f"""
/* ---------------------------------------------------------------- base */
QWidget {{
    background: {APP_BG};
    color: {INK};
    font-family: {FONT};
    font-size: 9pt;
}}
QToolTip {{
    background: {INK};
    color: #ffffff;
    border: none;
    padding: 5px 8px;
    font-size: 8.5pt;
}}

/* --------------------------------------------------------------- cards */
QFrame#card, QFrame#header, QFrame#actionbar {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QFrame#header {{ border-radius: 0; border-left: none;
                 border-right: none; border-top: none; }}
QFrame#actionbar {{ border-radius: 0; border-left: none;
                    border-right: none; border-bottom: none; }}

QLabel#appMark {{ background: transparent; padding-right: 2px; }}
QLabel#appTitle {{ font-size: 13pt; font-weight: 600; color: {INK};
                   background: transparent; }}
QLabel#appSubtitle {{ font-size: 8.5pt; color: {INK_FAINT};
                      background: transparent; }}
QLabel#pageTitle {{ font-size: 12pt; font-weight: 600; color: {INK};
                    background: transparent; }}
QLabel#pageDesc {{ font-size: 8.5pt; color: {INK_SOFT};
                   background: transparent; }}
QLabel#sectionLabel {{ font-size: 8pt; font-weight: 700; color: {INK_FAINT};
                       letter-spacing: 1px; background: transparent; }}
QLabel#hint {{ color: {INK_SOFT}; background: transparent; }}
QLabel#chip {{
    background: {ACCENT_WASH}; color: {ACCENT_DEEP};
    border: 1px solid #cadff7; border-radius: 9px;
    padding: 2px 9px; font-family: {MONO}; font-size: 8pt;
}}
QLabel#statusText {{ background: transparent; font-weight: 600; }}
QLabel#statusDot {{ background: transparent; font-size: 12pt; }}

/* --------------------------------------------------------- group boxes */
QGroupBox {{
    background: transparent;
    border: none;
    border-top: 1px solid {BORDER_SOFT};
    margin-top: 16px;
    padding-top: 4px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 0px;
    padding: 0 6px 0 0;
    color: {INK_FAINT};
    font-size: 8pt;
    font-weight: 700;
}}

/* -------------------------------------------------------------- inputs */
QLineEdit, QComboBox, QSpinBox, QTextEdit, QPlainTextEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 8px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}
QLineEdit:focus, QComboBox:focus, QTextEdit:focus {{
    border: 1px solid {ACCENT};
}}
QLineEdit:read-only {{ background: {SURFACE_ALT}; color: {INK_SOFT}; }}
QLineEdit:disabled, QComboBox:disabled {{
    background: {SURFACE_ALT}; color: {INK_FAINT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
/* No ::down-arrow rule: a border-triangle renders as a filled box on some
   styles, and the platform's own arrow is both correct and familiar. */
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_WASH};
    selection-color: {INK};
    outline: none;
}}
QCheckBox {{ background: transparent; spacing: 7px; }}
QCheckBox::indicator {{
    width: 15px; height: 15px;
    border: 1px solid {BORDER}; border-radius: 4px; background: {SURFACE};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT}; border-color: {ACCENT_DEEP};
}}

/* ------------------------------------------------------------- buttons */
QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 16px;
    color: {INK};
}}
QPushButton:hover {{ background: {SURFACE_ALT}; border-color: #c3c2bd; }}
QPushButton:pressed {{ background: {SURFACE_SUNK}; }}
QPushButton:disabled {{ color: {INK_FAINT}; border-color: {BORDER_SOFT};
                        background: {SURFACE_ALT}; }}

QPushButton#primary {{
    background: {ACCENT}; border: 1px solid {ACCENT_DEEP};
    color: #ffffff; font-weight: 600; padding: 6px 22px;
}}
QPushButton#primary:hover {{ background: {ACCENT_DEEP}; }}
QPushButton#primary:pressed {{ background: #174d90; }}
QPushButton#primary:disabled {{
    background: #b9cdea; border-color: #b9cdea; color: #ffffff;
}}
QPushButton#danger {{ color: {CRIT_INK}; }}
QPushButton#danger:hover {{ background: #fdeeee; border-color: #efbfbf; }}

/* ----------------------------------------------------------- navigator */
QTreeWidget#nav {{
    background: {SURFACE};
    border: none;
    outline: none;
    padding: 4px;
    show-decoration-selected: 1;
}}
QTreeWidget#nav::item {{
    padding: 5px 4px;
    border-radius: 5px;
    color: {INK};
}}
QTreeWidget#nav::item:hover {{ background: {SURFACE_ALT}; }}
QTreeWidget#nav::item:selected {{
    background: {ACCENT_WASH}; color: {ACCENT_DEEP};
}}
QTreeWidget#nav::branch {{ background: {SURFACE}; }}

/* ---------------------------------------------------------------- tabs */
QTabWidget::pane {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    top: -1px;
}}
QTabBar {{ background: transparent; qproperty-drawBase: 0; }}
QTabBar::tab {{
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 7px 14px;
    margin-right: 2px;
    color: {INK_SOFT};
}}
QTabBar::tab:hover {{ color: {INK}; }}
QTabBar::tab:selected {{
    color: {ACCENT_DEEP};
    border-bottom: 2px solid {ACCENT};
    font-weight: 600;
}}

/* -------------------------------------------------------------- tables */
QTableWidget, QTableView {{
    background: {SURFACE};
    alternate-background-color: {SURFACE_ALT};
    border: none;
    gridline-color: {BORDER_SOFT};
    selection-background-color: {ACCENT_WASH};
    selection-color: {INK};
}}
QHeaderView::section {{
    background: {SURFACE_ALT};
    border: none;
    border-right: 1px solid {BORDER_SOFT};
    border-bottom: 1px solid {BORDER};
    padding: 5px 7px;
    color: {INK_SOFT};
    font-weight: 600;
}}
QTableCornerButton::section {{
    background: {SURFACE_ALT}; border: none;
    border-bottom: 1px solid {BORDER};
}}

/* --------------------------------------------------------------- misc. */
QTextEdit#log {{
    background: {SURFACE};
    border: none;
    font-family: {MONO};
    font-size: 8.5pt;
}}
QProgressBar {{
    border: none; border-radius: 3px; height: 6px;
    background: {SURFACE_SUNK}; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

QSplitter::handle {{ background: {APP_BG}; }}
QSplitter::handle:horizontal {{ width: 8px; }}
QSplitter::handle:hover {{ background: {BORDER_SOFT}; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{
    background: transparent; width: 11px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #cdccc7; border-radius: 4px; min-height: 28px;
}}
QScrollBar::handle:vertical:hover {{ background: #b4b3ae; }}
QScrollBar:horizontal {{
    background: transparent; height: 11px; margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #cdccc7; border-radius: 4px; min-width: 28px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""
