"""
TREK Traceability -- Theme definitions & application
====================================================
Centralizes the app's colour palette so the whole UI can be re-skinned from
a single place, and adds a set of selectable themes.

Design goals
------------
* Preserve the app's ORIGINAL look. The default theme ("Charcoal") uses the
  exact colours trek_gui.py shipped with, so nothing changes visually unless
  the user picks a different theme.
* Provide additional themes inspired by the reference app
  (TREKsynch/qtools_trek/core/theme.py): dark and light variants with
  professional accent colours.
* Keep it simple: each theme is just a flat set of named colours. trek_gui.py
  reads the ACTIVE theme's colours into its module-level constants
  (DARK_BG, ACCENT, ...) at import/startup and rebuilds its QSS from them,
  so all 100+ existing f-string styles keep working unchanged.

How trek_gui.py uses this
-------------------------
    import trek_theme
    theme = trek_theme.get_active()          # dict of colours
    # ... module constants (DARK_BG, ACCENT, ...) are filled from `theme`
    app.setStyleSheet(trek_theme.build_stylesheet(theme))

The chosen theme name is persisted (next to the other app state) so it
survives restarts. Because trek_gui.py bakes colours into widgets at
construction time, switching themes at runtime re-applies the global
stylesheet immediately (covers most widgets) and is fully correct after a
restart -- see set_active()/needs_restart_note().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import trek_paths

# ---------------------------------------------------------------------------
# Theme palette definitions
# ---------------------------------------------------------------------------
# Every theme MUST define the same set of keys (the ones trek_gui.py's
# module-level colour constants map to), so switching themes is a pure
# value swap. Keys mirror the original trek_gui.py constants:
#   bg, panel, accent, accent_hover, success, warning, danger,
#   text, text_dim, border, row_alt
#
# Optional keys used only by build_stylesheet for finer control fall back to
# derived values when absent:
#   header_bg (defaults to panel), input_bg (defaults to panel)

THEMES: Dict[str, Dict[str, str]] = {
    # --- The ORIGINAL app palette, unchanged. Keep as the default so the
    #     app looks identical to before unless the user opts into another. ---
    "Charcoal (Default)": {
        "bg":           "#1e1e2e",
        "panel":        "#252535",
        "accent":       "#7c6af7",
        "accent_hover": "#9d8ff9",
        "success":      "#3ddc84",
        "warning":      "#f5c518",
        "danger":       "#ff6b6b",
        "text":         "#e0e0f0",
        "text_dim":     "#888899",
        "border":       "#3a3a55",
        "row_alt":      "#2a2a3e",
        "header_bg":    "#252535",
    },
    # --- Dark, cooler blue accent (reference "Dark"). ---
    "Slate Blue": {
        "bg":           "#1e232b",
        "panel":        "#262c35",
        "accent":       "#3b82f6",
        "accent_hover": "#2563eb",
        "success":      "#22c55e",
        "warning":      "#eab308",
        "danger":       "#ef4444",
        "text":         "#e6ebf1",
        "text_dim":     "#94a3b8",
        "border":       "#3a4450",
        "row_alt":      "#2b323c",
        "header_bg":    "#242b34",
    },
    # --- Very dark indigo (reference "Midnight"). ---
    "Midnight": {
        "bg":           "#0f1420",
        "panel":        "#171e2e",
        "accent":       "#6366f1",
        "accent_hover": "#4f46e5",
        "success":      "#34d399",
        "warning":      "#fbbf24",
        "danger":       "#f87171",
        "text":         "#e2e8f0",
        "text_dim":     "#8b95a7",
        "border":       "#2a3348",
        "row_alt":      "#1c2438",
        "header_bg":    "#151c2c",
    },
    # --- Nord palette (reference "Nord"): muted arctic dark. ---
    "Nord": {
        "bg":           "#2e3440",
        "panel":        "#3b4252",
        "accent":       "#88c0d0",
        "accent_hover": "#8fbcbb",
        "success":      "#a3be8c",
        "warning":      "#ebcb8b",
        "danger":       "#bf616a",
        "text":         "#eceff4",
        "text_dim":     "#9aa4b5",
        "border":       "#4c566a",
        "row_alt":      "#434c5e",
        "header_bg":    "#353c4a",
    },
    # --- Light theme (reference "Light"): clean professional light UI. ---
    "Light": {
        "bg":           "#f4f6f8",
        "panel":        "#ffffff",
        "accent":       "#2563eb",
        "accent_hover": "#1d4ed8",
        "success":      "#16a34a",
        "warning":      "#d97706",
        "danger":       "#dc2626",
        "text":         "#1a1f24",
        "text_dim":     "#6b7280",
        "border":       "#d8dee6",
        "row_alt":      "#eef1f4",
        "header_bg":    "#ffffff",
    },
    # --- Ocean (reference "Ocean"): light with teal accent. ---
    "Ocean": {
        "bg":           "#e8f4f8",
        "panel":        "#ffffff",
        "accent":       "#0891b2",
        "accent_hover": "#0e7490",
        "success":      "#0d9488",
        "warning":      "#d97706",
        "danger":       "#dc2626",
        "text":         "#0b2530",
        "text_dim":     "#5a7684",
        "border":       "#c2dae3",
        "row_alt":      "#dcedf3",
        "header_bg":    "#ffffff",
    },
    # --- Forest (reference "Forest"): light with green accent. ---
    "Forest": {
        "bg":           "#eef3ec",
        "panel":        "#ffffff",
        "accent":       "#16a34a",
        "accent_hover": "#15803d",
        "success":      "#16a34a",
        "warning":      "#ca8a04",
        "danger":       "#dc2626",
        "text":         "#1a2b1a",
        "text_dim":     "#5c7a5c",
        "border":       "#cdddc8",
        "row_alt":      "#e2ebe0",
        "header_bg":    "#ffffff",
    },
    # --- Sunset (reference "Sunset"): warm light with orange accent. ---
    "Sunset": {
        "bg":           "#fdf2ec",
        "panel":        "#ffffff",
        "accent":       "#ea580c",
        "accent_hover": "#c2410c",
        "success":      "#16a34a",
        "warning":      "#d97706",
        "danger":       "#dc2626",
        "text":         "#33211a",
        "text_dim":     "#8a6a5a",
        "border":       "#f0d6c4",
        "row_alt":      "#fae6da",
        "header_bg":    "#ffffff",
    },
}

DEFAULT_THEME = "Charcoal (Default)"


def _is_dark(hex_color: str) -> bool:
    """Rough luminance test: True if the colour is dark (needs light text)."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        # perceived luminance
        return (0.299 * r + 0.587 * g + 0.114 * b) < 128
    except (ValueError, IndexError):
        return True


def code_bg(theme: Dict[str, str]) -> str:
    """Background for monospace 'code' boxes in the HTML detail renderers.
    Explicit per-theme override via 'code_bg', else derived: on dark themes a
    slightly darker shade than the panel; on light themes a soft grey so the
    box is visible but text stays dark-on-light (readable)."""
    if theme.get("code_bg"):
        return theme["code_bg"]
    return "#1a1a2e" if _is_dark(theme["bg"]) else "#f0f2f5"


def req_color(theme: Dict[str, str]) -> str:
    """Accent colour used for requirement (SYR/SWR) headings in the HTML
    detail renderers. Overridable via 'req_color'; defaults to a blue that
    reads on the theme's panel background."""
    if theme.get("req_color"):
        return theme["req_color"]
    return "#7cb9e8" if _is_dark(theme["bg"]) else "#1d6fb8"


def swr_color(theme: Dict[str, str]) -> str:
    """Colour for SWR (software requirement) nodes/headings. Was hardcoded
    yellow (#f5c518) which is nearly invisible on light backgrounds. Derive
    a readable amber/gold: bright on dark themes, darker on light."""
    if theme.get("swr_color"):
        return theme["swr_color"]
    return "#f5c518" if _is_dark(theme["bg"]) else "#b8860b"


def related_color(theme: Dict[str, str]) -> str:
    """Colour for 'related SYT' nodes/headings. Was hardcoded light lavender
    (#c792ea) which washes out on light backgrounds. Derive a readable
    purple: light lavender on dark themes, deeper purple on light."""
    if theme.get("related_color"):
        return theme["related_color"]
    return "#c792ea" if _is_dark(theme["bg"]) else "#8e44ad"


def amber_color(theme: Dict[str, str]) -> str:
    """Generic amber used for 'similar' stats. Readable on both."""
    if theme.get("amber_color"):
        return theme["amber_color"]
    return "#e8b84b" if _is_dark(theme["bg"]) else "#b8860b"


def danger_text_color(theme: Dict[str, str]) -> str:
    """Red used as TEXT (not a button background) -- e.g. 'Duplicate' /
    'LLM: Same scenario' classification labels. theme['danger'] is tuned
    for white-on-red buttons, which on light themes (e.g. #dc2626) is
    still fairly washed out as small coloured TEXT on a white/panel
    background (~4.8:1, and the classic hardcoded #e74c3c it replaces was
    only ~3.8:1). Darken further for light themes so the label reads
    clearly at small sizes; dark themes keep the bright theme danger."""
    if theme.get("danger_text_color"):
        return theme["danger_text_color"]
    return theme["danger"] if _is_dark(theme["bg"]) else "#b91c1c"


def success_text_color(theme: Dict[str, str]) -> str:
    """Green used as TEXT -- e.g. 'Distinct' / 'LLM: Different scenario'
    classification labels. theme['success'] (e.g. #16a34a, ~3.3:1 on
    white) is fine as a button fill but too light for small text on a
    light background -- and the hardcoded #2ecc71 it replaces was only
    ~2.1:1 (nearly invisible). Darken further for light themes; dark
    themes keep the bright theme success colour."""
    if theme.get("success_text_color"):
        return theme["success_text_color"]
    return theme["success"] if _is_dark(theme["bg"]) else "#15803d"


def _mix(hex_a: str, hex_b: str, t: float) -> str:
    """Linear blend of two hex colours (t=0 -> a, t=1 -> b)."""
    try:
        a = hex_a.lstrip("#"); b = hex_b.lstrip("#")
        ar, ag, ab = int(a[0:2], 16), int(a[2:4], 16), int(a[4:6], 16)
        br, bg, bb = int(b[0:2], 16), int(b[2:4], 16), int(b[4:6], 16)
        r = round(ar + (br - ar) * t)
        g = round(ag + (bg - ag) * t)
        bl = round(ab + (bb - ab) * t)
        return f"#{r:02x}{g:02x}{bl:02x}"
    except (ValueError, IndexError):
        return hex_b


def selection_bg(theme: Dict[str, str]) -> str:
    """Row-selection background. A soft accent tint (accent blended toward the
    panel) rather than a solid accent fill, so an item's OWN foreground colour
    (set via setForeground: gold SWR, purple related-SYT, dim text, etc.)
    stays readable when selected -- solving the 'white text on light accent'
    invisibility on light themes."""
    if theme.get("selection_bg"):
        return theme["selection_bg"]
    # A SUBTLE accent tint -- enough to see which row is selected, but
    # not so strong that it drowns out the item's own foreground colours
    # (gold SWR, green SWT, purple related-SYT, red missing, etc.).
    # Light themes: 15% accent (very faint wash).
    # Dark themes:  20% accent (slightly more visible on dark panels).
    pct = 0.15 if not _is_dark(theme["bg"]) else 0.20
    return _mix(theme["panel"], theme["accent"], pct)


def on_accent_text(theme: Dict[str, str]) -> str:
    """Readable text/icon colour to place ON TOP of the theme's accent
    colour (e.g. a combobox popup's selected-row text, or a filled
    checkbox's tick). Most themes use a saturated-dark accent where white
    reads fine, but light-accent themes (e.g. Nord's pale teal #88c0d0,
    ~2:1 contrast with white) need dark text instead -- this picks
    whichever of white/near-black has better contrast against the accent,
    so no single theme needs a manual override."""
    if theme.get("on_accent_text"):
        return theme["on_accent_text"]
    return "#1a1a1a" if not _is_dark(theme["accent"]) else "#ffffff"


def dim_text(theme: Dict[str, str]) -> str:
    """A MORE READABLE 'dim' text colour than text_dim. text_dim is very low
    contrast on light themes (pale grey on white). This nudges it toward the
    main text colour so secondary labels stay legible."""
    if theme.get("dim_text"):
        return theme["dim_text"]
    # blend text_dim 45% toward full text for better contrast on both.
    return _mix(theme["text_dim"], theme["text"], 0.45)


# ---------------------------------------------------------------------------
# Checkbox tick icons rendered as real PNG files.
# ---------------------------------------------------------------------------
# Qt's stylesheet engine does NOT reliably render inline SVG data URIs
# (utf8 OR base64) for QCheckBox/indicator images -- it silently drops them,
# leaving just the coloured square. Rendering a real PNG with QPainter and
# referencing it by absolute file path in the QSS is the only fully reliable
# approach. We render once per app run into the data dir and reuse the paths.
_ICON_CACHE: Dict[str, str] = {}


def _icon_path(kind: str, color: str = "white") -> str:
    """Return an absolute file path to a 16x16 PNG icon on a transparent
    background, generating it on first use.

    kind: 'check' | 'dash' | 'arrow_right' | 'arrow_down' | 'chevron_down' | 'chevron_up'
    color: hex colour string for the icon (e.g. '#333333' for dark arrows
           on light themes, 'white' for dark themes).

    Paths use forward slashes (Qt QSS url() requirement on Windows).

    IMPORTANT: constructing a QPixmap before a QApplication/QGuiApplication
    exists is a Qt FATAL error (qFatal aborts the process -- it is NOT a
    catchable Python exception). trek_gui.py builds the initial STYLESHEET
    at module-import time, before QApplication is created, so this function
    MUST check for a live QApplication instance first and skip rendering
    (returning the last-known/empty path) rather than ever construct a
    QPixmap too early. main() rebuilds the stylesheet once QApplication
    exists, which then renders the real icons and everything picks them up.
    """
    # Cache key includes colour so dark/light variants coexist.
    cache_key = f"{kind}_{color}"
    try:
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            return _ICON_CACHE.get(cache_key, "")
    except Exception:
        return _ICON_CACHE.get(cache_key, "")

    if cache_key in _ICON_CACHE:
        return _ICON_CACHE[cache_key]
    try:
        from PySide6.QtGui import QPixmap, QPainter, QPen, QColor
        from PySide6.QtCore import Qt

        size = 16
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen(QColor(color))
        pen.setWidth(2)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        if kind == "check":
            p.drawLine(4, 9, 7, 12)
            p.drawLine(7, 12, 12, 5)
        elif kind == "dash":
            p.drawLine(4, 8, 12, 8)
        elif kind == "arrow_right":
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(color))
            p.drawPolygon(QPolygon([QPoint(5, 3), QPoint(12, 8), QPoint(5, 13)]))
        elif kind == "arrow_down":
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(color))
            p.drawPolygon(QPolygon([QPoint(3, 5), QPoint(13, 5), QPoint(8, 12)]))
        elif kind == "chevron_down":      # combo-box arrow (closed)
            p.drawLine(4, 6, 8, 10)
            p.drawLine(8, 10, 12, 6)
        elif kind == "chevron_up":        # combo-box arrow (open)
            p.drawLine(4, 10, 8, 6)
            p.drawLine(8, 6, 12, 10)
        p.end()

        fname = f"_trek_{kind}_{color.replace('#', '')}.png"
        out = trek_paths.icon_cache_dir() / fname
        pix.save(str(out), "PNG")
        path = str(out).replace("\\", "/")
        _ICON_CACHE[cache_key] = path
        return path
    except Exception:
        return ""

# Persist the chosen theme next to the other per-machine app state
# (trek_projects.json etc.) via trek_paths.data_file so it survives restarts.
_SETTINGS_PATH = trek_paths.data_file("trek_settings.json")


def theme_names() -> List[str]:
    """Return the list of selectable theme names (default listed first)."""
    names = list(THEMES.keys())
    if DEFAULT_THEME in names:
        names.remove(DEFAULT_THEME)
        names.insert(0, DEFAULT_THEME)
    return names


def get_theme(name: str) -> Dict[str, str]:
    """Return the colour dict for `name`, falling back to the default."""
    return THEMES.get(name, THEMES[DEFAULT_THEME])


# ---------------------------------------------------------------------------
# Persistence of the chosen theme name
# ---------------------------------------------------------------------------
def _load_settings() -> dict:
    try:
        if _SETTINGS_PATH.exists():
            with open(_SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _save_settings(data: dict) -> None:
    try:
        with open(_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def get_active_name() -> str:
    """Return the persisted active theme name (default if unset/unknown)."""
    name = _load_settings().get("theme")
    return name if name in THEMES else DEFAULT_THEME


def get_active() -> Dict[str, str]:
    """Return the colour dict for the persisted active theme."""
    return get_theme(get_active_name())


def set_active(name: str) -> None:
    """Persist `name` as the active theme (no-op if unknown)."""
    if name not in THEMES:
        return
    data = _load_settings()
    data["theme"] = name
    _save_settings(data)


def needs_restart_note() -> str:
    """Human-readable note about theme-switch behaviour, shown in the UI."""
    return "The new theme is applied now."


# ---------------------------------------------------------------------------
# Stylesheet builder -- mirrors the ORIGINAL trek_gui.py STYLESHEET exactly,
# but driven by the active theme's colours instead of hardcoded values, so
# the default theme is byte-for-byte the original look.
# ---------------------------------------------------------------------------
def build_stylesheet(theme: Dict[str, str]) -> str:
    DARK_BG      = theme["bg"]
    PANEL_BG     = theme["panel"]
    ACCENT       = theme["accent"]
    ACCENT_HOVER = theme["accent_hover"]
    SUCCESS      = theme["success"]
    WARNING      = theme["warning"]           # noqa: F841 (kept for parity)
    DANGER       = theme["danger"]            # noqa: F841 (kept for parity)
    TEXT         = theme["text"]
    TEXT_DIM     = theme["text_dim"]
    BORDER       = theme["border"]
    ROW_ALT      = theme["row_alt"]
    HEADER_SECTION_BG = theme.get("header_bg", PANEL_BG)
    SELECTION_BG = selection_bg(theme)
    DIM_TEXT     = dim_text(theme)
    ON_ACCENT_TEXT = on_accent_text(theme)
    # Icon colour: white on dark themes, dark grey on light themes.
    _icon_color = "#333333" if not _is_dark(DARK_BG) else "white"
    CHECK_ICON = _icon_path("check", _icon_color)
    DASH_ICON = _icon_path("dash", _icon_color)
    ARROW_RIGHT_ICON = _icon_path("arrow_right", _icon_color)
    ARROW_DOWN_ICON = _icon_path("arrow_down", _icon_color)
    # Drop-down chevrons: dim when idle, full text colour on hover, accent
    # and pointing UP while the list is open. Real images -- the previous
    # CSS "border triangle" trick rendered as a flat bar on Windows.
    COMBO_ARROW_ICON = _icon_path("chevron_down", TEXT_DIM)
    COMBO_ARROW_HOVER_ICON = _icon_path("chevron_down", TEXT)
    COMBO_ARROW_OPEN_ICON = _icon_path("chevron_up", ACCENT)
    # Spin-box (number field) step arrows -- same chevrons, same reason.
    SPIN_UP_ICON = _icon_path("chevron_up", TEXT_DIM)
    SPIN_DOWN_ICON = _icon_path("chevron_down", TEXT_DIM)
    SPIN_UP_HOVER_ICON = _icon_path("chevron_up", ACCENT)
    SPIN_DOWN_HOVER_ICON = _icon_path("chevron_down", ACCENT)
    SPIN_UP_OFF_ICON = _icon_path("chevron_up", BORDER)      # at max / disabled
    SPIN_DOWN_OFF_ICON = _icon_path("chevron_down", BORDER)  # at min / disabled

    return f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT};
    font-family: 'Inter', 'Segoe UI Variable', 'Segoe UI', system-ui, sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    font-weight: bold;
    color: {ACCENT};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
QPushButton {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {ACCENT_HOVER};
}}
QPushButton:disabled {{
    background-color: {ROW_ALT};
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
}}
QPushButton#btn_secondary {{
    background-color: {PANEL_BG};
    color: {TEXT};
    border: 1px solid {BORDER};
}}
QPushButton#btn_secondary:hover {{
    background-color: {BORDER};
    border-color: {ACCENT};
}}
QPushButton#btn_success {{
    background-color: {SUCCESS};
    color: white;
}}
QPushButton#btn_success:hover {{
    background-color: {SUCCESS};
    color: white;
}}
QListWidget, QTableWidget, QTreeWidget {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    alternate-background-color: {ROW_ALT};
    color: {TEXT};
    gridline-color: {BORDER};
    selection-background-color: {SELECTION_BG};
    selection-color: {TEXT};
    outline: none;
}}
QListWidget::item:selected, QTableWidget::item:selected, QTreeWidget::item:selected {{
    /* Soft accent-tinted highlight (not a solid accent fill), so each item's
       OWN foreground colour stays readable when selected -- fixes white-on-
       light-accent invisibility on light themes. */
    background-color: {SELECTION_BG};
    color: {TEXT};
}}
QListWidget::item:hover, QTableWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {ROW_ALT};
}}
QListWidget::item, QTreeWidget::item {{
    padding: 2px 0;
    min-height: 20px;
}}
/* Suppress ALL default branch decoration (dotted lines, squares, native
   expand/collapse icons) -- replaced with rendered PNG arrow icons below. */
QTreeView::branch {{
    background: transparent;
    border: none;
    border-image: none;
    image: none;
}}
/* Expand/collapse indicators for expandable tree rows (SYT/SYR/SWR etc.).
   Closed = right-pointing arrow ▶, open = down-pointing arrow ▼.
   Uses rendered 16x16 PNG icons (same approach as checkbox tick marks)
   instead of CSS border triangles, because Qt on Windows doesn't reliably
   suppress native branch decorations with border-image: none alone. */
QTreeView::branch:has-children:!has-siblings:closed,
QTreeView::branch:closed:has-children:has-siblings {{
    border: none;
    border-image: none;
    image: url("{ARROW_RIGHT_ICON}");
}}
QTreeView::branch:open:has-children:!has-siblings,
QTreeView::branch:open:has-children:has-siblings {{
    border: none;
    border-image: none;
    image: url("{ARROW_DOWN_ICON}");
}}
QHeaderView::section {{
    background-color: {HEADER_SECTION_BG};
    color: {DIM_TEXT};
    padding: 7px 8px;
    border: none;
    border-bottom: 1px solid {BORDER};
    border-right: 1px solid {BORDER};
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.3px;
}}
QLineEdit {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    color: {TEXT};
    selection-background-color: {SELECTION_BG};
    selection-color: {TEXT};
}}
QLineEdit:hover {{
    border-color: {ACCENT};
}}
QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}
QSpinBox, QDoubleSpinBox {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    color: {TEXT};
    min-height: 20px;
}}
QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {ACCENT};
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {ACCENT};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid {BORDER};
    border-top-right-radius: 5px;
    background: transparent;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid {BORDER};
    border-bottom-right-radius: 5px;
    background: transparent;
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {ROW_ALT};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url("{SPIN_UP_ICON}");
    width: 11px;
    height: 11px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url("{SPIN_DOWN_ICON}");
    width: 11px;
    height: 11px;
}}
QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{
    image: url("{SPIN_UP_HOVER_ICON}");
}}
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
    image: url("{SPIN_DOWN_HOVER_ICON}");
}}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled,
QSpinBox::up-arrow:off, QDoubleSpinBox::up-arrow:off {{
    image: url("{SPIN_UP_OFF_ICON}");
}}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled,
QSpinBox::down-arrow:off, QDoubleSpinBox::down-arrow:off {{
    image: url("{SPIN_DOWN_OFF_ICON}");
}}
QTextEdit {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    color: {TEXT};
    font-family: 'Consolas', monospace;
    font-size: 12px;
    selection-background-color: {SELECTION_BG};
    selection-color: {TEXT};
}}
QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    background-color: {PANEL_BG};
    text-align: center;
    color: {TEXT};
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}
QSplitter::handle {{
    background-color: {BORDER};
    width: 2px;
    height: 2px;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 32px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT};
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 5px;
    min-width: 32px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {ACCENT};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0px; width: 0px; background: none; border: none;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}
QStatusBar {{
    background-color: {DARK_BG};
    color: {TEXT_DIM};
    border-top: 1px solid {BORDER};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: {PANEL_BG};
    top: -1px;
}}
QTabBar::tab {{
    background: {DARK_BG};
    color: {TEXT_DIM};
    padding: 8px 20px;
    margin-right: 2px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-radius: 7px 7px 0 0;
    font-weight: 500;
}}
QTabBar::tab:hover {{
    color: {TEXT};
    background: {ROW_ALT};
}}
QTabBar::tab:selected {{
    background: {PANEL_BG};
    color: {ACCENT};
    font-weight: 600;
    border-bottom: 2px solid {ACCENT};
}}
QLabel#step_label {{
    color: {ACCENT};
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
QLabel#title_label {{
    color: {TEXT};
    font-size: 15px;
    font-weight: bold;
}}
QFrame#divider {{
    background-color: {BORDER};
    max-width: 1px;
}}
QCheckBox {{
    color: {TEXT};
    spacing: 7px;
}}
QCheckBox::indicator {{
    width: 17px;
    height: 17px;
    border: 1.5px solid {BORDER};
    border-radius: 4px;
    background: {PANEL_BG};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT};
    background: {ROW_ALT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: url("{CHECK_ICON}");
}}
QCheckBox::indicator:indeterminate {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: url("{DASH_ICON}");
}}
/* Checkable TREE items (e.g. the Step-2 test-case tree) use the same tick. */
QComboBox QAbstractItemView::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {PANEL_BG};
    margin-right: 6px;
}}
QComboBox QAbstractItemView::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: url("{CHECK_ICON}");
}}
QTreeWidget::indicator, QTreeView::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {PANEL_BG};
}}
QTreeWidget::indicator:checked, QTreeView::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: url("{CHECK_ICON}");
}}
QTreeWidget::indicator:indeterminate, QTreeView::indicator:indeterminate {{
    background: {ACCENT};
    border-color: {ACCENT};
    image: url("{DASH_ICON}");
}}
QComboBox {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 5px 14px 5px 12px;
    color: {TEXT};
    min-height: 22px;
    font-weight: 500;
}}
QComboBox:hover {{
    border-color: {ACCENT};
    background-color: {ROW_ALT};
}}
QComboBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox:on {{
    border: 1px solid {ACCENT};
    border-bottom-left-radius: 0;
    border-bottom-right-radius: 0;
}}
QComboBox:disabled {{
    color: {TEXT_DIM};
    background-color: {DARK_BG};
    border-color: {BORDER};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 28px;
    border-left: none;
    border-top-right-radius: 7px;
    border-bottom-right-radius: 7px;
    background-color: transparent;
}}
QComboBox::down-arrow {{
    image: url("{COMBO_ARROW_ICON}");
    width: 14px;
    height: 14px;
    margin-right: 8px;
}}
QComboBox::down-arrow:hover {{
    image: url("{COMBO_ARROW_HOVER_ICON}");
}}
QComboBox::down-arrow:on {{
    image: url("{COMBO_ARROW_OPEN_ICON}");
}}
/* Popup list -- polished dropdown menu look: generous padding, smooth
   accent hover/selection, rounded items, and a text colour that adapts
   to the accent brightness via ON_ACCENT_TEXT. */
QComboBox QAbstractItemView {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-top: 2px solid {ACCENT};
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
    selection-background-color: {ACCENT};
    selection-color: {ON_ACCENT_TEXT};
    color: {TEXT};
    padding: 4px 5px;
    outline: none;
}}
QComboBox QAbstractItemView::item {{
    min-height: 28px;
    padding: 5px 12px;
    border-radius: 5px;
    margin: 1px 0px;
}}
QComboBox QAbstractItemView::item:hover {{
    background-color: {SELECTION_BG};
    color: {TEXT};
}}
QComboBox QAbstractItemView::item:selected {{
    background-color: {ACCENT};
    color: {ON_ACCENT_TEXT};
}}
"""
