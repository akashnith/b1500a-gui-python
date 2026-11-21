"""Declarative multi-panel plotting, rendered with matplotlib.

Port of the ``Series`` / ``Panel`` model from ``pkg/src/plotwidget.h``.  Each
experiment *describes* its panels; this module is the only place that knows
how to draw them.

Two themes ship: ``light`` (the default, and what the GUI uses) and ``dark``,
the palette the C++ application draws with.  The series hues are the
light-mode validated categorical set, assigned in a fixed order and never
cycled past slot 8, so the same series index is the same colour in every
figure the program produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.ticker

__all__ = [
    "Series",
    "Panel",
    "PanelGrid",
    "palette",
    "render",
    "save",
    "without_text_panels",
    "has_text_panels",
    "set_theme",
    "theme_name",
    "BG",
    "PANEL_BG",
    "FG",
    "MUTED",
    "GRIDC",
    "FRAME",
]

#: Validated categorical hues, one per series slot, in a fixed order.  The
#: order is the colour-blind-safety mechanism, not decoration: adjacent slots
#: are the pairs a line chart puts next to each other.
_LIGHT_SERIES = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#6250d6",  # 7 violet
    "#e34948",  # 8 red
)

#: The same eight hues stepped for a dark surface -- not a second palette.
_DARK_SERIES = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#0ca30c",
    "#9085e9",
    "#e66767",
)

_THEMES = {
    "light": {
        "BG": "#ffffff",
        "PANEL_BG": "#fcfcfb",
        "FG": "#17171a",
        "MUTED": "#52514e",
        "GRIDC": "#e8e7e3",
        "FRAME": "#d8d7d2",
        "SERIES": _LIGHT_SERIES,
    },
    # The palette plotwidget.cpp draws with, kept so a figure can be made to
    # match the C++ application when the two are compared side by side.
    "dark": {
        "BG": "#0f131a",
        "PANEL_BG": "#171d28",
        "FG": "#e8ecf3",
        "MUTED": "#8793a8",
        "GRIDC": "#2a3550",
        "FRAME": "#33405a",
        "SERIES": _DARK_SERIES,
    },
}

_ACTIVE = "light"

BG = _THEMES[_ACTIVE]["BG"]
PANEL_BG = _THEMES[_ACTIVE]["PANEL_BG"]
FG = _THEMES[_ACTIVE]["FG"]
MUTED = _THEMES[_ACTIVE]["MUTED"]
GRIDC = _THEMES[_ACTIVE]["GRIDC"]

#: Longest panel title, in characters, that a folded-in series
#: name may produce before the legend box is kept instead.
_TITLE_FOLD_CHARS = 44

#: Decades a log axis may span before the bottom is clipped away.
_LOG_DECADES = 8
FRAME = _THEMES[_ACTIVE]["FRAME"]
_PALETTE = _THEMES[_ACTIVE]["SERIES"]


def set_theme(name: str) -> None:
    """Switch every figure colour to theme ``name`` (``light`` or ``dark``).

    ``render`` and ``palette`` read these names when they are called, not when
    this module is imported, so a switch applies to the next figure drawn.
    Importers that bind the constants directly (``from .plotting import BG``)
    keep whatever was current at their import, which is why the GUI reads them
    through the module instead.
    """
    global _ACTIVE, BG, PANEL_BG, FG, MUTED, GRIDC, FRAME, _PALETTE
    try:
        t = _THEMES[name]
    except KeyError:
        raise ValueError(
            f"unknown theme {name!r}; choose from {', '.join(_THEMES)}"
        ) from None
    _ACTIVE = name
    BG, PANEL_BG = t["BG"], t["PANEL_BG"]
    FG, MUTED = t["FG"], t["MUTED"]
    GRIDC, FRAME = t["GRIDC"], t["FRAME"]
    _PALETTE = t["SERIES"]


def theme_name() -> str:
    """The theme :func:`set_theme` last selected."""
    return _ACTIVE


def palette(i: int) -> str:
    """Series colour ``i``, wrapping like ``PlotWidget::palette``."""
    return _PALETTE[i % len(_PALETTE)]


@dataclass
class Series:
    """One curve: x/y data plus how to draw it."""

    name: str = ""
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    color: str = ""  # blank = "take slot 0 of whatever theme is active"
    style: str = "line"  # "line" | "markers" | "line_markers"
    dashed: bool = False


@dataclass
class Panel:
    """One subplot: curves, reference lines, or a block of monospace text."""

    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    logx: bool = False
    logy: bool = False
    legend: bool = True
    series: List[Series] = field(default_factory=list)
    vlines: List[Tuple[float, str]] = field(default_factory=list)
    hlines: List[Tuple[float, str]] = field(default_factory=list)
    text: List[str] = field(default_factory=list)
    #: Grid cell.  Negative means "unstated": :func:`render` then puts the
    #: panel in the next free cell, reading order.  Leaving these at 0 used
    #: to mean every unpositioned panel in a grid landed on the same cell
    #: and drew over the one before it.
    row: int = -1
    col: int = -1
    row_span: int = 1
    col_span: int = 1


@dataclass
class PanelGrid:
    """A set of panels laid out on a ``rows`` x ``cols`` grid."""

    panels: List[Panel] = field(default_factory=list)
    rows: int = 1
    cols: int = 1


def _placements(grid: "PanelGrid") -> List[Tuple["Panel", int, int]]:
    """Resolve every panel to a cell, filling unstated ones in reading order.

    Panels that name a cell keep it, and the cells they span are reserved
    first, so an explicitly placed panel and an automatic one never collide.
    """
    rows = max(grid.rows, 1)
    cols = max(grid.cols, 1)

    def span(r: int, c: int, p: "Panel"):
        for rr in range(r, min(r + max(p.row_span, 1), rows)):
            for cc in range(c, min(c + max(p.col_span, 1), cols)):
                yield rr, cc

    taken = set()
    for p in grid.panels:
        if p.row >= 0 and p.col >= 0:
            r = min(p.row, rows - 1)
            c = min(p.col, cols - 1)
            taken.update(span(r, c, p))

    out: List[Tuple["Panel", int, int]] = []
    cursor = 0
    for p in grid.panels:
        if p.row >= 0 and p.col >= 0:
            out.append((p, min(p.row, rows - 1), min(p.col, cols - 1)))
            continue
        while cursor < rows * cols and divmod(cursor, cols) in taken:
            cursor += 1
        r, c = divmod(min(cursor, rows * cols - 1), cols)
        taken.update(span(r, c, p))
        out.append((p, r, c))
        cursor += 1
    return out


#: One type scale for every figure, in points. Sizes are the only thing the
#: panels may vary, and they do not: a figure whose titles change size from
#: panel to panel reads as several figures sharing a canvas.
_TYPE = {
    "title": 9.5,
    "label": 8.5,
    "tick": 8.0,
    "legend": 7.5,
    "mono": 7.6,
}

#: Preferred sans faces, best first. Resolved once against what is actually
#: installed -- naming a font per artist makes matplotlib warn for every
#: missing one, on every draw, which buries real warnings in font noise.
_FONT_WANTED = ["Segoe UI", "Inter", "Helvetica Neue", "Helvetica", "Arial",
                "DejaVu Sans"]
_FONTS_READY = False


def _ensure_fonts() -> None:
    """Point matplotlib at the best installed face, once per process."""
    global _FONTS_READY
    if _FONTS_READY:
        return
    try:
        from matplotlib import font_manager

        have = {f.name for f in font_manager.fontManager.ttflist}
        stack = [name for name in _FONT_WANTED if name in have]
    except Exception:
        stack = []
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = stack + ["DejaVu Sans",
                                                      "sans-serif"]
    matplotlib.rcParams["mathtext.default"] = "regular"
    _FONTS_READY = True

#: Beyond this many points a marker per point is a blob, not a mark.
_MARKER_BUDGET = 48

#: Width of a text-only column, relative to a column that holds a plot.
_TEXT_COLUMN_WIDTH = 0.62


def _style_axes(ax, panel: "Panel") -> None:
    """Grid, spines and ticks: recessive, so the data is the only loud thing."""
    ax.set_facecolor(PANEL_BG)
    # Hairline, solid, one step off the surface, and under the data.
    ax.grid(True, color=GRIDC, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)

    # Two spines, not four. A box around a plot is ink that is not data, and
    # the top and right edges carry no scale.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FRAME)
        ax.spines[side].set_linewidth(0.9)

    ax.tick_params(which="major", colors=MUTED, labelsize=_TYPE["tick"],
                   direction="out", length=3.5, width=0.8, pad=3)
    ax.tick_params(which="minor", colors=GRIDC, direction="out",
                   length=2.0, width=0.6)
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_color(MUTED)

    # A log axis with no minor ticks hides where inside a decade a point
    # sits, which on a current axis is most of the information.
    if panel.logy:
        ax.yaxis.set_minor_locator(
            matplotlib.ticker.LogLocator(base=10.0, subs="auto", numticks=12)
        )
    if panel.logx:
        ax.xaxis.set_minor_locator(
            matplotlib.ticker.LogLocator(base=10.0, subs="auto", numticks=12)
        )
    if not panel.logy:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4),
                            useMathText=True)
        ax.yaxis.get_offset_text().set_color(MUTED)
        ax.yaxis.get_offset_text().set_fontsize(_TYPE["tick"] - 0.5)


def _title(ax, text: str) -> None:
    ax.set_title(text, color=FG, fontsize=_TYPE["title"], loc="left",
                 fontweight="600", pad=7)


def has_text_panels(grid: "PanelGrid") -> bool:
    """Whether ``grid`` carries any text-only panel."""
    return any(p.text and not p.series for p in grid.panels)


def without_text_panels(grid: "PanelGrid") -> "PanelGrid":
    """``grid`` with its text-only panels dropped and the rest re-flowed.

    The summary block belongs in a saved figure, where the image is the only
    thing that travels. On screen it is a third of the canvas spent
    restating numbers that are already in the metrics table, so the window
    drops it and the file keeps it.

    Positions are cleared rather than kept: leaving a panel at the cell it
    was given would hold open the gap the removed panel used to fill.
    """
    keep = [p for p in grid.panels if not (p.text and not p.series)]
    if len(keep) == len(grid.panels):
        return grid

    import copy as _copy

    panels = []
    for p in keep:
        q = _copy.copy(p)
        q.row = -1
        q.col = -1
        q.row_span = 1
        q.col_span = 1
        panels.append(q)
    n = len(panels)
    if n <= 1:
        rows, cols = 1, max(n, 1)
    elif n <= 3:
        rows, cols = 1, n
    else:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    return PanelGrid(panels=panels, rows=rows, cols=cols)


def render(fig, grid: PanelGrid, *, summary: bool = True) -> None:
    """Draw ``grid`` onto a matplotlib ``Figure`` (cleared first).

    ``summary=False`` drops text-only panels, which is what the on-screen
    canvas wants: those numbers are in the metrics table beside it.
    """
    if not summary:
        grid = without_text_panels(grid)

    _ensure_fonts()
    fig.clear()
    fig.patch.set_facecolor(BG)
    # Constrained layout measures the rendered text and then allocates, so a
    # long tick label or a two-line title cannot collide with the panel next
    # to it. Fixed hspace/wspace could only ever be right for one figure.
    try:
        fig.set_layout_engine("constrained")
        fig.get_layout_engine().set(w_pad=0.045, h_pad=0.045,
                                    wspace=0.06, hspace=0.08)
    except Exception:
        fig.subplots_adjust(hspace=0.42, wspace=0.30)

    rows = max(grid.rows, 1)
    cols = max(grid.cols, 1)
    placed = _placements(grid)

    # A column holding nothing but text does not need a plot's width. Giving
    # it one squeezes the panels that have data in them, which is the wrong
    # way round: the summary is a caption, not a third of the figure.
    text_cols = set(range(cols))
    for panel, _r, c in placed:
        if not (panel.text and not panel.series):
            for cc in range(c, min(c + max(panel.col_span, 1), cols)):
                text_cols.discard(cc)
    widths = [_TEXT_COLUMN_WIDTH if c in text_cols else 1.0
              for c in range(cols)]
    gs = fig.add_gridspec(rows, cols,
                          width_ratios=widths if cols > 1 else None)

    for p, r0, c0 in placed:
        r1 = min(r0 + max(p.row_span, 1), rows)
        c1 = min(c0 + max(p.col_span, 1), cols)
        ax = fig.add_subplot(gs[r0:r1, c0:c1])

        if p.text and not p.series:
            ax.set_facecolor(PANEL_BG)
            ax.set_xticks([])
            ax.set_yticks([])
            for side, spine in ax.spines.items():
                spine.set_visible(side in ("left",))
                spine.set_color(FRAME)
            if p.title:
                _title(ax, p.title)
            ax.text(
                0.03, 0.97, "\n".join(p.text),
                transform=ax.transAxes, va="top", ha="left",
                family="monospace", fontsize=_TYPE["mono"], color=FG,
                linespacing=1.45,
            )
            continue

        _style_axes(ax, p)
        if p.title:
            _title(ax, p.title)
        if p.xlabel:
            ax.set_xlabel(p.xlabel, color=MUTED, fontsize=_TYPE["label"],
                          labelpad=4)
        if p.ylabel:
            ax.set_ylabel(p.ylabel, color=MUTED, fontsize=_TYPE["label"],
                          labelpad=4)
        if p.logx:
            ax.set_xscale("log")
        if p.logy:
            ax.set_yscale("log")

        labels: List[str] = []
        for s in p.series:
            if not len(s.x) or not len(s.y):
                continue
            colour = s.color or palette(0)
            n = min(len(s.x), len(s.y))
            wants_markers = s.style in ("markers", "line_markers")
            linestyle = (
                "none" if s.style == "markers"
                else ("--" if s.dashed else "-")
            )
            label = s.name or None
            if label:
                labels.append(label)

            # A marker on every point of a 400-point sweep is a blob. Thin
            # them so they stay marks; the line still carries every point.
            every = None
            if wants_markers and linestyle != "none" and n > _MARKER_BUDGET:
                every = max(1, n // _MARKER_BUDGET)

            ax.plot(
                s.x[:n], s.y[:n],
                color=colour,
                marker="o" if wants_markers else None,
                markersize=4.6,
                markevery=every,
                # A ring in the surface colour keeps a marker legible where
                # it crosses another line or another marker.
                markeredgecolor=PANEL_BG,
                markeredgewidth=0.9,
                linewidth=1.9,
                linestyle=linestyle,
                solid_capstyle="round",
                solid_joinstyle="round",
                label=label,
                zorder=3,
            )

        # A log axis stretched over fifteen decades shows one: a current at
        # the noise floor is not data, and letting it set the bottom of the
        # scale squeezes the decades the measurement is actually in into a
        # few pixels. Eight decades is more than any of these instruments
        # resolves in one sweep.
        if p.logy:
            positive = [abs(y) for s in p.series for y in s.y
                        if y == y and y != 0 and abs(y) != float("inf")]
            positive = [y for y in positive if y > 0]
            if positive:
                top = max(positive)
                floor = max(min(positive), top / 10.0 ** _LOG_DECADES)
                if top > floor:
                    ax.set_ylim(floor / 2.0, top * 2.0)

        for value, color in p.vlines:
            ax.axvline(value, color=color, linewidth=1.0, linestyle="--",
                       alpha=0.75, zorder=1)
        for value, color in p.hlines:
            ax.axhline(value, color=color, linewidth=1.0, linestyle="--",
                       alpha=0.75, zorder=1)

        # A legend box for a single curve is a box saying what the title
        # should have said, so fold the name into the title instead -- but
        # only while the result still fits. A title long enough to run into
        # the panel beside it is worse than the box it replaced, so past
        # that length the box stays.
        folded = False
        if len(labels) == 1 and p.legend:
            only = labels[0]
            merged = f"{p.title}  -  {only}" if p.title else only
            if only.lower() in (p.title or "").lower():
                folded = True
            elif len(merged) <= _TITLE_FOLD_CHARS:
                _title(ax, merged)
                folded = True
        if p.legend and labels and not folded:
            leg = ax.legend(
                fontsize=_TYPE["legend"],
                facecolor=PANEL_BG,
                edgecolor=FRAME,
                labelcolor=FG,
                framealpha=0.92,
                borderpad=0.5,
                handlelength=1.6,
                handletextpad=0.6,
                labelspacing=0.35,
                borderaxespad=0.5,
            )
            if leg:
                leg.get_frame().set_linewidth(0.6)


def save(grid: PanelGrid, path: str, size=(12.0, 7.5), dpi: int = 200) -> str:
    """Render ``grid`` to an image file and return the path.

    Saved at print resolution and *with* its summary panel: a file leaves the
    program, and the numbers have to leave with it.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    fig = Figure(figsize=size, dpi=dpi)
    render(fig, grid, summary=True)
    fig.savefig(path, facecolor=BG, dpi=dpi)
    return path
