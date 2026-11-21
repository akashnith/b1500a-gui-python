"""Instrument-independent core: waveforms, limits, analysis, figures.

Nothing in here opens a session or touches a driver, so every module is
importable and unit-testable on a machine with no instrument, no VISA and no
Qt.  That is the whole point of the split: the parts that decide *what* to
measure are separated from the parts that know *how* to talk to the
hardware, which live in :mod:`b1500_studio.instruments`.

    plan          waveform model on the 10 ns grid, and SI parsing
    driver        Hardware / Results / Driver interface
    runner        executes one experiment and collects its results
    verification  dry-run safety checks, no instrument needed
    plotting      declarative panels rendered with matplotlib
    ferro         ferroelectric analysis (polarization, PUND, leakage)
"""

from __future__ import annotations

from .driver import Channel, Driver, DriverError, Hardware, Results, irange_amps
from .plan import (
    GRID,
    Event,
    Plan,
    PlanError,
    Vec,
    Wave,
    parse_list,
    parse_si,
    secs,
    ticks,
)
from .plotting import Panel, PanelGrid, Series, palette, render, set_theme

__all__ = [
    "GRID",
    "Plan",
    "PlanError",
    "Vec",
    "Event",
    "Wave",
    "parse_si",
    "parse_list",
    "ticks",
    "secs",
    "Driver",
    "DriverError",
    "Hardware",
    "Results",
    "Channel",
    "irange_amps",
    "Panel",
    "PanelGrid",
    "Series",
    "palette",
    "render",
    "set_theme",
]
