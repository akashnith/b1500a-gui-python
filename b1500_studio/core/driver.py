"""Instrument abstraction: hardware settings, results and the driver interface.

Faithful port of ``pkg/src/driver.h``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .plan import Plan

__all__ = [
    "Hardware",
    "irange_amps",
    "Channel",
    "Results",
    "Driver",
    "DriverError",
    "LogFn",
]

LogFn = Optional[Callable[[str], None]]


class DriverError(RuntimeError):
    """Any instrument-level failure."""


@dataclass
class Hardware:
    """Channel assignment, ranges and safety limits."""

    # WGFMU channel IDs follow the slot holding the B1530A: slot 2 -> 201/202.
    # Check with `smu modules` or WGFMU_getChannelIds if a run reports that
    # the channel does not exist.
    force_ch: int = 201
    sink_ch: int = 202
    force_meas: str = "V"  # 'V' | 'I' | '0' (off)
    sink_meas: str = "I"  # 'V' | 'I' | '0' (off)
    irange: int = 4  # 0..4 = 1uA, 10uA, 100uA, 1mA, 10mA
    vmax: float = 1.0  # safety limit on |V|
    edge: float = 10e-9  # rise = fall time
    settle: float = 0.3  # fraction of a flat top skipped by measure windows
    sink_bias: float = 0.0
    max_points: int = 200_000


_IRANGE_AMPS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def irange_amps(i: int) -> float:
    """Current-range index 0..4 -> full-scale amps (clamped, like the C++)."""
    return _IRANGE_AMPS[0 if i < 0 else (4 if i > 4 else i)]


@dataclass
class Channel:
    """Time/value arrays returned by one measured channel."""

    t: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)


@dataclass
class Results:
    """Per-channel measured data, keyed by channel id."""

    ch: Dict[int, Channel] = field(default_factory=dict)

    def current(self, hw: Hardware) -> List[float]:
        """The channel that is measuring current, sink first then force."""
        if hw.sink_meas == "I" and hw.sink_ch in self.ch:
            return self.ch[hw.sink_ch].y
        if hw.force_meas == "I" and hw.force_ch in self.ch:
            return self.ch[hw.force_ch].y
        raise DriverError(
            "no channel is set to measure current (Instrument tab)"
        )


class Driver(abc.ABC):
    """Instrument interface. Implemented by the WGFMU and demo drivers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short driver name, e.g. ``"WGFMU"`` or ``"DEMO"``."""

    @abc.abstractmethod
    def open(self, address: str) -> None:
        """Open a session to ``address`` (a VISA resource string)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close the session; must be safe to call when not open."""

    @abc.abstractmethod
    def channels(self) -> List[int]:
        """Channel ids reported by the instrument."""

    @abc.abstractmethod
    def run(self, plan: Plan, hw: Hardware, log: LogFn = None) -> Results:
        """Execute ``plan`` and return the measured data."""

    @abc.abstractmethod
    def abort(self) -> None:
        """Abort a run in progress."""

    # Context-manager sugar so callers cannot leak a session.
    def __enter__(self) -> "Driver":
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.close()
        except Exception:
            pass
