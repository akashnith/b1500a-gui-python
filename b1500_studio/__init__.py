"""B1500 Studio -- a measurement suite for the Keysight B1500A.

Three instruments, one program.  The **SMU** does DC: staircase I-V, forming,
retention, endurance -- everything that needs real current compliance and an
integrating ADC.  The **B1530A WGFMU** does fast pulses: a waveform is built
on a 10 ns grid, handed over, and played.  The **B1520A MFCMU** measures
capacitance.  Which instrument a measurement belongs to is a physical fact,
not a preference: below a few microseconds a pulse has to come from the
WGFMU, and at or above that the SMU is the better instrument.

Layout::

    core/          instrument-independent: waveforms, limits, analysis, figures
    instruments/   the drivers, and the simulator that stands in for them
    experiments/   WGFMU pulse protocols -- synapse, neuron, ferroelectric
    protocols/     SMU DC protocols -- I-V, forming, threshold, endurance
    gui/           PySide6 application
    cli.py         command line entry point

Only :mod:`b1500_studio.instruments` opens a session, and even there VISA and
the DLL are imported on first use.  Everything else is unit-testable with no
instrument, no VISA and no Qt installed, which is why the whole protocol
catalogue can be exercised against the simulator before it is pointed at a
device.
"""

from __future__ import annotations

__version__ = "1.1.0"

from .core.driver import (
    Channel,
    Driver,
    DriverError,
    Hardware,
    Results,
    irange_amps,
)
from .core.plan import (
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

__all__ = [
    "__version__",
    # plan
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
    # driver
    "Driver",
    "DriverError",
    "Hardware",
    "Results",
    "Channel",
    "irange_amps",
    # factories
    "make_driver",
    "make_smu",
]


def make_driver(kind: str = "demo", dll_path=None, **kw) -> Driver:
    """A WGFMU driver: ``"demo"``, ``"wgfmu"`` (subprocess) or ``"wgfmu-inproc"``.

    ``"wgfmu"`` is the recommended hardware driver: it runs the DLL in a child
    process so a VISA hang is reported instead of freezing the caller.
    """
    k = kind.strip().lower()
    if k in ("demo", "sim", "simulate"):
        from .instruments.demo import DemoDriver

        return DemoDriver(**kw)
    if k in ("wgfmu", "remote", "hardware"):
        from .instruments.session import RemoteWgfmu

        return RemoteWgfmu(dll_path, **kw)
    if k in ("wgfmu-inproc", "inproc"):
        from .instruments.wgfmu_dll import Wgfmu

        return Wgfmu(dll_path, **kw)
    raise ValueError(f"unknown driver {kind!r}")


def make_smu(kind: str = "demo", resource: str = "", **kw):
    """An SMU: ``"demo"`` for the simulator, ``"smu"`` for the instrument.

    Both answer the same calls, so a protocol never learns which it got --
    which is what lets the whole DC catalogue be tested with nothing
    connected.
    """
    k = kind.strip().lower()
    if k in ("demo", "sim", "simulate"):
        from .instruments.demo import DemoSmu

        return DemoSmu(**kw)
    if k in ("smu", "visa", "hardware"):
        from .instruments.smu import Smu

        return Smu(resource, **kw) if resource else Smu(**kw)
    raise ValueError(f"unknown SMU {kind!r}")
