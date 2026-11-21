"""Everything that talks to an instrument -- and the simulator that does not.

    smu           B1500A SMU over VISA, speaking FLEX (DC and staircase)
    cmu           B1520A MFCMU: capacitance, C-V and Rayleigh sweeps
    wgfmu_dll     ctypes binding to Keysight's wgfmu.dll
    session       the same WGFMU driver in a killable subprocess
    diagnostics   why a connection failed, in the order the evidence says
    demo          simulated SMU, MFCMU and WGFMU -- no instrument needed

Importing this package does **not** load VISA or the DLL: each module pulls
its dependency in only when a session is actually opened, so the CLI, the
tests and the GUI all start on a machine with neither installed.

:mod:`demo` belongs here rather than beside the analysis because it stands in
for a driver.  It implements the same surface as the real ones, which is what
lets every protocol be exercised end to end with nothing connected.
"""

from __future__ import annotations

__all__ = ["open_smu", "open_wgfmu"]


def open_smu(resource: str = "", *, demo: bool = False, log=None, **kw):
    """A connected SMU: the simulator when ``demo``, the instrument otherwise.

    Both answer the same calls, so a protocol never learns which it got.
    """
    if demo:
        from .demo import DemoSmu

        device = DemoSmu(log=log, **kw)
    else:
        from .smu import Smu

        device = Smu(resource, log=log, **kw)
    device.open(resource or None)
    return device


def open_wgfmu(dll_path=None, *, demo: bool = False, **kw):
    """A WGFMU driver: the simulator when ``demo``, else the subprocess one.

    The subprocess driver is the right default for hardware -- it runs the
    DLL in a child process, so a VISA hang is reported rather than freezing
    whatever called it.
    """
    if demo:
        from .demo import DemoDriver

        return DemoDriver(**kw)
    from .session import RemoteWgfmu

    return RemoteWgfmu(dll_path, **kw)
