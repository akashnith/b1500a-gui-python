"""Keysight B1530A WGFMU instrument library, loaded at run time via ctypes.

Faithful port of ``pkg/src/driver_wgfmu.cpp``.

Design note -- why there is no pyvisa here
------------------------------------------
``wgfmu.dll`` opens its **own** VISA session internally inside
``WGFMU_openSession()``.  It does not need, and must not be gated behind,
``pyvisa.ResourceManager()``.  The older single-file GUI refused to call into
the DLL unless an out-of-process ``viOpenDefaultRM()`` probe succeeded; that
probe is unrelated to this code path, so it only ever produced false
negatives.  This module does exactly what the working C++ driver does:
``LoadLibrary`` then ``WGFMU_openSession``.

If the Keysight IO Libraries services are not running, ``WGFMU_openSession``
blocks inside the DLL with no timeout and no way to abort.  That hang cannot
be fixed from Python -- see :mod:`b1500_studio.session` for the subprocess
wrapper that contains it, and the project README for the service fix.
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import List, Optional

from ..core.driver import Channel, Driver, DriverError, Hardware, LogFn, Results
from ..core.plan import Plan, secs

__all__ = [
    "Wgfmu",
    "WgfmuError",
    "normalize_address",
    "DEFAULT_ADDRESS",
    "DEFAULT_DLL",
]

DEFAULT_ADDRESS = "GPIB0::16::INSTR"
DEFAULT_DLL = "wgfmu.dll"  # resolved from System32 by the normal DLL search


class WgfmuError(DriverError):
    """A WGFMU_* call returned a negative status, or the DLL is unusable."""


# ------------------------------------------------------------- wgfmu.h enums

WGFMU_OPERATION_MODE_FASTIV = 2001
WGFMU_OPERATION_MODE_SMU = 2003
WGFMU_FORCE_VOLTAGE_RANGE_AUTO = 3000
WGFMU_FORCE_VOLTAGE_RANGE_3V = 3001
WGFMU_FORCE_VOLTAGE_RANGE_5V = 3002
WGFMU_MEASURE_MODE_VOLTAGE = 4000
WGFMU_MEASURE_MODE_CURRENT = 4001
WGFMU_MEASURE_VOLTAGE_RANGE_5V = 5001
WGFMU_MEASURE_CURRENT_RANGE_1UA = 6001
WGFMU_MEASURE_CURRENT_RANGE_10UA = 6002
WGFMU_MEASURE_CURRENT_RANGE_100UA = 6003
WGFMU_MEASURE_CURRENT_RANGE_1MA = 6004
WGFMU_MEASURE_CURRENT_RANGE_10MA = 6005
WGFMU_MEASURE_EVENT_DATA_AVERAGED = 12000

_IRANGE_ENUM = (
    WGFMU_MEASURE_CURRENT_RANGE_1UA,
    WGFMU_MEASURE_CURRENT_RANGE_10UA,
    WGFMU_MEASURE_CURRENT_RANGE_100UA,
    WGFMU_MEASURE_CURRENT_RANGE_1MA,
    WGFMU_MEASURE_CURRENT_RANGE_10MA,
)

_c_int_p = ctypes.POINTER(ctypes.c_int)
_c_dbl_p = ctypes.POINTER(ctypes.c_double)
_STR = ctypes.c_char_p
_I = ctypes.c_int
_D = ctypes.c_double

#: name -> (argtypes, required)
_PROTOTYPES = {
    "openSession": ([_STR], True),
    "closeSession": ([], True),
    "initialize": ([], True),
    "clear": ([], True),
    "connect": ([_I], True),
    "disconnect": ([_I], True),
    "setOperationMode": ([_I, _I], True),
    "setForceVoltageRange": ([_I, _I], True),
    "setMeasureMode": ([_I, _I], True),
    "setMeasureVoltageRange": ([_I, _I], True),
    "setMeasureCurrentRange": ([_I, _I], True),
    "createPattern": ([_STR, _D], True),
    "addVector": ([_STR, _D, _D], True),
    "setMeasureEvent": ([_STR, _STR, _D, _I, _D, _D, _I], True),
    "addSequence": ([_I, _STR, _D], True),
    "execute": ([], True),
    "abort": ([], True),
    "waitUntilCompleted": ([], True),
    "getMeasureValueSize": ([_I, _c_int_p, _c_int_p], True),
    "getMeasureValue": ([_I, _I, _c_dbl_p, _c_dbl_p], True),
    "getChannelIdSize": ([_c_int_p], True),
    "getChannelIds": ([_c_int_p, _c_int_p], True),
    "getErrorSize": ([_c_int_p], True),
    "getError": ([ctypes.c_char_p, _c_int_p], True),
    "setTimeout": ([_D], False),  # optional
}


def _b(s: str) -> bytes:
    return s.encode("ascii")


def normalize_address(resource: str) -> str:
    """VISA resource -> the address form ``wgfmu.dll`` accepts.

    The WGFMU library wants ``GPIB0::16``. Handing it the full VISA resource
    ``GPIB0::16::INSTR`` makes ``WGFMU_openSession`` fail with
    ``rc=-6, +100 "Undefined GPIB command"`` -- confirmed on the instrument.
    """
    address = (resource or "").strip()
    while address.upper().endswith("::INSTR"):
        address = address[: -len("::INSTR")]
    return address


class Wgfmu(Driver):
    """Run-time loaded ``wgfmu.dll`` driver."""

    def __init__(self, dll_path: Optional[str] = None) -> None:
        if sys.platform == "win32" and ctypes.sizeof(ctypes.c_void_p) != 8:
            raise WgfmuError(
                "use 64-bit Python: the installed wgfmu.dll is x64"
            )
        path = dll_path or DEFAULT_DLL
        # Dependent DLLs (including the VISA library) sit beside an explicitly
        # given wgfmu.dll; mirror the C++ LoadLibrary search behaviour.
        if sys.platform == "win32" and os.path.dirname(path):
            folder = os.path.dirname(os.path.abspath(path))
            if os.path.isdir(folder):
                try:
                    os.add_dll_directory(folder)
                except OSError:
                    pass
        loader = ctypes.WinDLL if sys.platform == "win32" else ctypes.CDLL
        try:
            self._lib = loader(path)
        except OSError as exc:
            raise WgfmuError(
                f"cannot load {path}: {exc}\n"
                "(bitness mismatch, or wgfmu.dll / its VISA dependency not found)"
            ) from exc

        self._f = {}
        missing = []
        for name, (argtypes, required) in _PROTOTYPES.items():
            try:
                fn = getattr(self._lib, "WGFMU_" + name)
            except AttributeError:
                self._f[name] = None
                if required:
                    missing.append("WGFMU_" + name)
                continue
            fn.argtypes = argtypes
            fn.restype = ctypes.c_int
            self._f[name] = fn
        if missing:
            raise WgfmuError("wgfmu.dll lacks: " + " ".join(missing))

        self._path = path
        self._open = False

    # ------------------------------------------------------------- plumbing

    @property
    def name(self) -> str:
        return "WGFMU"

    @property
    def dll_path(self) -> str:
        return self._path

    def _last_error(self) -> str:
        n = ctypes.c_int(0)
        if self._f["getErrorSize"](ctypes.byref(n)) < 0 or n.value <= 0:
            return ""
        buf = ctypes.create_string_buffer(n.value + 2)
        size = ctypes.c_int(n.value + 1)
        if self._f["getError"](buf, ctypes.byref(size)) < 0:
            return ""
        return buf.value.decode(errors="replace").strip()

    def _chk(self, rc: int, what: str) -> None:
        if rc >= 0:
            return
        detail = self._last_error()
        raise WgfmuError(
            f"{what} failed ({rc})" + (f": {detail}" if detail else "")
        )

    # -------------------------------------------------------------- session

    def open(self, address: str = DEFAULT_ADDRESS) -> None:
        """Open a session, trying the library's address form first.

        A VISA resource string is accepted and trimmed, so callers can pass
        whatever the rest of the program uses.
        """
        candidates = [normalize_address(address)]
        if address.strip() != candidates[0]:
            candidates.append(address.strip())

        last = None
        for candidate in candidates:
            try:
                self._chk(
                    self._f["openSession"](_b(candidate)),
                    f"openSession({candidate})",
                )
            except WgfmuError as exc:
                last = exc
                continue
            self._open = True
            self._chk(self._f["initialize"](), "initialize")
            return
        raise last if last else WgfmuError("openSession failed")

    def close(self) -> None:
        if not self._open:
            return
        self._f["initialize"]()
        self._f["closeSession"]()
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def channels(self) -> List[int]:
        n = ctypes.c_int(0)
        self._chk(
            self._f["getChannelIdSize"](ctypes.byref(n)), "getChannelIdSize"
        )
        ids = (ctypes.c_int * max(n.value, 1))()
        size = ctypes.c_int(n.value)
        self._chk(
            self._f["getChannelIds"](ids, ctypes.byref(size)), "getChannelIds"
        )
        return list(ids[: n.value])

    def abort(self) -> None:
        if self._open:
            self._f["abort"]()

    def route_to_smu(self, address: str, chans) -> None:
        """Switch the RSU outputs to the SMU path (operation mode SMU)."""
        self._chk(
            self._f["openSession"](_b(address)), f"openSession({address})"
        )
        try:
            for ch in chans:
                self._chk(
                    self._f["setOperationMode"](ch, WGFMU_OPERATION_MODE_SMU),
                    "setOperationMode(SMU)",
                )
                self._chk(self._f["connect"](ch), "connect")
        finally:
            self._f["closeSession"]()

    # ------------------------------------------------------------------ run

    def run(self, plan: Plan, hw: Hardware, log: LogFn = None) -> Results:
        """Execute ``plan`` on ``hw.force_ch``, sinking at ``hw.sink_bias``."""
        if not self._open:
            raise WgfmuError("WGFMU session not open")
        f = self._f
        plan.validate(hw.vmax, hw.max_points)

        mv = plan.max_abs_v()
        force_range = (
            WGFMU_FORCE_VOLTAGE_RANGE_3V
            if mv <= 3.0
            else WGFMU_FORCE_VOLTAGE_RANGE_5V
            if mv <= 5.0
            else WGFMU_FORCE_VOLTAGE_RANGE_AUTO
        )
        irange = _IRANGE_ENUM[max(0, min(4, hw.irange))]

        chs = (hw.force_ch, hw.sink_ch)
        modes = (hw.force_meas, hw.sink_meas)
        pattern_names = ("force", "sink")

        out = Results()
        try:
            self._chk(f["clear"](), "clear")

            for ch, mode in zip(chs, modes):
                self._chk(
                    f["setOperationMode"](ch, WGFMU_OPERATION_MODE_FASTIV),
                    "setOperationMode",
                )
                self._chk(
                    f["setForceVoltageRange"](ch, force_range),
                    "setForceVoltageRange",
                )
                if mode == "V":
                    self._chk(
                        f["setMeasureMode"](ch, WGFMU_MEASURE_MODE_VOLTAGE),
                        "setMeasureMode(V)",
                    )
                    self._chk(
                        f["setMeasureVoltageRange"](
                            ch, WGFMU_MEASURE_VOLTAGE_RANGE_5V
                        ),
                        "setMeasureVoltageRange",
                    )
                elif mode == "I":
                    self._chk(
                        f["setMeasureMode"](ch, WGFMU_MEASURE_MODE_CURRENT),
                        "setMeasureMode(I)",
                    )
                    self._chk(
                        f["setMeasureCurrentRange"](ch, irange),
                        "setMeasureCurrentRange",
                    )
                self._chk(f["connect"](ch), "connect")

            # Force pattern: the plan itself.
            pf, ps = _b("force"), _b("sink")
            self._chk(f["createPattern"](pf, 0.0), "createPattern(force)")
            for v in plan.vec:
                self._chk(
                    f["addVector"](pf, secs(v.n), v.v), "addVector(force)"
                )
            # Sink pattern: one flat vector at the sink bias.
            self._chk(
                f["createPattern"](ps, hw.sink_bias), "createPattern(sink)"
            )
            self._chk(
                f["addVector"](ps, secs(plan.total()), hw.sink_bias),
                "addVector(sink)",
            )

            for i, (ch, mode) in enumerate(zip(chs, modes)):
                pn = pf if i == 0 else ps
                if mode != "0":
                    for e in plan.ev:
                        self._chk(
                            f["setMeasureEvent"](
                                pn,
                                _b(e.name),
                                secs(e.t),
                                int(e.points),
                                secs(e.interval),
                                secs(e.avg),
                                WGFMU_MEASURE_EVENT_DATA_AVERAGED,
                            ),
                            f"setMeasureEvent({e.name})",
                        )
                self._chk(f["addSequence"](ch, pn, 1.0), "addSequence")

            if f["setTimeout"] is not None:
                f["setTimeout"](plan.duration() * 2.0 + 60.0)

            self._chk(f["execute"](), "execute")
            self._chk(f["waitUntilCompleted"](), "waitUntilCompleted")

            for ch, mode in zip(chs, modes):
                if mode == "0":
                    continue
                meas, tot = ctypes.c_int(0), ctypes.c_int(0)
                self._chk(
                    f["getMeasureValueSize"](
                        ch, ctypes.byref(meas), ctypes.byref(tot)
                    ),
                    "getMeasureValueSize",
                )
                n = meas.value
                ts = (ctypes.c_double * max(n, 1))()
                ys = (ctypes.c_double * max(n, 1))()
                tv, yv = ctypes.c_double(0.0), ctypes.c_double(0.0)
                for k in range(n):
                    self._chk(
                        f["getMeasureValue"](
                            ch, k, ctypes.byref(tv), ctypes.byref(yv)
                        ),
                        "getMeasureValue",
                    )
                    ts[k], ys[k] = tv.value, yv.value
                if n != tot.value and log:
                    log(
                        f"ch{ch}: {n}/{tot.value} points returned (aborted?)"
                    )
                out.ch[ch] = Channel(t=list(ts[:n]), y=list(ys[:n]))
        finally:
            for ch in chs:
                try:
                    f["disconnect"](ch)
                except Exception:
                    pass
        return out
