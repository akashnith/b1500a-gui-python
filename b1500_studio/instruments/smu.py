"""Keysight B1500A SMU driver (pyvisa + FLEX command set).

The WGFMU path (:mod:`b1500_studio.wgfmu_dll`) drives nanosecond pulses
through ``wgfmu.dll``. This module is the other half: the ordinary SMUs,
reached over GPIB with FLEX commands, for DC spot readings, staircase I-V
sweeps and pulsed measurements.

Hang protection
---------------
``pyvisa.ResourceManager()`` calls ``viOpenDefaultRM()``, which blocks forever
when the Windows VISA stack is broken (typically the Keysight IO Libraries
services being stopped). Unlike the WGFMU path, that failure *is* predictable
from outside: :func:`probe_visa` runs the same call in a short-lived
subprocess and caches the answer, so :meth:`Smu.open` refuses with a clear
message instead of freezing the caller.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..core.driver import DriverError

__all__ = [
    "Smu",
    "SmuError",
    "VisaUnavailable",
    "SweepResult",
    "Module",
    "I_RANGES",
    "SMU_MODELS",
    "WGFMU_MODELS",
    "CMU_MODELS",
    "probe_visa",
    "DEFAULT_RESOURCE",
    "TIMEOUT_MS",
]

DEFAULT_RESOURCE = "GPIB0::16::INSTR"
TIMEOUT_MS = 60_000

SMU_MODELS = {
    "B1510A": "HPSMU",
    "B1511A": "MPSMU",
    "B1511B": "MPSMU",
    "B1512A": "MCSMU",
    "B1513A": "HVSMU",
    "B1514A": "MCSMU",
    "B1517A": "HRSMU",
}
WGFMU_MODELS = ("B1530A",)
CMU_MODELS = ("B1520A",)  # MFCMU

#: Current-measurement ranges: 0 auto, +n limited auto, -n fixed.
I_RANGES: Dict[str, int] = {
    "Auto": 0,
    "Auto, min 1 nA": 11,
    "Auto, min 1 uA": 14,
    "Auto, min 100 uA": 16,
    "Fixed 1 uA": -14,
    "Fixed 10 uA": -15,
    "Fixed 100 uA": -16,
    "Fixed 1 mA": -17,
}

#: One datum in an ``FMT 1`` reply: status, channel, type, value.
DATA_RE = re.compile(r"([A-Z])([A-Z])([A-Z])\s*([+-]?\d+\.?\d*E[+-]\d+)")
VAL_RE = re.compile(r"[+-]?\d+\.?\d*E[+-]\d+")

#: Status letters that mean the reading hit a limit.
COMPLIANCE_FLAGS = ("C", "T")


class SmuError(DriverError):
    """An SMU/FLEX command failed, or a reply could not be parsed."""


class VisaUnavailable(SmuError):
    """The VISA layer never answered; see :func:`probe_visa`."""


# --------------------------------------------------------------- VISA probe

_PROBE_CODE = (
    "import pyvisa, sys\n"
    "rm = pyvisa.ResourceManager()\n"
    "sys.stdout.write(str(getattr(rm.visalib, 'library_path', 'ok')))\n"
)

_VISA_OK: Optional[bool] = None
_VISA_DETAIL = ""

VISA_HELP = """\
VISA did not answer viOpenDefaultRM().

This is a Windows VISA installation problem, not a problem with this program
or with the instrument. The usual cause is that the Keysight IO Libraries
services are stopped, so VISA never replies and the call blocks forever.

Fix, in an Administrator PowerShell:

  Set-Service KeysightDistributedInfrastructureService -StartupType Automatic
  Set-Service AgilentIOLibrariesService               -StartupType Automatic
  Start-Service KeysightDistributedInfrastructureService
  Start-Service AgilentIOLibrariesService

If NI-VISA was installed after the Keysight IO Libraries Suite, make Keysight
the primary VISA again and reboot:

  & "C:\\Program Files\\Keysight\\IO Libraries Suite\\primary-visa-utility.exe"

Run `python -m b1500_studio check` for a full diagnosis.
"""


def probe_visa(timeout_s: float = 15.0, force: bool = False) -> bool:
    """Is ``pyvisa.ResourceManager()`` usable? Answered out of process.

    A broken VISA dispatcher parks the calling thread inside the DLL forever
    and pyvisa offers no timeout, so the only safe way to ask is in a child
    process that can be killed. The answer is cached.
    """
    global _VISA_OK, _VISA_DETAIL
    if _VISA_OK is not None and not force:
        return _VISA_OK
    try:
        p = subprocess.run(
            [sys.executable, "-I", "-c", _PROBE_CODE],
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        _VISA_OK, _VISA_DETAIL = False, f"no answer within {timeout_s:.0f} s"
        return False
    except Exception:
        _VISA_OK, _VISA_DETAIL = True, ""  # cannot probe; do not block the user
        return True
    if p.returncode != 0:
        _VISA_OK = False
        _VISA_DETAIL = (p.stderr or b"").decode("utf-8", "replace").strip()
        return False
    _VISA_OK, _VISA_DETAIL = True, ""
    return True


def _require_visa(timeout_s: float = 15.0) -> None:
    if not probe_visa(timeout_s):
        head = f"VISA is not usable: {_VISA_DETAIL}" if _VISA_DETAIL else "VISA is not usable."
        raise VisaUnavailable(head + "\n\n" + VISA_HELP)


def _open_hint(resource: str, exc: Exception) -> str:
    """Turn a raw pyvisa open failure into something actionable."""
    text = str(exc)
    if "VI_ERROR_RSRC_NFOUND" in text or "not found" in text.lower():
        if resource.upper().startswith("GPIB"):
            return (
                f"{text}\n\nVISA is alive, but there is no {resource}.\n"
                "  - Is the USB-GPIB controller (82357A/B) plugged in with its "
                "green READY LED on, and listed in Device Manager under "
                "'GPIB Interfaces'?\n"
                "  - Does the board number (GPIB0 vs GPIB1) match Keysight "
                "Connection Expert?\n"
                "  - Is the B1500A powered on and booted, with the cable seated?"
            )
        return f"{text}\n\nVISA is alive but cannot find {resource}."
    if "VI_ERROR_RSRC_LOCKED" in text:
        return (
            f"{text}\n\nThe resource is locked by another program. Close "
            "Connection Expert, EasyEXPERT, Interactive IO or another Python "
            "session holding the instrument."
        )
    if "timeout" in text.lower():
        return f"{text}\n\nThe instrument did not answer. Confirm the GPIB address."
    return text


# ------------------------------------------------------------- data objects


@dataclass
class Module:
    """One installed module reported by ``UNT? 0``."""

    slot: int
    model: str
    kind: str  # "SMU" | "WGFMU" | "MFCMU"
    family: str = ""
    channels: List[int] = field(default_factory=list)


@dataclass
class SweepResult:
    """One staircase sweep."""

    t: List[float] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    i: List[float] = field(default_factory=list)
    status: List[str] = field(default_factory=list)
    label: str = ""

    def __len__(self) -> int:
        return len(self.i)

    @property
    def compliance_hits(self) -> int:
        """How many points hit the current compliance."""
        return sum(1 for s in self.status if s in COMPLIANCE_FLAGS)

    def resistance(self) -> List[float]:
        """Pointwise V/I, NaN where I is ~zero."""
        out = []
        for v, i in zip(self.v, self.i):
            out.append(v / i if abs(i) > 1e-15 else float("nan"))
        return out


# ---------------------------------------------------------------- the driver


class Smu:
    """B1500A SMU over GPIB, speaking FLEX."""

    def __init__(
        self,
        resource: str = DEFAULT_RESOURCE,
        timeout_ms: int = TIMEOUT_MS,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.resource = resource
        self.timeout_ms = timeout_ms
        self._log = log
        self.rm = None
        self.inst = None

    # ------------------------------------------------------------ plumbing

    @property
    def name(self) -> str:
        return "SMU"

    @property
    def connected(self) -> bool:
        return self.inst is not None

    def log(self, message: str) -> None:
        if self._log:
            self._log(message)

    def open(self, resource: Optional[str] = None) -> None:
        """Open the VISA session, refusing fast if VISA is unusable."""
        import pyvisa

        _require_visa()
        self.resource = resource or self.resource
        self.rm = pyvisa.ResourceManager()
        try:
            self.inst = self.rm.open_resource(self.resource)
        except Exception as exc:
            try:
                self.rm.close()
            except Exception:
                pass
            self.rm = None
            raise SmuError(_open_hint(self.resource, exc)) from exc
        self.inst.timeout = self.timeout_ms
        self.inst.read_termination = "\n"
        self.inst.write_termination = "\n"
        self.log(f"connected to {self.resource}")

    def close(self) -> None:
        """Zero the outputs, release the channels and close the session."""
        if self.inst is not None:
            # CL only. *CLS is NOT supported by this mainframe -- it answers
            # +100 "Undefined GPIB command." -- so sending it did nothing
            # except leave an error sitting in the queue for whatever
            # connected next to trip over. The exception handler hid the
            # failure without preventing the side effect.
            for cmd in ("CL",):
                try:
                    self.inst.write(cmd)
                    time.sleep(0.05)
                except Exception:
                    pass
            try:
                self.inst.close()
            except Exception:
                pass
        if self.rm is not None:
            try:
                self.rm.close()
            except Exception:
                pass
        self.inst = self.rm = None

    def __enter__(self) -> "Smu":
        if not self.connected:
            self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write(self, cmd: str) -> None:
        if self.inst is None:
            raise SmuError("not connected")
        self.log(f">> {cmd}")
        self.inst.write(cmd)

    def read_line(self) -> str:
        r = self.inst.read().strip()
        self.log(f"<< {r}")
        return r

    def query(self, cmd: str) -> str:
        if self.inst is None:
            raise SmuError("not connected")
        self.log(f">> {cmd}")
        r = self.inst.query(cmd).strip()
        self.log(f"<< {r}")
        return r

    def errors(self) -> List[str]:
        """Drain ``ERR?`` until it reports 0."""
        out: List[str] = []
        for _ in range(10):
            e = self.query("ERR?")
            if e.split(",")[0].strip() == "0":
                break
            out.append(e)
        return out

    def checked(self, cmd: str, fatal: bool = True) -> None:
        """Send ``cmd`` and raise if the instrument flags an error."""
        self.write(cmd)
        errs = self.errors()
        if errs:
            msg = f"'{cmd}' -> ERR? {' | '.join(errs)}"
            if fatal:
                raise SmuError(msg)
            self.log(f"!! (non-fatal) {msg}")

    # -------------------------------------------------------- identification

    def idn(self) -> str:
        return self.query("*IDN?")

    def scan(self) -> List[Module]:
        """``UNT? 0`` -> the installed modules."""
        mods: List[Module] = []
        for i, entry in enumerate(self.query("UNT? 0").split(";")):
            model = entry.split(",")[0].strip()
            slot = i + 1
            if model in SMU_MODELS:
                mods.append(
                    Module(slot, model, "SMU", SMU_MODELS[model], [slot])
                )
            elif model in WGFMU_MODELS:
                mods.append(
                    Module(slot, model, "WGFMU", "",
                           [slot * 100 + 1, slot * 100 + 2])
                )
            elif model in CMU_MODELS:
                mods.append(Module(slot, model, "MFCMU", "", [slot]))
        return mods

    @staticmethod
    def parse_value(raw: str) -> Tuple[float, str]:
        """``(value, status_char)`` from an ``FMT 1`` reply."""
        m = DATA_RE.search(raw)
        if m:
            return float(m.group(4)), m.group(1)
        m = VAL_RE.search(raw)
        if m:
            return float(m.group(0)), "?"
        raise SmuError(f"cannot parse reply: {raw!r}")

    # ---------------------------------------------------------------- setup

    def setup(
        self,
        force_ch: int,
        common_ch: Optional[int] = None,
        icomp: float = 1e-3,
        plc: float = 1.0,
    ) -> None:
        """Enable the channels and park them at 0 V."""
        # No *CLS: this mainframe answers +100 "Undefined GPIB command." to
        # it, so sending it only ADDS an error to the queue that the
        # errors() call below then has to drain. Draining the queue is the
        # part that was wanted; the *CLS was working against it.
        self.errors()
        used = [force_ch] + ([common_ch] if common_ch else [])
        self.checked("FMT 1")
        self.checked(f"AIT 0,2,{plc:g}", fatal=False)
        self.checked("CN " + ",".join(str(c) for c in used))
        if common_ch:
            self.checked(f"DV {common_ch},0,0,{icomp:.6g}")
        self.checked(f"DV {force_ch},0,0,{icomp:.6g}")

    def safe_zero(self, chans: Sequence[int] = ()) -> None:
        """Drive every named channel to 0 V and release the outputs."""
        if self.inst is None:
            return
        for ch in chans:
            try:
                self.inst.write(f"DV {ch},0,0")
            except Exception:
                pass
        try:
            self.inst.write("CL")
        except Exception:
            pass

    # ---------------------------------------------------------- measurement

    def spot_i(self, ch: int, rng: int = 0) -> Tuple[float, str]:
        """One current reading, ``(amps, status)``."""
        return self.parse_value(self.query(f"TI {ch},{rng}"))

    def spot_v(self, ch: int, rng: int = 0) -> Tuple[float, str]:
        """One voltage reading, ``(volts, status)``."""
        return self.parse_value(self.query(f"TV {ch},{rng}"))

    def read_at(
        self, ch: int, v_read: float, icomp: float, rng: int = 0,
        settle: float = 0.0,
    ) -> Tuple[float, str]:
        """Bias to ``v_read``, measure current, return to 0 V."""
        self.write(f"DV {ch},0,{v_read:.6g},{icomp:.6g}")
        if settle > 0:
            time.sleep(settle)
        value, status = self.spot_i(ch, rng)
        self.write(f"DV {ch},0,0,{icomp:.6g}")
        return value, status

    def pulse(
        self, ch: int, amplitude: float, width: float, icomp: float,
        hardware: bool = True,
    ) -> Tuple[float, str]:
        """One SMU pulse.

        ``hardware=True`` uses the SMU's own pulse generator (``PT``/``PV``)
        and returns the measured current. ``hardware=False`` falls back to a
        software-timed bias, which returns NaN -- widths below a few ms are
        not meaningful there, and nothing below ~1 ms is trustworthy at all.
        """
        if hardware:
            period = max(0.005, 1.1 * width)
            self.write(f"PT 0,{width:.6g},{period:.6g}")
            self.write(f"PV {ch},0,0,{amplitude:.6g},{icomp:.6g}")
            self.write("XE")
            return self.parse_value(self.read_line())
        self.write(f"DV {ch},0,{amplitude:.6g},{icomp:.6g}")
        time.sleep(width)
        self.write(f"DV {ch},0,0,{icomp:.6g}")
        return float("nan"), "?"

    # ---------------------------------------------------- staircase sweeps

    def sweep_setup(
        self, ch: int, rng: int = 0, hold: float = 0.0,
        delay: float = 0.0, sdelay: float = 0.0,
    ) -> None:
        """Configure a staircase sweep with time stamps (``MM 2``)."""
        for cmd in (
            "FMT 1,1",
            "TSC 1",
            f"MM 2,{ch}",
            f"CMM {ch},1",
            f"RI {ch},{rng}",
            f"WT {hold:.6g},{delay:.6g},{sdelay:.6g}",
            "WM 2,1",
        ):
            self.checked(cmd)

    def sweep(
        self,
        ch: int,
        start: float,
        stop: float,
        points: int,
        icomp: float,
        double: bool = False,
        hold: float = 0.0,
        delay: float = 0.0,
        sdelay: float = 0.0,
        plc: float = 1.0,
        label: str = "",
    ) -> SweepResult:
        """Run one staircase sweep. ``double`` sweeps up then back down."""
        if points < 2:
            raise SmuError("a sweep needs at least 2 points")
        mode = 3 if double else 1
        self.write(
            f"WV {ch},{mode},0,{start:.6g},{stop:.6g},{int(points)},{icomp:.6g}"
        )
        errs = self.errors()
        if errs:
            raise SmuError("WV -> ERR? " + " | ".join(errs))

        # Give VISA enough time for the whole staircase before it times out.
        est = hold + (2 if double else 1) * points * (
            delay + sdelay + plc * 0.02 + 0.005
        )
        self.inst.timeout = max(self.inst.timeout, int((2 * est + 30) * 1000))

        self.write("TSR")
        self.write("XE")
        self.query("*OPC?")
        errs = self.errors()
        if errs:
            raise SmuError("sweep: ERR? " + " | ".join(errs))

        nub = self.query("NUB?")
        raw = self.read_line()

        T: List[float] = []
        Iv: List[float] = []
        Vv: List[float] = []
        st: List[str] = []
        for m in DATA_RE.finditer(raw):
            kind, value = m.group(3), float(m.group(4))
            if kind == "T":
                T.append(value)
            elif kind == "I":
                Iv.append(value)
                st.append(m.group(1))
            elif kind == "V":
                Vv.append(value)

        k = min(len(Iv), len(Vv))
        if k < 2:
            raise SmuError(f"sweep returned {k} points (NUB? {nub})")
        if len(T) >= k:
            t = T[:k]
        else:
            step = delay + plc * 0.02
            t = [n * step for n in range(k)]

        self.log(
            f"** sweep {start:+.3g} -> {stop:+.3g} V: {k} points (NUB? {nub})"
        )
        return SweepResult(
            t=t,
            v=Vv[:k],
            i=Iv[:k],
            status=st[:k],
            label=label or f"{start:+.3g} to {stop:+.3g} V",
        )

    def sweep_end(self) -> None:
        """Leave sweep mode."""
        for cmd in ("TSC 0", "FMT 1"):
            try:
                self.inst.write(cmd)
            except Exception:
                pass


# -------------------------------------------------------------- plotting


def iv_panels(results: Sequence[SweepResult], title: str = "SMU I-V"):
    """Describe an I-V figure for one or more sweeps as a panel grid."""
    from ..core.plotting import Panel, PanelGrid, Series, palette

    linear = Panel(
        title=title, xlabel="V (V)", ylabel="I (uA)", row=0, col=0
    )
    log = Panel(
        title="|I| (log)", xlabel="V (V)", ylabel="|I| (A)",
        logy=True, row=0, col=1,
    )
    resistance = Panel(
        title="Resistance", xlabel="V (V)", ylabel="|V/I| (ohm)",
        logy=True, row=1, col=0,
    )
    summary = Panel(title="Summary", row=1, col=1)

    text: List[str] = []
    for n, r in enumerate(results):
        colour = palette(n)
        linear.series.append(
            Series(name=r.label, x=list(r.v), y=[x * 1e6 for x in r.i],
                   color=colour, style="line_markers")
        )
        log.series.append(
            Series(name=r.label, x=list(r.v),
                   y=[abs(x) if abs(x) > 1e-15 else 1e-15 for x in r.i],
                   color=colour, style="line_markers")
        )
        res = [abs(x) for x in r.resistance()]
        resistance.series.append(
            Series(name=r.label, x=list(r.v), y=res,
                   color=colour, style="line_markers")
        )

        finite = [x for x in res if x == x and x > 0]
        text.append(r.label)
        text.append(f"  points           = {len(r)}")
        text.append(f"  I max            = {max((abs(x) for x in r.i), default=0):.4g} A")
        if finite:
            text.append(f"  R min            = {min(finite):.4g} ohm")
            text.append(f"  R max            = {max(finite):.4g} ohm")
            text.append(f"  on/off ratio     = {max(finite) / min(finite):.4g}")
        if r.compliance_hits:
            text.append(f"  ! compliance hit on {r.compliance_hits} point(s)")
        text.append("")

    summary.text = text
    return PanelGrid(panels=[linear, log, resistance, summary], rows=2, cols=2)


def export_sweep_csv(result: SweepResult, path: str) -> str:
    """Write one sweep to ``path`` as CSV and return the path."""
    import csv
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# sweep", result.label])
        w.writerow(["# points", len(result)])
        w.writerow(["# compliance_hits", result.compliance_hits])
        w.writerow(["t_s", "V_V", "I_A", "status", "R_ohm"])
        for t, v, i, s, r in zip(
            result.t, result.v, result.i, result.status, result.resistance()
        ):
            w.writerow([
                f"{t:.10g}", f"{v:.10g}", f"{i:.10g}", s,
                "" if r != r else f"{r:.10g}",
            ])
    return path
