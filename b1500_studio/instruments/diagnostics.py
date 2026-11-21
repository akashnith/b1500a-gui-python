"""Connection diagnostics for the B1500A / B1530A stack on Windows.

Everything here is read-only and hang-proof: each probe that could enter the
VISA layer runs in a subprocess with a timeout, so a broken VISA installation
reports a clear failure instead of freezing.

The checks are ordered the way the failures actually cascade:

1. Is the process 64-bit and is ``wgfmu.dll`` present?
2. Are the Keysight IO Libraries services running?  If they are not, every
   VISA call blocks forever and nothing below can succeed.
3. Does ``pyvisa`` manage ``viOpenDefaultRM()``?  (SMU path.)
4. Does ``WGFMU_openSession()`` return?  (Fast-pulse path.)
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .session import VISA_HANG_HELP

__all__ = [
    "Check",
    "Diagnosis",
    "diagnose",
    "KEYSIGHT_SERVICES",
    "INSTRUMENT_ABSENT_HELP",
]

INSTRUMENT_ABSENT_HELP = """\
The address is configured in VISA, but nothing answered at it.

That is the instrument, not this program:

  1. Is the B1500A powered on and finished booting?
  2. Is the GPIB cable seated at both ends?
  3. Does the address on the mainframe (System > Config) match the one above?
     The factory default is 17.

A listed address is not a reachable one. VISA keeps listing a GPIB address
that was configured once in Connection Expert, whether or not anything is
still at the far end -- so `list_resources()` showing it proves nothing.

Cross-check in Keysight Connection Expert and press Refresh All. If it cannot
see the instrument either, no amount of restarting VISA services will help."""

#: Services the Keysight VISA layer needs before it will answer.
KEYSIGHT_SERVICES = (
    "KeysightDistributedInfrastructureService",
    "AgilentIOLibrariesService",
    "KeysightInstrumentDiscoveryService",
)

_PROBE_RM = (
    "import pyvisa,sys\n"
    "rm=pyvisa.ResourceManager()\n"
    "sys.stdout.write(str(getattr(rm.visalib,'library_path','?'))+'\\n')\n"
    "sys.stdout.write(repr(rm.list_resources()))\n"
)


#: Does anything actually ANSWER at the address?
#
# ``list_resources()`` does not answer this. A GPIB address configured once in
# Connection Expert keeps being listed whether or not the instrument is
# powered on, so a listed address proves only that somebody once told VISA the
# address existed. Opening it and asking ``*IDN?`` is the real test, and it is
# what separates "the mainframe is off" from every software fault below it.
_PROBE_INSTRUMENT = """\
import pyvisa,sys
rm=pyvisa.ResourceManager()
try:
    inst=rm.open_resource(%r, open_timeout=5000)
    inst.timeout=5000
    sys.stdout.write(inst.query('*IDN?').strip())
    inst.close()
except Exception as exc:
    sys.stdout.write('ERR %%s: %%s' %% (type(exc).__name__, exc))
"""


@dataclass
class Check:
    """One diagnostic step."""

    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        return f"[{'OK  ' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


@dataclass
class Diagnosis:
    """Full diagnostic report."""

    checks: List[Check] = field(default_factory=list)
    advice: str = ""

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def text(self) -> str:
        out = [c.line() for c in self.checks]
        if self.advice:
            out += ["", self.advice]
        return "\n".join(out)


def _service_state(name: str) -> str:
    """``RUNNING`` / ``STOPPED`` / ``MISSING`` / ``UNKNOWN``."""
    if sys.platform != "win32":
        return "MISSING"
    try:
        p = subprocess.run(
            ["sc", "query", name],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return "UNKNOWN"
    if p.returncode != 0:
        return "MISSING"
    out = p.stdout.upper()
    if "RUNNING" in out:
        return "RUNNING"
    if "STOPPED" in out:
        return "STOPPED"
    return "UNKNOWN"


def _run_probe(code: str, timeout: float):
    """Run ``code`` in a clean subprocess. -> (status, output)."""
    try:
        p = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "timeout", ""
    except Exception as exc:
        return "error", str(exc)
    out = (p.stdout or b"").decode("utf-8", "replace").strip()
    err = (p.stderr or b"").decode("utf-8", "replace").strip()
    return ("ok", out) if p.returncode == 0 else ("error", err or out)


def diagnose(
    address: str = "GPIB0::16::INSTR",
    dll_path: Optional[str] = None,
    timeout: float = 25.0,
    probe_instrument: bool = True,
) -> Diagnosis:
    """Run every check and return a report with actionable advice."""
    d = Diagnosis()
    dll = dll_path or "wgfmu.dll"

    # 1. interpreter + DLL -------------------------------------------------
    bits = ctypes.sizeof(ctypes.c_void_p) * 8
    d.checks.append(
        Check(
            "python",
            bits == 64,
            f"{sys.version.split()[0]} {bits}-bit"
            + ("" if bits == 64 else "  -- wgfmu.dll is x64, use 64-bit Python"),
        )
    )

    found = None
    if os.path.dirname(dll):
        found = dll if os.path.isfile(dll) else None
    else:
        for folder in (
            os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"),
        ):
            candidate = os.path.join(folder, dll)
            if os.path.isfile(candidate):
                found = candidate
                break
    d.checks.append(
        Check("wgfmu.dll", found is not None, found or f"{dll} not found")
    )

    # 2. Keysight services -------------------------------------------------
    states: Dict[str, str] = {s: _service_state(s) for s in KEYSIGHT_SERVICES}
    stopped = [s for s, v in states.items() if v == "STOPPED"]
    present = [s for s, v in states.items() if v != "MISSING"]
    services_ok = bool(present) and not stopped
    d.checks.append(
        Check(
            "Keysight IO services",
            services_ok,
            ", ".join(f"{s.replace('Service', '')}={v}" for s, v in states.items()),
        )
    )

    # 3. pyvisa resource manager (SMU path) --------------------------------
    status, out = _run_probe(_PROBE_RM, timeout)
    d.checks.append(
        Check(
            "pyvisa viOpenDefaultRM()",
            status == "ok",
            out.replace("\n", " | ")
            if status == "ok"
            else (
                f"hung (> {timeout:.0f} s) -- VISA never answered"
                if status == "timeout"
                else out
            ),
        )
    )

    # 4. does anything answer at the address? ------------------------------
    #
    # Deliberately before the WGFMU probe. If the mainframe is off, that probe
    # fails too, and its failure says nothing about the fast-pulse path.
    instrument_absent = False
    if probe_instrument:
        status, out = _run_probe(_PROBE_INSTRUMENT % (address,), timeout)
        answered = status == "ok" and not out.startswith("ERR")
        instrument_absent = status == "ok" and out.startswith("ERR")
        d.checks.append(
            Check(
                f"instrument at {address}",
                answered,
                out
                if status == "ok"
                else (
                    f"hung (> {timeout:.0f} s) -- VISA never answered"
                    if status == "timeout"
                    else out
                ),
            )
        )

    # 5. WGFMU session (fast-pulse path) -----------------------------------
    if probe_instrument:
        # wgfmu.dll wants "GPIB0::16", not the full VISA resource.
        from .wgfmu_dll import normalize_address

        wgfmu_address = normalize_address(address)
        # The DLL keeps a readable message for the last failure. Reporting
        # only "rc=-5" threw that away and left the caller guessing, when the
        # DLL was willing to say 'address "GPIB0::16" is not found'.
        code = (
            "import ctypes,sys\n"
            "from ctypes import c_char_p,c_int,byref,POINTER\n"
            f"lib=ctypes.WinDLL({dll!r})\n"
            "lib.WGFMU_openSession.argtypes=[c_char_p]\n"
            "lib.WGFMU_getErrorSize.argtypes=[POINTER(c_int)]\n"
            "lib.WGFMU_getError.argtypes=[c_char_p,POINTER(c_int)]\n"
            f"rc=lib.WGFMU_openSession({wgfmu_address.encode()!r})\n"
            "msg=''\n"
            "if rc!=0:\n"
            "    n=c_int(0)\n"
            "    if lib.WGFMU_getErrorSize(byref(n))==0 and n.value>0:\n"
            "        buf=ctypes.create_string_buffer(n.value+1)\n"
            "        size=c_int(n.value+1)\n"
            "        if lib.WGFMU_getError(buf,byref(size))==0:\n"
            "            msg=buf.value.decode('latin-1','replace').strip()\n"
            "else:\n"
            "    lib.WGFMU_closeSession()\n"
            "sys.stdout.write('rc=%d%s' % (rc, (' -- '+' '.join(msg.split())) if msg else ''))\n"
        )
        status, out = _run_probe(code, timeout)
        d.checks.append(
            Check(
                f"WGFMU_openSession({wgfmu_address})",
                status == "ok" and out.startswith("rc=0"),
                out
                if status == "ok"
                else (
                    f"hung (> {timeout:.0f} s) -- blocked inside wgfmu.dll"
                    if status == "timeout"
                    else out
                ),
            )
        )

    # advice ---------------------------------------------------------------
    #
    # Ordered by what the evidence actually supports. The old code sent every
    # failure that was not a stopped service to VISA_HANG_HELP, so a DLL that
    # answered instantly with a specific error was reported as "wgfmu.dll
    # stopped responding ... the services are not running" while the services
    # were visibly RUNNING two lines above. That advice sent people to restart
    # a VISA layer that was working, and it hid the real cause.
    hung = any(not c.ok and "hung (>" in c.detail for c in d.checks)
    if stopped:
        d.advice = (
            "The Keysight IO Libraries services are stopped, so every VISA call\n"
            "blocks forever. Start them from an Administrator PowerShell:\n\n"
            + "\n".join(
                f"  Set-Service {s} -StartupType Automatic\n  Start-Service {s}"
                for s in stopped
            )
            + "\n\n" + VISA_HANG_HELP
        )
    elif hung:
        d.advice = VISA_HANG_HELP
    elif instrument_absent:
        d.advice = INSTRUMENT_ABSENT_HELP
    elif not d.ok:
        # Something failed, fast, with a message of its own. Let that message
        # stand rather than overwriting it with a guess.
        d.advice = (
            "The failing check above reports the actual error. Nothing hung and\n"
            "the Keysight services are running, so this is not the VISA-restart\n"
            "case."
        )
    return d
