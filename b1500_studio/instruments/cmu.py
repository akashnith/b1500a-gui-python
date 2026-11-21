"""B1520A MFCMU: capacitance measurements over FLEX.

Ports the ``fe.cv`` and ``fe.rayleigh`` paths from the Qt tree's
``B1500Visa::runCmu`` and ``runRayleigh``.

These are not WGFMU waveforms, so they do not go through the experiment
registry: like the SMU staircase in :mod:`b1500_studio.smu`, the instrument
sweeps them itself and this module only sets them up and reads the result
back.

Two measurements, and the difference between them is the point:

* **C-V** sweeps the DC bias at a fixed AC level. A ferroelectric gives the
  butterfly, and the gap between the two branches is the memory window.
* **Rayleigh** sweeps the **AC oscillator level** at a fixed DC bias -- the
  opposite of a C-V. Below the coercive field the permittivity rises linearly
  with drive amplitude, ``eps_r(E0) = eps_init + alpha*E0``, and ``alpha`` is
  the irreversible domain-wall term. The MFCMU has no sweep mode for its
  oscillator, so each amplitude is set with its own ``ACV`` and read as a spot
  measurement.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from ..core import ferro
from .smu import DEFAULT_RESOURCE, Smu, SmuError

__all__ = [
    "CmuError",
    "CvPoint",
    "CvResult",
    "Cmu",
    "DemoCmu",
    "rayleigh_fit",
]

#: ``ACT`` settings for the four integration levels, slowest last.
INTEGRATION = ("ACT 0,1", "ACT 2,1", "ACT 2,4", "ACT 2,16")

#: Cp-G measurement mode. The MFCMU can report several equivalent circuits;
#: parallel capacitance with conductance is the one a leaky dielectric should
#: be read in, because the loss goes into G instead of corrupting C.
IMPEDANCE_CP_G = 100


class CmuError(SmuError):
    """The MFCMU refused a command or returned something unusable."""


@dataclass
class CvPoint:
    """One capacitance reading."""

    bias_v: float
    capacitance_f: float
    conductance_s: float
    ac_level_v: float = 0.0
    branch: str = ""


@dataclass
class CvResult:
    """A capacitance sweep plus what it was measured with."""

    points: List[CvPoint] = field(default_factory=list)
    frequency_hz: float = 0.0
    area_cm2: float = 0.0
    thickness_nm: float = 0.0
    label: str = ""

    def __len__(self) -> int:
        return len(self.points)

    def permittivity(self) -> List[float]:
        """Relative permittivity at every point."""
        return [
            ferro.permittivity(p.capacitance_f, self.area_cm2, self.thickness_nm)
            for p in self.points
        ]

    def memory_window_v(self) -> float:
        """Separation of the two C-V branches at their capacitance peaks.

        On a ferroelectric the butterfly's two lobes sit at the coercive
        voltages of the two sweep directions, so the gap between them is the
        memory window. Returns NaN when only one branch was measured.
        """
        peaks = {}
        for p in self.points:
            if not p.branch:
                continue
            best = peaks.get(p.branch)
            if best is None or p.capacitance_f > best.capacitance_f:
                peaks[p.branch] = p
        if len(peaks) < 2:
            return float("nan")
        values = [p.bias_v for p in peaks.values()]
        return abs(values[0] - values[1])


def rayleigh_fit(
    ac_levels_v: Sequence[float],
    capacitances_f: Sequence[float],
    area_cm2: float,
    thickness_nm: float,
) -> dict:
    """Fit ``eps_r(E0) = eps_init + alpha*E0`` over the AC amplitude sweep.

    ``E0`` is the AC field amplitude in kV/cm, so ``alpha`` comes out in
    cm/kV, which is how the Rayleigh coefficient is normally quoted.

    ``eps_init`` is the reversible lattice-plus-pinned-wall response;
    **alpha is the irreversible domain-wall term**. A large alpha relative to
    eps_init is the signature of a composition at a morphotropic phase
    boundary, where polarization rotation is nearly barrier-free and the walls
    are mobile.
    """
    if len(ac_levels_v) < 2 or thickness_nm <= 0.0 or area_cm2 <= 0.0:
        return {}
    # V -> kV/cm: E = V / d, with d in cm.
    thickness_cm = thickness_nm * 1.0e-7
    fields = [v / thickness_cm / 1000.0 for v in ac_levels_v]
    eps = [ferro.permittivity(c, area_cm2, thickness_nm) for c in capacitances_f]
    usable = [(f, e) for f, e in zip(fields, eps) if e > 0.0]
    if len(usable) < 2:
        return {}
    fx = [f for f, _ in usable]
    ey = [e for _, e in usable]
    alpha = ferro.slope(fx, ey)
    eps_init = sum(ey) / len(ey) - alpha * (sum(fx) / len(fx))
    out = {
        "eps_init": eps_init,
        "rayleigh_alpha_cm_kV": alpha,
    }
    # The share of the response the walls contribute at the largest drive.
    largest = max(fx)
    total = eps_init + alpha * largest
    if total > 0.0:
        out["domain_wall_fraction_percent"] = 100.0 * (alpha * largest) / total
    return out


class Cmu(Smu):
    """The MFCMU, driven over the same FLEX session the SMU uses."""

    def __init__(
        self,
        resource: str = DEFAULT_RESOURCE,
        channel: int = 0,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(resource=resource, log=log)
        self.channel = channel

    @property
    def name(self) -> str:
        return "MFCMU"

    def find_channel(self) -> int:
        """Slot of the installed B1520A, or raise saying what is installed."""
        if self.channel:
            return self.channel
        modules = self.scan()
        for module in modules:
            if "B1520A" in module.model:
                self.channel = module.slot
                return self.channel
        installed = ", ".join(f"{m.slot}:{m.model}" for m in modules) or "none"
        raise CmuError(
            "no B1520A MFCMU is installed in this mainframe; capacitance "
            f"measurements need one. Installed: {installed}"
        )

    def _setup(self, frequency_hz: float, integration: int) -> int:
        channel = self.find_channel()
        self.checked("FMT 1,0")
        self.checked(f"CN {channel}")
        self.checked(INTEGRATION[max(0, min(3, integration))])
        self.checked(f"FC {channel},{frequency_hz:g}")
        self.checked(f"IMP {IMPEDANCE_CP_G}")
        return channel

    def _teardown(self, channel: int) -> None:
        # DZ then CL, in that order: DZ acts THROUGH the output switch, so it
        # is only legal while the switch is still closed.
        for cmd in ("DZ", f"CL {channel}", "TSC 0", "FMT 1"):
            try:
                self.write(cmd)
            except Exception:
                pass

    def cv_sweep(
        self,
        vmax: float,
        steps: int = 101,
        frequency_hz: float = 1.0e6,
        ac_level_v: float = 0.03,
        hold_s: float = 0.1,
        step_delay_s: float = 0.01,
        integration: int = 1,
        area_cm2: float = 1.0e-4,
        thickness_nm: float = 10.0,
    ) -> CvResult:
        """DC bias staircase at a fixed AC level, swept by the instrument."""
        channel = self._setup(frequency_hz, integration)
        result = CvResult(
            frequency_hz=frequency_hz,
            area_cm2=area_cm2,
            thickness_nm=thickness_nm,
            label=f"C-V {frequency_hz:g} Hz",
        )
        try:
            self.checked("TSC 1")
            self.checked(f"ACV {channel},{ac_level_v:g}")
            self.checked("WMDCV 2,1")           # double sweep, linear
            self.checked(f"WTDCV {hold_s:g},{step_delay_s:g},0")
            self.checked(f"WDCV {channel},3,{-vmax:g},{vmax:g},{steps}")
            self.checked(f"MM 18,{channel}")    # C-V sweep measurement
            self.checked(f"IMP {IMPEDANCE_CP_G}")
            self.checked("LMN 0")
            self.checked(f"RC {channel},0")
            self.write("XE")

            raw = self.read_line()
            errors = self.errors()
            if errors:
                raise CmuError("C-V sweep reported: " + "; ".join(errors))
            result.points = self._parse_sweep(raw, vmax, steps)
        finally:
            self._teardown(channel)
        return result

    def _parse_sweep(self, raw: str, vmax: float, steps: int) -> List[CvPoint]:
        """Split the FMT 1,0 data block into (bias, C, G) triples.

        The MFCMU answers with typed fields -- ``C`` for capacitance, ``Y``
        for conductance, ``V`` for the bias it was measured at -- so the
        parse keys off the type letter rather than off position. Position
        would break the moment an extra field appeared.
        """
        points: List[CvPoint] = []
        current = {}
        for field_text in raw.replace("\r", "").replace("\n", ",").split(","):
            text = field_text.strip()
            if not text:
                continue
            kind = text[0] if text[0].isalpha() else ""
            body = text[1:] if kind else text
            # Status letters precede the type on some firmware; take the last
            # alphabetic character before the number as the type.
            while body and body[0].isalpha():
                kind = body[0]
                body = body[1:]
            try:
                value = float(body)
            except ValueError:
                continue
            if kind == "C":
                if "C" in current:
                    points.append(self._emit(current, len(points), steps))
                    current = {}
                current["C"] = value
            elif kind == "Y":
                current["Y"] = value
            elif kind == "V":
                current["V"] = value
        if "C" in current:
            points.append(self._emit(current, len(points), steps))
        return points

    @staticmethod
    def _emit(fields: dict, index: int, steps: int) -> CvPoint:
        # A double sweep returns the up branch then the down branch, so the
        # halfway point is where the direction reverses.
        branch = "up" if index < max(steps, 1) else "down"
        return CvPoint(
            bias_v=fields.get("V", 0.0),
            capacitance_f=fields.get("C", 0.0),
            conductance_s=fields.get("Y", 0.0),
            branch=branch,
        )

    def rayleigh_sweep(
        self,
        ac_levels_v: Sequence[float],
        dc_bias_v: float = 0.0,
        frequency_hz: float = 1.0e5,
        step_delay_s: float = 0.05,
        integration: int = 2,
        area_cm2: float = 1.0e-4,
        thickness_nm: float = 10.0,
    ) -> CvResult:
        """Spot capacitance at each AC oscillator level, at a fixed DC bias."""
        channel = self._setup(frequency_hz, integration)
        result = CvResult(
            frequency_hz=frequency_hz,
            area_cm2=area_cm2,
            thickness_nm=thickness_nm,
            label=f"Rayleigh at {dc_bias_v:g} V",
        )
        try:
            self.checked(f"DCV {channel},{dc_bias_v:g}")
            for level in ac_levels_v:
                # One ACV per point: the MFCMU has no sweep mode for its
                # oscillator level, which is exactly why this measurement is
                # a loop of spot readings rather than an instrument sweep.
                self.checked(f"ACV {channel},{level:g}")
                time.sleep(max(step_delay_s, 0.0))
                self.write(f"TC {channel},0")
                reply = self.read_line()
                errors = self.errors()
                if errors:
                    raise CmuError("Rayleigh point reported: " + "; ".join(errors))
                values = self._spot_values(reply)
                if len(values) < 2:
                    raise CmuError(f"invalid MFCMU response: {reply!r}")
                result.points.append(
                    CvPoint(
                        bias_v=dc_bias_v,
                        capacitance_f=values[0],
                        conductance_s=values[1],
                        ac_level_v=level,
                        branch="rayleigh",
                    )
                )
        finally:
            self._teardown(channel)
        return result

    @staticmethod
    def _spot_values(reply: str) -> List[float]:
        out: List[float] = []
        for text in reply.replace("\r", "").replace("\n", ",").split(","):
            body = text.strip()
            while body and body[0].isalpha():
                body = body[1:]
            try:
                out.append(float(body))
            except ValueError:
                continue
        return out


class DemoCmu(Cmu):
    """A simulated ferroelectric capacitor, so the paths run with no rig."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.channel = 8

    @property
    def name(self) -> str:
        return "DEMO MFCMU"

    def open(self, resource: Optional[str] = None) -> None:
        self.log("demo MFCMU: no instrument opened")

    def close(self) -> None:
        pass

    def find_channel(self) -> int:
        return self.channel

    @staticmethod
    def _capacitance(bias_v: float, ac_level_v: float, rising: bool) -> float:
        """Butterfly in bias, Rayleigh-linear in drive amplitude.

        Both dependencies are present at once because they are the two
        measurements this class exists to serve, and a device that only had
        one of them could not exercise both.
        """
        c0 = 2.0e-10
        vc = 1.0 if rising else -1.0
        # The butterfly peak sits at the coercive voltage of the direction
        # being swept, which is what separates the two branches.
        butterfly = 1.0 + 1.5 * math.exp(-((bias_v - vc) ** 2) / 0.18)
        rayleigh = 1.0 + 6.0 * ac_level_v
        return c0 * butterfly * rayleigh

    def cv_sweep(self, vmax: float, steps: int = 101, frequency_hz: float = 1.0e6,
                 ac_level_v: float = 0.03, hold_s: float = 0.1,
                 step_delay_s: float = 0.01, integration: int = 1,
                 area_cm2: float = 1.0e-4, thickness_nm: float = 10.0) -> CvResult:
        result = CvResult(frequency_hz=frequency_hz, area_cm2=area_cm2,
                          thickness_nm=thickness_nm, label=f"C-V {frequency_hz:g} Hz")
        for branch, rising in (("up", True), ("down", False)):
            for k in range(steps):
                f = 0.0 if steps <= 1 else k / (steps - 1)
                bias = (-vmax + 2.0 * vmax * f) if rising else (vmax - 2.0 * vmax * f)
                c = self._capacitance(bias, ac_level_v, rising)
                result.points.append(
                    CvPoint(bias_v=bias, capacitance_f=c,
                            conductance_s=c * 1.0e-3, ac_level_v=ac_level_v,
                            branch=branch)
                )
        return result

    def rayleigh_sweep(self, ac_levels_v: Sequence[float], dc_bias_v: float = 0.0,
                       frequency_hz: float = 1.0e5, step_delay_s: float = 0.05,
                       integration: int = 2, area_cm2: float = 1.0e-4,
                       thickness_nm: float = 10.0) -> CvResult:
        result = CvResult(frequency_hz=frequency_hz, area_cm2=area_cm2,
                          thickness_nm=thickness_nm,
                          label=f"Rayleigh at {dc_bias_v:g} V")
        for level in ac_levels_v:
            c = self._capacitance(dc_bias_v, level, True)
            result.points.append(
                CvPoint(bias_v=dc_bias_v, capacitance_f=c, conductance_s=c * 1.0e-3,
                        ac_level_v=level, branch="rayleigh")
            )
        return result
