"""Ferroelectric protocols.

Port of the ``fe.*`` family in the Qt tree's ``patternbuilder.cpp`` and
``measurement.cpp``, using the analysis in :mod:`b1500_studio.ferro`.

The integration window is the measurement
-----------------------------------------
Switching current is ``A*dP/dt``, so it flows only while the voltage is
*moving*. On the flat top of a pulse ``dV/dt`` is zero and the only thing left
to integrate is leakage -- which is identical on the P and the U pulse, so
``P - U`` collapses to noise whatever the device is doing.

Keysight's FERAM PUND therefore integrates the **ramps**, and integrates the
rising and falling ramp **separately**. This module does the same, which is
what makes a result here comparable with one from EasyEXPERT. The Qt
implementation of this arithmetic was checked against a genuine EasyEXPERT
export and reproduced all eight stored charges at zero relative error.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from ..core import ferro
from ..core.driver import Hardware, Results
from ..core.plan import Event, Wave, secs, ticks
from ..core.plotting import Panel, PanelGrid, Series, palette
from .base import (
    NaN,
    Experiment,
    Params,
    RunResult,
    RunSpec,
    finish_table,
)
from .registry import grouped, num_field

__all__ = ["FePund"]


# Phase codes carried in RunSpec.ev_phase. The sign of the pulse is implicit:
# P and U are positive, N and D negative.
PH_PRESET, PH_P, PH_U, PH_N, PH_D = 0, 1, 2, 3, 4
#: RunSpec.ev_idx distinguishes the two ramps of one pulse.
RAMP_RISE, RAMP_FALL = 0, 1

_PHASE_NAME = {PH_PRESET: "preset", PH_P: "P", PH_U: "U", PH_N: "N", PH_D: "D"}


def ramp_event(
    r: RunSpec,
    name: str,
    start: int,
    end: int,
    samples: int,
    phase: int,
    ramp: int,
    amp: float,
    width: float,
) -> None:
    """Spread ``samples`` measurement points uniformly across one ramp.

    One event rather than one per point: the WGFMU samples on a fixed
    interval, and asking for N points at that interval is exactly what the
    instrument does natively.
    """
    span = max(int(end) - int(start), 1)
    n = max(int(samples), 1)
    interval = max(1, span // n)
    r.plan.ev.append(
        Event(name=name, t=int(start), points=n, interval=interval, avg=1)
    )
    r.ev_phase.append(phase)
    r.ev_idx.append(ramp)
    r.ev_amp.append(amp)
    r.ev_width.append(width)
    # No single bias: the ramp sweeps across the event, which is the
    # point of it. Marked swept so the read-bias check skips it.
    r.ev_measure_v.append(NaN)
    r.ev_swept.append(True)


class FePund(Experiment):
    """PUND switched polarization (``fe.pund`` in the Qt tree)."""

    key = "pund"

    @property
    def name(self) -> str:
        return "PUND switched polarization"

    # -------------------------------------------------------------- fields

    def fields(self) -> List[dict]:
        return grouped(
            "PUND",
            num_field("pund_v", "Pulse amplitude [V]", "2.0"),
            num_field(
                "pund_w",
                "Pulse top width [s]",
                "1u",
                "Flat top. It is deliberately NOT integrated: dV/dt is zero "
                "there, so the only thing it would add is leakage.",
            ),
            num_field(
                "pund_edge",
                "Rise / fall time [s]",
                "1u",
                "The ramps are the measurement. Keysight requires "
                "edge / sample_interval > 10, so a short edge cannot carry "
                "many samples.",
            ),
            num_field(
                "pund_delay",
                "Delay between pulses [s]",
                "1m",
                "Keep this SHORT. If the device back-switches before U "
                "arrives, U reads switching that P already did and the "
                "reported Psw comes out low.",
            ),
            num_field(
                "pund_preset_delay",
                "Settle after preset [s]",
                "1",
                "Long enough for the poled state to relax. This is a "
                "different job from the inter-pulse delay, which is why it is "
                "a separate parameter -- EasyEXPERT uses one DelayTime for "
                "both.",
            ),
            num_field("pund_samples", "Samples per ramp", "50"),
            num_field("area", "Electrode area [cm^2]", "1e-4"),
            num_field("thickness", "Film thickness [m]", "10n"),
        )

    # --------------------------------------------------------------- build

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "ferro"
        r.label = "PUND"
        W = Wave(r.plan)

        vmax = abs(p.num("pund_v"))
        top = ticks(p.num("pund_w"))
        edge = ticks(p.num("pund_edge"))
        delay = ticks(p.num("pund_delay"))
        preset_delay = ticks(p.num("pund_preset_delay"))
        samples = int(p.num("pund_samples"))

        r.v["vmax"] = vmax
        r.v["area"] = p.num("area")
        r.v["thickness"] = p.num("thickness")
        r.v["edge_s"] = secs(edge)

        def pulse(amplitude: float, phase: int, measured: bool, trailing: int) -> None:
            name = _PHASE_NAME[phase]
            rise_start = W.now()
            W.to(amplitude, edge)
            if measured:
                ramp_event(
                    r, name, rise_start, W.now(), samples, phase, RAMP_RISE,
                    amplitude, secs(top),
                )
            W.hold(top)
            fall_start = W.now()
            W.to(0.0, edge)
            if measured:
                # The trailing ramp is a separate integral, reported as the
                # "a" charges (ChargePa and friends), exactly as EasyEXPERT
                # does. It is not folded into the rising one.
                ramp_event(
                    r, name + "a", fall_start, W.now(), samples, phase, RAMP_FALL,
                    amplitude, secs(top),
                )
            W.hold(trailing)

        # Preset leaves the device down; P and N switch it, U and D repeat the
        # same pulse on an already-switched state, so the difference is the
        # switched charge alone.
        #
        # EasyEXPERT's FERAM PUND is a four-pulse P,U,N,D with no preset: it
        # relies on the trailing D of the previous burst to pole the device,
        # which is undefined on a virgin or unknown sample. This emits the
        # classical five-pulse preset,P,U,N,D instead.
        pulse(-vmax, PH_PRESET, False, preset_delay)
        pulse(+vmax, PH_P, True, delay)
        pulse(+vmax, PH_U, True, delay)
        pulse(-vmax, PH_N, True, delay)
        pulse(-vmax, PH_D, True, delay)
        return [r]

    # ------------------------------------------------------------- analyze

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        t, ei = r.plan.samples()
        current = res.current(hw)
        n = min(len(t), len(ei), len(current))

        area = r.v.get("area", 0.0)
        thickness_m = r.v.get("thickness", 0.0)
        thickness_nm = thickness_m * 1.0e9
        to_field = 10.0 / thickness_nm if thickness_nm > 0.0 else 0.0

        # Samples belonging to each event, in device order.
        per_event: dict[int, List[int]] = {}
        for i in range(n):
            per_event.setdefault(int(ei[i]), []).append(i)

        # Integrate every ramp on its own. Plain running sum, no drift
        # removal: the ramp charge IS the signal here, and leakage is removed
        # by the P-U difference, not by flattening the integral.
        charge_of: dict[Tuple[int, int], float] = {}
        per_sample_q: List[float] = [0.0] * n
        for event_index, indices in per_event.items():
            if len(indices) < 2:
                continue
            te = [t[k] for k in indices]
            ie = [current[k] for k in indices]
            dt = (te[-1] - te[0]) / (len(te) - 1)
            q = ferro.cumulative_charge(te, ie, remove_drift=False, dt=dt)
            for slot, k in enumerate(indices):
                if slot < len(q):
                    per_sample_q[k] = q[slot]
            charge_of[(r.ev_phase[event_index], r.ev_idx[event_index])] = q[-1]

        def q_of(phase: int, ramp: int) -> float:
            return charge_of.get((phase, ramp), NaN)

        q_p, q_u = q_of(PH_P, RAMP_RISE), q_of(PH_U, RAMP_RISE)
        q_n, q_d = q_of(PH_N, RAMP_RISE), q_of(PH_D, RAMP_RISE)
        q_pa, q_ua = q_of(PH_P, RAMP_FALL), q_of(PH_U, RAMP_FALL)
        q_na, q_da = q_of(PH_N, RAMP_FALL), q_of(PH_D, RAMP_FALL)

        # Keysight's eight charges, in coulombs as EasyEXPERT reports them.
        for key, value in (
            ("charge_p_C", q_p), ("charge_u_C", q_u),
            ("charge_n_C", q_n), ("charge_d_C", q_d),
            ("charge_pa_C", q_pa), ("charge_ua_C", q_ua),
            ("charge_na_C", q_na), ("charge_da_C", q_da),
        ):
            if math.isfinite(value):
                R.summary.append((key, value))

        if all(math.isfinite(x) for x in (q_p, q_u, q_n, q_d)):
            psw = q_p - q_u
            qsw = 0.5 * (abs(q_p - q_u) + abs(q_n - q_d))
            R.summary.append(("psw_C", psw))
            R.summary.append(("qsw_C", qsw))
            if area > 0.0:
                R.summary.append(("psw_uC_cm2", psw * 1.0e6 / area))
                R.summary.append(("qsw_uC_cm2", qsw * 1.0e6 / area))
            m = ferro.analyze_pund(q_p, q_u, q_n, q_d, area)
            if m.valid:
                R.summary.append(("p_star_uC_cm2", m.p_star))
                R.summary.append(("p_hat_uC_cm2", m.p_hat))
                R.summary.append(("dp_plus_uC_cm2", m.p_switched_plus))
                R.summary.append(("dp_minus_uC_cm2", m.p_switched_minus))
                R.summary.append(("two_pr_uC_cm2", m.two_pr))
                R.summary.append(("pr_uC_cm2", m.pr))
        else:
            R.notes.append(
                "Not all four PUND ramps were measured; Psw is unavailable."
            )

        # The subtracted P-E loop.
        #
        # Integrating the P current alone leaves a loop that still carries the
        # dielectric and leakage charge. Subtracting U point by point removes
        # everything that repeats on a second identical pulse.
        #
        # The subtraction is INDEX-MATCHED, not interpolated against voltage:
        # P and U walk the same ramp with the same sample count, so sample k
        # of one is at the same instant of the waveform as sample k of the
        # other. Pairing by voltage would merge the rising and falling
        # branches and destroy the loop.
        drive: List[float] = []
        loop_p: List[float] = []
        running = 0.0
        for phase_pair, ramp in (
            ((PH_P, PH_U), RAMP_RISE),
            ((PH_P, PH_U), RAMP_FALL),
            ((PH_N, PH_D), RAMP_RISE),
            ((PH_N, PH_D), RAMP_FALL),
        ):
            switching, reference = phase_pair
            a = self._event_samples(r, per_event, switching, ramp)
            b = self._event_samples(r, per_event, reference, ramp)
            count = min(len(a), len(b))
            if count < 2:
                continue
            dt = (t[a[count - 1]] - t[a[0]]) / (count - 1)
            for k in range(count):
                running += (current[a[k]] - current[b[k]]) * dt
                polarization = running * 1.0e6 / area if area > 0.0 else 0.0
                # The drive level at this instant, as a field when the
                # thickness is known.
                level = self._level_at(r, per_event, a[k], ei)
                drive.append(level * to_field if to_field > 0.0 else level)
                loop_p.append(polarization)

        R.a["loop_drive"] = drive
        R.a["loop_p"] = loop_p
        R.a["t"] = list(t[:n])
        R.a["I"] = list(current[:n])
        R.v["field_unit"] = 1.0 if to_field > 0.0 else 0.0

        finish_table(R, r, t, ei, n, current, per_sample_q, "charge_C")
        return R

    @staticmethod
    def _event_samples(
        r: RunSpec, per_event: dict, phase: int, ramp: int
    ) -> List[int]:
        for event_index, indices in per_event.items():
            if r.ev_phase[event_index] == phase and r.ev_idx[event_index] == ramp:
                return indices
        return []

    @staticmethod
    def _level_at(r: RunSpec, per_event: dict, sample: int, ei: Sequence[int]) -> float:
        """Drive amplitude of the event this sample belongs to."""
        event_index = int(ei[sample])
        return r.ev_amp[event_index] if event_index < len(r.ev_amp) else 0.0

    # ---------------------------------------------------------------- draw

    def draw(self, results: List[RunResult]) -> PanelGrid:
        current_series: List[Series] = []
        loop_series: List[Series] = []
        for k, R in enumerate(results):
            colour = palette(k)
            if R.a.get("t"):
                current_series.append(
                    Series(x=R.a["t"], y=R.a["I"], name=R.label, color=colour)
                )
            if R.a.get("loop_drive"):
                loop_series.append(
                    Series(
                        x=R.a["loop_drive"],
                        y=R.a["loop_p"],
                        name=R.label,
                        color=colour,
                    )
                )
        field_unit = bool(results and results[0].v.get("field_unit", 0.0))
        return PanelGrid(
            rows=1,
            cols=3,
            panels=[
                Panel(
                    title="Measured current",
                    xlabel="Time (s)",
                    ylabel="Current (A)",
                    series=current_series,
                ),
                Panel(
                    title="PUND: P-U and N-D, dielectric and leakage removed",
                    xlabel="Electric field (MV/cm)" if field_unit else "Drive (V)",
                    ylabel="Polarization (uC/cm2)",
                    series=loop_series,
                ),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )
