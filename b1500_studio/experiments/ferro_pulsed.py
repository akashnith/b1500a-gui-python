"""Ferroelectric protocols built from pulse trains with PUND-style readout.

Ports of ``fe.speed``, ``fe.endurance``, ``fe.accumulation`` and
``fe.afe_pund`` from the Qt tree.

All four share one idea: a *pair* of identical read pulses, where the first
switches whatever is switchable and the second repeats it on an already
switched device. The difference between them is the switched charge, with the
dielectric and leakage response -- which is identical on both -- removed. The
protocols differ only in what they do to the device between read pairs.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from ..core import ferro
from ..core.driver import Hardware, Results
from ..core.plan import Wave, secs, ticks
from ..core.plotting import Panel, PanelGrid, Series, palette
from .base import NaN, Experiment, Params, RunResult, RunSpec, finish_table
from .ferroelectric import ramp_event
from .registry import grouped, num_field

__all__ = ["FeSpeed", "FeEndurance", "FeAccumulation", "FeAfePund"]

#: Phase codes. The read pair is always (switching, reference).
PH_SWITCH, PH_REFERENCE = 1, 2


def log_points(a: float, b: float, n: int) -> List[float]:
    """``n`` logarithmically spaced values from ``a`` to ``b``."""
    if n <= 1 or a <= 0.0 or b <= 0.0:
        return [a]
    return [a * (b / a) ** (i / (n - 1)) for i in range(n)]


def read_pair(
    r: RunSpec,
    W: Wave,
    amplitude: float,
    edge: int,
    top: int,
    samples: int,
    gap: int,
    idx: int,
    label: float,
) -> None:
    """Two identical ramped pulses, measured on the rising edge only.

    Only the rising ramp carries switching current, exactly as in PUND: on
    the flat top dV/dt is zero, so integrating it would add leakage and
    nothing else.
    """
    for phase in (PH_SWITCH, PH_REFERENCE):
        name = f"{'P' if phase == PH_SWITCH else 'U'}{idx}"
        start = W.now()
        W.to(amplitude, edge)
        ramp_event(r, name, start, W.now(), samples, phase, idx, amplitude, label)
        W.hold(top)
        W.to(0.0, edge)
        W.hold(gap)


def pair_charges(
    r: RunSpec,
    t: Sequence[float],
    ei: Sequence[int],
    current: Sequence[float],
    n: int,
) -> Dict[int, Tuple[float, float]]:
    """Integrated (switching, reference) charge per read-pair index."""
    per_event: Dict[int, List[int]] = {}
    for i in range(n):
        per_event.setdefault(int(ei[i]), []).append(i)

    out: Dict[int, List[float]] = {}
    for event_index, indices in per_event.items():
        if len(indices) < 2:
            continue
        phase = r.ev_phase[event_index]
        if phase not in (PH_SWITCH, PH_REFERENCE):
            continue
        te = [t[k] for k in indices]
        ie = [current[k] for k in indices]
        dt = (te[-1] - te[0]) / (len(te) - 1)
        # No drift removal: the ramp charge IS the signal, and leakage is
        # removed by the P-U difference rather than by flattening it.
        q = ferro.cumulative_charge(te, ie, remove_drift=False, dt=dt)[-1]
        slot = out.setdefault(r.ev_idx[event_index], [NaN, NaN])
        slot[0 if phase == PH_SWITCH else 1] = q
    return {k: (v[0], v[1]) for k, v in out.items()}


def nonlinearity_percent(values: Sequence[float]) -> float:
    """Deviation of the midpoint from a straight line between the ends.

    This is what decides whether a device is usable as an analog weight: a
    perfectly linear update is 0%, and the sign says which way it bends.
    """
    if len(values) < 3:
        return NaN
    first, last = values[0], values[-1]
    span = last - first
    if abs(span) < 1e-30:
        return NaN
    mid = values[len(values) // 2]
    ideal = first + 0.5 * span
    return 100.0 * (mid - ideal) / span


def _geometry() -> List[dict]:
    return [
        num_field("area", "Electrode area [cm^2]", "1e-4"),
        num_field("thickness", "Film thickness [m]", "10n"),
    ]


class _PairExperiment(Experiment):
    """Common analysis for the protocols that read with a P/U pair."""

    x_label = "x"
    x_unit = ""
    logx = False

    def _psw_series(
        self, r: RunSpec, res: Results, hw: Hardware
    ) -> Tuple[RunResult, List[float], List[float]]:
        R = RunResult(label=r.label)
        t, ei = r.plan.samples()
        current = res.current(hw)
        n = min(len(t), len(current))
        area = r.v.get("area", 0.0)

        charges = pair_charges(r, t, ei, current, n)
        xs: List[float] = []
        psw: List[float] = []
        for idx in sorted(charges):
            q_switch, q_reference = charges[idx]
            if not (math.isfinite(q_switch) and math.isfinite(q_reference)):
                continue
            label = r.ev_width[self._event_for(r, idx)]
            xs.append(label)
            scale = 1.0e6 / area if area > 0.0 else 0.0
            psw.append((q_switch - q_reference) * scale)

        R.a["x"] = xs
        R.a["psw"] = psw
        finish_table(R, r, t, ei, n, current, [0.0] * n, "charge_C")
        return R, xs, psw

    @staticmethod
    def _event_for(r: RunSpec, idx: int) -> int:
        for e, value in enumerate(r.ev_idx):
            if value == idx:
                return e
        return 0

    def draw(self, results: List[RunResult]) -> PanelGrid:
        series = [
            Series(x=R.a.get("x", []), y=R.a.get("psw", []), name=R.label,
                   color=palette(k), style="line_markers")
            for k, R in enumerate(results)
        ]
        return PanelGrid(
            rows=1,
            cols=2,
            panels=[
                Panel(
                    title=self.name,
                    xlabel=f"{self.x_label} ({self.x_unit})" if self.x_unit
                    else self.x_label,
                    ylabel="Switched polarization (uC/cm2)",
                    logx=self.logx,
                    series=series,
                ),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )


class FeSpeed(_PairExperiment):
    """Ferroelectric switching kinetics (``fe.speed``)."""

    key = "fe_speed"
    x_label = "Write pulse width"
    x_unit = "s"
    logx = True

    @property
    def name(self) -> str:
        return "FE switching speed"

    def fields(self) -> List[dict]:
        return grouped(
            "Switching speed",
            num_field("speed_write_v", "Write amplitude [V]", "-2.5"),
            num_field("speed_preset_v", "Preset amplitude [V]", "2.5"),
            num_field("speed_w_min", "Shortest write width [s]", "100n"),
            num_field("speed_w_max", "Longest write width [s]", "100u"),
            num_field("speed_points", "Width points", "8"),
            num_field(
                "speed_read_width",
                "Read pulse top width [s]",
                "10u",
                "Deliberately slow. The time resolution lives entirely in "
                "the write pulse; a slow read puts the charge on a sensitive "
                "current range instead of saturating it.",
            ),
            num_field("speed_edge", "Read rise / fall [s]", "2u"),
            num_field("speed_write_edge", "Write rise / fall [s]", "50n"),
            num_field("speed_settle", "Settle between steps [s]", "100u"),
            num_field("speed_samples", "Samples per read ramp", "40"),
            *_geometry(),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "ferro"
        r.label = "FE switching speed"
        W = Wave(r.plan)

        write_v = p.num("speed_write_v")
        preset_v = p.num("speed_preset_v")
        points = max(1, int(p.num("speed_points")))
        read_top = ticks(p.num("speed_read_width"))
        read_edge = ticks(p.num("speed_edge"))
        write_edge = ticks(p.num("speed_write_edge"))
        settle = ticks(p.num("speed_settle"))
        samples = int(p.num("speed_samples"))
        r.v["area"] = p.num("area")
        r.v["thickness"] = p.num("thickness")

        widths = log_points(p.num("speed_w_min"), p.num("speed_w_max"), points)
        for k, width in enumerate(widths):
            w = ticks(width)
            # Pole one way with a wide pulse ...
            W.to(preset_v, read_edge)
            W.hold(read_top)
            W.to(0.0, read_edge)
            W.hold(settle)
            # ... then the WRITE pulse of the width under test, the other way.
            # This is the only thing that varies between points.
            W.to(write_v, write_edge)
            W.hold(w)
            W.to(0.0, write_edge)
            W.hold(settle)
            # Read back at the preset polarity: the first read switches
            # whatever the write left switchable, the second repeats it.
            read_pair(r, W, preset_v, read_edge, read_top, samples, settle, k,
                      secs(w))
            W.hold(settle)
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R, widths, psw = self._psw_series(r, res, hw)
        if len(psw) < 2:
            R.notes.append("Not enough width points to extract kinetics.")
            return R

        saturated = max(psw)
        R.summary.append(("psw_saturated_uC_cm2", saturated))
        R.summary.append(("psw_shortest_uC_cm2", psw[0]))
        R.summary.append(("psw_longest_uC_cm2", psw[-1]))

        # t50: the width at which the response first reaches half its
        # saturated value, interpolated LOGARITHMICALLY between the two
        # widths that bracket it -- the x axis is decades, so a linear
        # interpolation across a decade would be badly wrong.
        half = 0.5 * saturated
        t50 = NaN
        for i in range(1, len(psw)):
            if psw[i - 1] < half <= psw[i]:
                lo, hi = psw[i - 1], psw[i]
                f = 0.0 if hi == lo else (half - lo) / (hi - lo)
                if widths[i - 1] > 0.0 and widths[i] > 0.0:
                    t50 = widths[i - 1] * (widths[i] / widths[i - 1]) ** f
                break
        if math.isfinite(t50):
            R.summary.append(("t50_switching_s", t50))
        else:
            R.notes.append(
                "Psw never crossed half its maximum; widen the width range "
                "so both the unswitched and the saturated ends are covered."
            )
        return R


class FeEndurance(_PairExperiment):
    """Fatigue and wake-up (``fe.endurance``)."""

    key = "fe_endurance"
    x_label = "Stress cycles"
    logx = True

    @property
    def name(self) -> str:
        return "FE endurance / fatigue"

    def fields(self) -> List[dict]:
        return grouped(
            "Endurance",
            num_field("end_v", "Amplitude [V]", "2.5"),
            num_field("end_width", "Pulse top width [s]", "1u"),
            num_field("end_edge", "Rise / fall [s]", "1u"),
            num_field("end_delay", "Delay between pulses [s]", "10u"),
            num_field("end_samples", "Samples per ramp", "30"),
            num_field(
                "end_max_cycles",
                "Maximum stress cycles",
                "10k",
                "Stress cycles are emitted as explicit vectors, so the "
                "reachable count is bounded by pattern memory. Millions of "
                "cycles need WGFMU sequence looping, which this build does "
                "not express.",
            ),
            num_field("end_reads_per_decade", "PUND reads per decade", "2"),
            *_geometry(),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "ferro"
        r.label = "FE endurance"
        W = Wave(r.plan)

        vmax = abs(p.num("end_v"))
        top = ticks(p.num("end_width"))
        edge = ticks(p.num("end_edge"))
        delay = ticks(p.num("end_delay"))
        samples = int(p.num("end_samples"))
        total = int(p.num("end_max_cycles"))
        per_decade = max(1.0, p.num("end_reads_per_decade"))
        r.v["area"] = p.num("area")
        r.v["thickness"] = p.num("thickness")

        def stress(cycles: int) -> None:
            # The stress is the same bipolar burst as the probe, just not
            # measured. Using a different waveform to stress than to read
            # would make the fatigue curve describe a history the read never
            # sampled.
            for _ in range(cycles):
                for amplitude in (+vmax, -vmax):
                    W.to(amplitude, edge)
                    W.hold(top)
                    W.to(0.0, edge)
                    W.hold(delay)

        index = 0
        read_pair(r, W, vmax, edge, top, samples, delay, index, 0.0)
        index += 1

        done = 0
        previous = 0
        steps = int(math.floor(math.log10(max(total, 1)) * per_decade)) if total >= 1 else 0
        for k in range(steps + 1):
            target = int(round(10.0 ** (k / per_decade)))
            if target <= previous or target > total:
                continue
            previous = target
            stress(target - done)
            done = target
            read_pair(r, W, vmax, edge, top, samples, delay, index, float(target))
            index += 1
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R, cycles, psw = self._psw_series(r, res, hw)
        if not psw:
            R.notes.append("No PUND reads were measured.")
            return R

        pristine = psw[0]
        R.summary.append(("psw_pristine_uC_cm2", pristine))
        R.summary.append(("psw_max_uC_cm2", max(psw)))
        R.summary.append(("psw_min_uC_cm2", min(psw)))
        if abs(pristine) > 1e-30:
            # Both are reported, because a single number cannot say which is
            # happening: HfO2-based films normally WAKE UP over the first
            # 1e3-1e5 cycles before they fatigue, so a reading taken at one
            # cycle count can show either.
            R.summary.append(
                ("wakeup_percent", 100.0 * (max(psw) - pristine) / abs(pristine))
            )
            R.summary.append(
                ("fatigue_percent", 100.0 * (min(psw) - pristine) / abs(pristine))
            )
        return R


class FeAccumulation(_PairExperiment):
    """Accumulative polarization switching (``fe.accumulation``)."""

    key = "fe_accumulation"
    x_label = "Cumulative pulses"
    logx = False

    @property
    def name(self) -> str:
        return "Accumulative polarization switching"

    def fields(self) -> List[dict]:
        return grouped(
            "Accumulation",
            num_field(
                "acc_pulse_v",
                "Sub-coercive pulse amplitude [V]",
                "0.8",
                "Deliberately below the coercive voltage: no single pulse "
                "should switch the device. If it does, this measures "
                "ordinary switching, not accumulation.",
            ),
            num_field("acc_pulse_width", "Sub-coercive pulse width [s]", "1u"),
            num_field(
                "acc_interval",
                "Interval between pulses [s]",
                "10u",
                "Matters as much as the amplitude: the induced state partly "
                "relaxes between pulses, so a train that works at 1 us "
                "spacing can fail entirely at 1 ms.",
            ),
            num_field("acc_pulses_per_step", "Pulses between reads", "10"),
            num_field("acc_steps", "Read steps", "8"),
            num_field("acc_read_v", "Read amplitude [V]", "2.5"),
            num_field("acc_read_width", "Read top width [s]", "5u"),
            num_field("acc_read_edge", "Read rise / fall [s]", "2u"),
            num_field("acc_samples", "Samples per read ramp", "30"),
            *_geometry(),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "ferro"
        r.label = "Accumulative switching"
        W = Wave(r.plan)

        pulse_v = p.num("acc_pulse_v")
        pulse_w = ticks(p.num("acc_pulse_width"))
        interval = ticks(p.num("acc_interval"))
        per_step = max(1, int(p.num("acc_pulses_per_step")))
        steps = max(1, int(p.num("acc_steps")))
        read_v = p.num("acc_read_v")
        read_top = ticks(p.num("acc_read_width"))
        read_edge = ticks(p.num("acc_read_edge"))
        samples = int(p.num("acc_samples"))
        edge = max(1, ticks(hw.edge))
        r.v["area"] = p.num("area")
        r.v["thickness"] = p.num("thickness")

        for step in range(steps):
            # Reset before every train, and apply the CUMULATIVE count from
            # that reset state.
            #
            # The PUND readout is destructive: it leaves the device fully
            # switched to the read polarity, so a train applied after it has
            # nothing left to switch. Reading once and then continuing the
            # train -- which is the obvious structure, and the one the Qt
            # implementation uses -- measures the first point and then a flat
            # line of zeros. The cost is a run quadratic in the step count,
            # which is inherent to a destructive readout, not a choice.
            W.to(-read_v, read_edge)
            W.hold(read_top)
            W.to(0.0, read_edge)
            W.hold(interval)

            applied = (step + 1) * per_step
            for _ in range(applied):
                W.to(pulse_v, edge)
                W.hold(pulse_w)
                W.to(0.0, edge)
                W.hold(interval)
            read_pair(r, W, read_v, read_edge, read_top, samples, interval,
                      step, float(applied))
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R, pulses, psw = self._psw_series(r, res, hw)
        if len(psw) < 2:
            R.notes.append("Not enough read steps to show accumulation.")
            return R

        # The read reports what is LEFT to switch, not what has switched.
        #
        # After a reset the device is fully down, so the first read switches
        # all of it. Each sub-coercive train leaves less for the next read to
        # find, so psw falls. The accumulated polarization -- the quantity
        # this measurement exists to produce, and the one that looks like an
        # LTP curve -- is the complement of that.
        reference = max(psw)
        accumulated = [reference - value for value in psw]
        R.a["psw"] = accumulated

        R.summary.append(("psw_remaining_first_uC_cm2", psw[0]))
        R.summary.append(("psw_remaining_final_uC_cm2", psw[-1]))
        R.summary.append(("p_accumulated_final_uC_cm2", accumulated[-1]))
        R.summary.append(("p_accumulated_max_uC_cm2", max(accumulated)))
        if abs(reference) > 1e-30:
            R.summary.append(
                ("accumulation_percent", 100.0 * accumulated[-1] / abs(reference))
            )
        R.summary.append(
            ("update_nonlinearity_percent", nonlinearity_percent(accumulated))
        )
        # Worth stating plainly: accumulative switching is debated, and the
        # effect can be extrinsic -- charge trapping in the stack rather than
        # true partial reversal. Comparing against fe_speed at the same
        # amplitude is the way to tell; genuine partial switching should
        # follow the same kinetics.
        R.notes.append(
            "Accumulation can be extrinsic (charge trapping). Cross-check "
            "against fe_speed at the same amplitude: genuine partial "
            "switching follows the same kinetics."
        )
        return R


class FeAfePund(Experiment):
    """AFE-PUND: PUND with the rest level at the relaxation voltage."""

    key = "afe_pund"

    @property
    def name(self) -> str:
        return "AFE-PUND switched polarization"

    def fields(self) -> List[dict]:
        return grouped(
            "AFE-PUND",
            num_field("afep_v", "Pulse amplitude [V]", "3.0"),
            num_field(
                "afep_relax_plus",
                "Positive relaxation voltage [V]",
                "1.5",
                "Run `afe` first: it reports both relaxation voltages from "
                "the four current peaks of the double loop.",
            ),
            num_field("afep_relax_minus", "Negative relaxation voltage [V]", "-1.5"),
            num_field("afep_width", "Pulse top width [s]", "1u"),
            num_field("afep_edge", "Rise / fall [s]", "1u"),
            num_field(
                "afep_relax",
                "Relaxation hold between pulses [s]",
                "50u",
                "Held at the relaxation voltage, not at zero. At zero the "
                "induced ferroelectric phase decays before U arrives and "
                "P - U reports nothing at all.",
            ),
            num_field("afep_samples", "Samples per ramp", "40"),
            *_geometry(),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "afe"
        r.label = "AFE-PUND"
        W = Wave(r.plan)

        vmax = abs(p.num("afep_v"))
        edge = ticks(p.num("afep_edge"))
        top = ticks(p.num("afep_width"))
        relax_time = ticks(p.num("afep_relax"))
        samples = int(p.num("afep_samples"))
        r.v["area"] = p.num("area")
        r.v["thickness"] = p.num("thickness")

        index = 0

        def afe_pulse(baseline: float, peak: float, phase: int, idx: int) -> None:
            start = W.now()
            W.to(peak, edge)
            ramp_event(r, f"{'P' if phase == PH_SWITCH else 'U'}{idx}", start,
                       W.now(), samples, phase, idx, peak, baseline)
            W.hold(top)
            # Back to the RELAXATION voltage, not to zero. This is the whole
            # method: at zero the field-induced ferroelectric phase decays and
            # by the time U arrives the device is antiferroelectric again,
            # switching exactly as much as P did, so P - U is zero whatever
            # the material is doing.
            W.to(baseline, edge)
            W.hold(relax_time)

        for baseline, peak in (
            (p.num("afep_relax_plus"), +vmax),
            (p.num("afep_relax_minus"), -vmax),
        ):
            W.to(baseline, edge)
            W.hold(relax_time)
            afe_pulse(baseline, peak, PH_SWITCH, index)
            afe_pulse(baseline, peak, PH_REFERENCE, index)
            index += 1
            W.to(0.0, edge)
            W.hold(relax_time)
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        t, ei = r.plan.samples()
        current = res.current(hw)
        n = min(len(t), len(current))
        area = r.v.get("area", 0.0)
        scale = 1.0e6 / area if area > 0.0 else 0.0

        charges = pair_charges(r, t, ei, current, n)
        names = {0: "plus", 1: "minus"}
        switched: List[float] = []
        for idx in sorted(charges):
            q_switch, q_reference = charges[idx]
            if not (math.isfinite(q_switch) and math.isfinite(q_reference)):
                continue
            value = (q_switch - q_reference) * scale
            switched.append(value)
            R.summary.append((f"p_pu_{names.get(idx, idx)}_uC_cm2", value))
            # How much of the induced phase was still there to give back.
            if abs(q_switch) > 1e-30:
                R.summary.append(
                    (
                        f"back_switching_{names.get(idx, idx)}_percent",
                        100.0 * q_reference / q_switch,
                    )
                )
        if switched:
            R.summary.append(
                ("p_switched_uC_cm2", sum(abs(x) for x in switched) / len(switched))
            )
        else:
            R.notes.append("Neither AFE-PUND sequence produced a usable pair.")

        finish_table(R, r, t, ei, n, current, [0.0] * n, "charge_C")
        return R

    def draw(self, results: List[RunResult]) -> PanelGrid:
        return PanelGrid(
            rows=1,
            cols=1,
            panels=[Panel(title="Summary", text=self.summary_lines(results))],
        )
