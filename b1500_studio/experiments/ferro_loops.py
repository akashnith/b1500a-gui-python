"""Ferroelectric protocols driven by a triangle or a staircase.

Ports of ``fe.pv``, ``fe.iv``, ``fe.afe`` and ``fe.ftj`` from the Qt tree.

What they share is that the drive voltage is the independent variable and the
measurement runs *along* it, so the analysis works on (V, I) pairs rather than
on per-pulse scalars. The pulse-based members of the family -- PUND and its
relatives -- live in :mod:`ferroelectric`.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from ..core import ferro
from ..core.driver import Hardware, Results
from ..core.plan import Event, Wave, secs, ticks
from ..core.plotting import Panel, PanelGrid, Series, palette
from .base import NaN, Experiment, Params, RunResult, RunSpec, finish_table
from .registry import grouped, num_field

__all__ = ["FePv", "FeIv", "FeAfe", "FeFtj"]


def triangle(W: Wave, amplitude: float, period: int) -> Tuple[int, int]:
    """``0 -> +A -> 0 -> -A -> 0`` over one period. Returns (start, end).

    A triangle, not a sine: the ramp rate is then constant on each limb, so
    the displacement current ``C dV/dt`` is a flat pedestal rather than a
    curve, and the switching peaks sit on top of something featureless.
    """
    start = W.now()
    quarter = max(1, period // 4)
    W.to(amplitude, quarter)
    W.to(0.0, quarter)
    W.to(-amplitude, quarter)
    W.to(0.0, quarter)
    return start, W.now()


def sweep_event(
    r: RunSpec,
    name: str,
    start: int,
    end: int,
    samples: int,
    phase: int,
    idx: int,
    amp: float,
) -> None:
    """Spread ``samples`` points uniformly across a swept span."""
    span = max(int(end) - int(start), 1)
    n = max(int(samples), 1)
    interval = max(1, span // n)
    r.plan.ev.append(Event(name=name, t=int(start), points=n, interval=interval, avg=1))
    r.ev_phase.append(phase)
    r.ev_idx.append(idx)
    r.ev_amp.append(amp)
    r.ev_width.append(secs(span))
    r.ev_measure_v.append(NaN)
    r.ev_swept.append(True)


def _loop_metrics(
    R: RunResult, v: Sequence[float], current: Sequence[float],
    t: Sequence[float], area: float, thickness_nm: float,
) -> Tuple[List[float], List[float]]:
    """Integrate to polarization and report the loop figures."""
    if len(v) < 4:
        R.notes.append("Too few samples to extract a loop.")
        return [], []
    dt = (t[-1] - t[0]) / max(len(t) - 1, 1)
    # Drift removal is mandatory here, unlike PUND: a P-V loop is read at
    # V = 0, and an unclosed loop puts the whole leakage integral straight
    # into Pr.
    charge = ferro.cumulative_charge(t, current, remove_drift=True, dt=dt)
    p = ferro.polarization(charge, area, centre=True)
    m = ferro.analyze_loop(v, p, current, v, thickness_nm)
    if m.valid:
        R.summary.append(("two_pr_uC_cm2", m.two_pr))
        R.summary.append(("pr_plus_uC_cm2", m.pr_plus))
        R.summary.append(("pr_minus_uC_cm2", m.pr_minus))
        R.summary.append(("p_max_uC_cm2", m.p_max))
        R.summary.append(("vc_plus_V", m.vc_plus))
        R.summary.append(("vc_minus_V", m.vc_minus))
        R.summary.append(("vc_mean_V", m.vc_mean))
        R.summary.append(("imprint_V", m.imprint))
        if thickness_nm > 0.0:
            R.summary.append(("ec_plus_MV_cm", m.ec_plus))
            R.summary.append(("ec_minus_MV_cm", m.ec_minus))
            R.summary.append(("eps_r", m.eps_r))
        R.summary.append(("peak_current_A", m.peak_current))
    else:
        R.notes.append("Loop did not close; Pr and Vc are unavailable.")
    return p, charge


def _geometry_fields() -> List[dict]:
    return [
        num_field("area", "Electrode area [cm^2]", "1e-4"),
        num_field("thickness", "Film thickness [m]", "10n"),
    ]


class _TriangleLoop(Experiment):
    """Shared build/analyse for the single-triangle loop protocols."""

    prefix = ""
    phase_name = "loop"
    #: Which simulated device this protocol expects to meet.
    hint = "ferro"

    def _common(self, p: Params, hw: Hardware, label: str) -> RunSpec:
        r = RunSpec()
        r.plan.hint = self.hint
        r.label = label
        W = Wave(r.plan)

        vmax = abs(p.num(f"{self.prefix}_v"))
        period = ticks(1.0 / max(p.num(f"{self.prefix}_f"), 1e-6))
        samples = int(p.num(f"{self.prefix}_points"))
        presets = int(p.num(f"{self.prefix}_preset_cycles"))
        rest = ticks(p.num(f"{self.prefix}_rest"))

        r.v["area"] = p.num("area")
        r.v["thickness"] = p.num("thickness")
        r.v["vmax"] = vmax

        # Preconditioning loops are driven but NOT measured. The first loop on
        # a rested film is not the same as the tenth -- it carries wake-up and
        # whatever the last measurement left behind -- so reporting it as the
        # loop would report history rather than the material.
        for _ in range(presets):
            triangle(W, vmax, period)
        W.hold(rest)
        start, end = triangle(W, vmax, period)
        sweep_event(r, self.phase_name, start, end, samples, 0, 0, vmax)
        return r

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        t, ei = r.plan.samples()
        current = res.current(hw)
        v = r.plan.sample_voltages()
        n = min(len(t), len(current), len(v))
        t, current, v = list(t[:n]), list(current[:n]), list(v[:n])

        area = r.v.get("area", 0.0)
        thickness_nm = r.v.get("thickness", 0.0) * 1.0e9
        p, charge = _loop_metrics(R, v, current, t, area, thickness_nm)

        R.a["v"] = v
        R.a["p"] = p
        R.a["i"] = current
        R.a["t"] = t
        finish_table(R, r, t, ei, n, current, p if p else [0.0] * n, "P_uC_cm2")
        return R

    def draw(self, results: List[RunResult]) -> PanelGrid:
        loops: List[Series] = []
        currents: List[Series] = []
        for k, R in enumerate(results):
            colour = palette(k)
            if R.a.get("v") and R.a.get("p"):
                loops.append(Series(x=R.a["v"], y=R.a["p"], name=R.label, color=colour))
            if R.a.get("v") and R.a.get("i"):
                currents.append(
                    Series(x=R.a["v"], y=R.a["i"], name=R.label, color=colour)
                )
        return PanelGrid(
            rows=1,
            cols=3,
            panels=[
                Panel(
                    title="P-V hysteresis loop",
                    xlabel="Applied voltage (V)",
                    ylabel="Polarization (uC/cm2)",
                    series=loops,
                ),
                Panel(
                    title="Switching current",
                    xlabel="Applied voltage (V)",
                    ylabel="Current (A)",
                    series=currents,
                ),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )


class FePv(_TriangleLoop):
    """P-V hysteresis loop (``fe.pv``)."""

    key = "pv"
    prefix = "pv"

    @property
    def name(self) -> str:
        return "P-V hysteresis loop"

    def fields(self) -> List[dict]:
        return grouped(
            "P-V loop",
            num_field("pv_v", "Amplitude [V]", "3.0"),
            num_field("pv_f", "Triangle frequency [Hz]", "1k"),
            num_field("pv_points", "Samples per loop", "200"),
            num_field(
                "pv_preset_cycles",
                "Preconditioning loops",
                "2",
                "Driven but not measured. The first loop on a rested film "
                "carries wake-up and whatever the last run left behind.",
            ),
            num_field("pv_rest", "Rest before the measured loop [s]", "1m"),
            *_geometry_fields(),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        return [self._common(p, hw, "P-V loop")]


class FeAfe(_TriangleLoop):
    """Antiferroelectric double hysteresis loop (``fe.afe``)."""

    key = "afe"
    prefix = "afe"
    phase_name = "loop"
    hint = "afe"

    @property
    def name(self) -> str:
        return "Antiferroelectric double loop"

    def fields(self) -> List[dict]:
        return grouped(
            "AFE loop",
            num_field("afe_v", "Amplitude [V]", "4.0"),
            num_field("afe_f", "Triangle frequency [Hz]", "1k"),
            num_field("afe_points", "Samples per loop", "400"),
            num_field("afe_preset_cycles", "Preconditioning loops", "2"),
            num_field("afe_rest", "Rest before the measured loop [s]", "1m"),
            *_geometry_fields(),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        return [self._common(p, hw, "AFE double loop")]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = super().analyze(r, res, hw)
        v = R.a.get("v", [])
        current = R.a.get("i", [])
        p = R.a.get("p", [])
        if len(v) < 8:
            return R

        thickness_nm = r.v.get("thickness", 0.0) * 1.0e9
        to_field = 10.0 / thickness_nm if thickness_nm > 0.0 else 0.0

        # An antiferroelectric shows FOUR switching-current peaks per cycle,
        # not two: the field-induced AFE->FE transition going out and the
        # FE->AFE relaxation coming back, in each polarity. Finding the
        # extremum in each quarter of the triangle locates them.
        quarter = len(v) // 4
        # The sign is the direction the POLARIZATION moves in that quarter,
        # not the sign of the voltage. The triangle runs 0 -> +A -> 0 -> -A
        # -> 0, so P rises in quarters 1 and 4 and falls in 2 and 3. Using
        # the voltage's sign instead put the backward transition above the
        # forward one and made the hysteresis come out negative.
        names = (
            ("v_afe_to_fe_plus_V", +1),   # 0 -> +A, P rising
            ("v_fe_to_afe_plus_V", -1),   # +A -> 0, P falling back
            ("v_afe_to_fe_minus_V", -1),  # 0 -> -A, P falling
            ("v_fe_to_afe_minus_V", +1),  # -A -> 0, P rising back
        )
        found: List[float] = []
        for q, (label, sign) in enumerate(names):
            lo, hi = q * quarter, min((q + 1) * quarter, len(v))
            if hi - lo < 2:
                found.append(NaN)
                continue
            # Positive quarters look for the maximum current, negative ones
            # for the minimum: the transition shows as a peak of the sign the
            # polarization is moving in.
            best = lo
            for i in range(lo, hi):
                if (sign > 0 and current[i] > current[best]) or (
                    sign < 0 and current[i] < current[best]
                ):
                    best = i
            found.append(v[best])
            R.summary.append((label, v[best]))
            if to_field > 0.0:
                R.summary.append((label.replace("_V", "_MV_cm"), v[best] * to_field))

        if all(math.isfinite(x) for x in found[:2]):
            R.summary.append(("transition_hysteresis_V", found[0] - found[1]))

        # The identification test. A genuine antiferroelectric has near-zero
        # remanence by construction, so 2Pr is meaningless for it and is not
        # reported as a figure of merit. A large ratio means the loop is
        # ferroelectric or leakage-driven, not antiferroelectric -- and a
        # pinched loop from defects can mimic a double loop closely enough to
        # fool a visual check.
        if p:
            p_max = max(abs(x) for x in p)
            at_zero = ferro.cross_values(v, p, 0.0, [True] * len(v), 0) or [0.0]
            remanence = sum(abs(x) for x in at_zero) / len(at_zero)
            if p_max > 0.0:
                ratio = remanence / p_max
                R.summary.append(("remanence_ratio", ratio))
                if ratio > 0.3:
                    R.notes.append(
                        f"remanence_ratio = {ratio:.2f}: this loop is not "
                        "antiferroelectric. A real AFE sits near zero."
                    )

        # Energy storage, the other reason to run this measurement.
        #
        # The work density is the integral of E dD, which on this loop is
        # V dP, NOT P dV: those differ by integration by parts, and using the
        # wrong one gave a recovered energy larger than the charged energy --
        # 102% efficiency out of a passive capacitor.
        #
        # Only the positive-field charge/discharge is counted. Charging is
        # where P rises with V, recovery is where it falls back.
        if p and thickness_nm > 0.0:
            charged = recovered = 0.0
            for i in range(1, len(v)):
                dp = p[i] - p[i - 1]
                v_mid = 0.5 * (v[i] + v[i - 1])
                if v_mid < 0.0:
                    continue
                work = v_mid * dp * 1.0e-6  # V * uC/cm^2 -> J/cm^2
                if dp > 0.0:
                    charged += work
                else:
                    recovered -= work
            # J/cm^2 over the film thickness -> J/cm^3.
            scale = 1.0 / (thickness_nm * 1.0e-7)
            charged_density = abs(charged) * scale
            recovered_density = abs(recovered) * scale
            R.summary.append(("energy_charged_J_cm3", charged_density))
            R.summary.append(("energy_recovered_J_cm3", recovered_density))
            if charged_density > 0.0:
                efficiency = 100.0 * recovered_density / charged_density
                R.summary.append(("energy_efficiency_percent", efficiency))
                if efficiency > 100.5:
                    R.notes.append(
                        f"energy_efficiency_percent = {efficiency:.1f}: a "
                        "passive capacitor cannot return more than it stored, "
                        "so the loop is not closed or the drift removal failed."
                    )
        return R


class FeIv(Experiment):
    """FE I-V / switching current at two frequencies (``fe.iv``)."""

    key = "fe_iv"

    @property
    def name(self) -> str:
        return "FE I-V / switching current"

    def fields(self) -> List[dict]:
        return grouped(
            "FE I-V",
            num_field("iv_v", "Amplitude [V]", "3.0"),
            num_field("iv_points", "Samples per triangle", "200"),
            num_field("iv_f_low", "Low frequency [Hz]", "100"),
            num_field("iv_f_high", "High frequency [Hz]", "10k"),
            num_field("iv_rest", "Rest between the two [s]", "1m"),
            *_geometry_fields(),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "ferro"
        r.label = "FE I-V"
        W = Wave(r.plan)

        vmax = abs(p.num("iv_v"))
        samples = int(p.num("iv_points"))
        rest = ticks(p.num("iv_rest"))
        r.v["area"] = p.num("area")
        r.v["thickness"] = p.num("thickness")
        r.v["f_low"] = p.num("iv_f_low")
        r.v["f_high"] = p.num("iv_f_high")

        # The same triangle at two ramp rates. Switching current scales with
        # dV/dt and ohmic leakage does not, which is the whole separation.
        for leg, frequency in enumerate((p.num("iv_f_low"), p.num("iv_f_high"))):
            if leg > 0:
                W.hold(rest)
            start, end = triangle(W, vmax, ticks(1.0 / max(frequency, 1e-6)))
            sweep_event(r, f"f{leg + 1}", start, end, samples, leg, leg, vmax)
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        t, ei = r.plan.samples()
        current = res.current(hw)
        v = r.plan.sample_voltages()
        n = min(len(t), len(current), len(v))

        legs: dict = {0: ([], [], []), 1: ([], [], [])}
        for i in range(n):
            leg = r.ev_idx[int(ei[i])]
            if leg in legs:
                legs[leg][0].append(v[i])
                legs[leg][1].append(current[i])
                legs[leg][2].append(t[i])

        f_low = r.v.get("f_low", 0.0)
        f_high = r.v.get("f_high", 0.0)
        v1, i1, _ = legs[0]
        v2, i2, _ = legs[1]

        leak = ferro.leakage_from_two_frequencies(v1, i1, f_low, v2, i2, f_high)
        if leak:
            # What is left after removing the frequency-independent part is
            # the displacement + switching current, which is what a P-V loop
            # integrates. Reporting the leak itself is the point: it says how
            # much of the "loop" was never ferroelectric.
            R.a["v_leak"] = list(v1)
            R.a["i_leak"] = leak
            peak_leak = max((abs(x) for x in leak), default=0.0)
            # Referenced to the HIGH-frequency leg, which is the one a P-V
            # loop is actually measured at. Comparing against the slow leg
            # measures the wrong thing: the slow leg has the least
            # displacement current, so the share comes out inflated.
            peak_total = max((abs(x) for x in i2), default=0.0)
            R.summary.append(("peak_leakage_A", peak_leak))
            R.summary.append(("peak_current_A", peak_total))
            if peak_total > 0.0:
                share = 100.0 * peak_leak / peak_total
                R.summary.append(("leakage_share_percent", share))
                if share > 100.0:
                    # Not a leaky device -- an invalid separation. The method
                    # assumes the non-leak current scales linearly with
                    # frequency, which fails when a switching spike is
                    # narrower than the sample interval at one of the two
                    # frequencies and is therefore averaged away at that one.
                    R.notes.append(
                        f"leakage_share_percent = {share:.0f}: a leakage "
                        "larger than the total current is impossible, so the "
                        "two-frequency separation is invalid here. Raise the "
                        "sample count, or move the two frequencies closer "
                        "together, so both legs resolve the switching peak."
                    )
                elif share > 50.0:
                    R.notes.append(
                        f"{share:.0f}% of the peak current is frequency-"
                        "independent leakage; a P-V loop from this device "
                        "would be mostly leakage, not polarization."
                    )
        else:
            R.notes.append("Two-frequency separation needs both legs measured.")

        R.a["v1"], R.a["i1"] = v1, i1
        R.a["v2"], R.a["i2"] = v2, i2
        finish_table(R, r, t, ei, n, current, v[:n], "drive_V")
        return R

    def draw(self, results: List[RunResult]) -> PanelGrid:
        raw: List[Series] = []
        leak: List[Series] = []
        for k, R in enumerate(results):
            if R.a.get("v1"):
                raw.append(
                    Series(x=R.a["v1"], y=R.a["i1"], name="low f", color=palette(0))
                )
            if R.a.get("v2"):
                raw.append(
                    Series(x=R.a["v2"], y=R.a["i2"], name="high f", color=palette(1))
                )
            if R.a.get("v_leak"):
                leak.append(
                    Series(
                        x=R.a["v_leak"], y=R.a["i_leak"], name="leakage",
                        color=palette(2),
                    )
                )
        return PanelGrid(
            rows=1,
            cols=3,
            panels=[
                Panel(
                    title="I-V at two ramp rates",
                    xlabel="Applied voltage (V)",
                    ylabel="Current (A)",
                    series=raw,
                ),
                Panel(
                    title="Separated ohmic leakage",
                    xlabel="Applied voltage (V)",
                    ylabel="Current (A)",
                    series=leak,
                ),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )


class FeFtj(Experiment):
    """Ferroelectric tunnel junction electroresistance (``fe.ftj``)."""

    key = "ftj"

    @property
    def name(self) -> str:
        return "Ferroelectric memristive (FTJ)"

    def fields(self) -> List[dict]:
        return grouped(
            "FTJ",
            num_field("ftj_pole_v", "Maximum poling amplitude [V]", "3.0"),
            num_field("ftj_steps", "Poling steps around the loop", "41"),
            num_field("ftj_pole_width", "Poling pulse width [s]", "1u"),
            num_field("ftj_relax", "Relax before reading [s]", "1m"),
            num_field(
                "ftj_read_v",
                "Read voltage [V]",
                "0.1",
                "Must be far below the coercive voltage: the read has to "
                "report the state, not change it.",
            ),
            num_field("ftj_read_width", "Read pulse width [s]", "10u"),
            *_geometry_fields(),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "ferro"
        r.label = "FTJ electroresistance"
        W = Wave(r.plan)

        vmax = abs(p.num("ftj_pole_v"))
        steps = max(2, int(p.num("ftj_steps")))
        pole_width = ticks(p.num("ftj_pole_width"))
        relax = ticks(p.num("ftj_relax"))
        read_v = p.num("ftj_read_v")
        read_width = ticks(p.num("ftj_read_width"))
        edge = max(1, ticks(hw.edge))
        r.v["read_v"] = read_v

        for k in range(steps):
            # The poling amplitude walks a closed loop 0 -> +Vmax -> -Vmax -> 0,
            # so the resistance is measured on both branches and the memory
            # window is visible as the gap between them.
            phase_pos = 0.0 if steps <= 1 else 4.0 * k / float(steps - 1)
            if phase_pos <= 1.0:
                pole = vmax * phase_pos
            elif phase_pos <= 3.0:
                pole = vmax * (2.0 - phase_pos)
            else:
                pole = vmax * (phase_pos - 4.0)

            W.to(pole, edge)
            W.hold(pole_width)
            W.to(0.0, edge)
            W.hold(relax)

            W.to(read_v, edge)
            start = W.now()
            W.hold(read_width)
            # One averaged read per poling step. The independent variable is
            # the poling voltage, so the plot is R against V_pole.
            r.plan.ev.append(
                Event(name=f"read{k}", t=start, points=1, interval=max(1, read_width),
                      avg=max(1, read_width))
            )
            r.ev_phase.append(1)
            r.ev_idx.append(k)
            r.ev_amp.append(pole)
            r.ev_width.append(secs(pole_width))
            r.ev_measure_v.append(read_v)
            r.ev_swept.append(False)
            W.to(0.0, edge)
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        t, ei = r.plan.samples()
        current = res.current(hw)
        n = min(len(t), len(current))
        read_v = r.v.get("read_v", 0.0)

        pole: List[float] = []
        resistance: List[float] = []
        for i in range(n):
            e = int(ei[i])
            pole.append(r.ev_amp[e])
            resistance.append(
                abs(read_v / current[i]) if abs(current[i]) > 1e-15 else NaN
            )

        finite = [x for x in resistance if math.isfinite(x) and x > 0.0]
        if finite:
            r_on, r_off = min(finite), max(finite)
            R.summary.append(("r_on_ohm", r_on))
            R.summary.append(("r_off_ohm", r_off))
            # The TER ratio is what an FTJ exists to provide.
            R.summary.append(("ter_ratio", r_off / r_on))
            R.summary.append(("ter_percent", 100.0 * (r_off - r_on) / r_on))
        else:
            R.notes.append("No finite read currents; TER is unavailable.")

        R.a["pole"] = pole
        R.a["r"] = resistance
        finish_table(R, r, t, ei, n, current, resistance, "R_ohm")
        return R

    def draw(self, results: List[RunResult]) -> PanelGrid:
        series = [
            Series(x=R.a.get("pole", []), y=R.a.get("r", []), name=R.label,
                   color=palette(k), style="line_markers")
            for k, R in enumerate(results)
        ]
        return PanelGrid(
            rows=1,
            cols=2,
            panels=[
                Panel(
                    title="Tunnelling electroresistance",
                    xlabel="Poling voltage (V)",
                    ylabel="Read resistance (ohm)",
                    logy=True,
                    series=series,
                ),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )
