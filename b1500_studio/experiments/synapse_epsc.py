"""Synapse: single-pulse EPSC.

Port of ``class SynapseEPSC`` in ``pkg/src/experiments.cpp``.
Baseline reads -> one stimulus pulse -> logarithmic read train, with
exponential / KWW decay fits and an approximate pulse energy.
"""

from __future__ import annotations

import math
from typing import List

from ..core.driver import Hardware, Results
from ..core.plan import Wave, parse_list, secs, ticks
from ..core.plotting import Panel, PanelGrid, Series, palette
from .base import (
    NaN,
    Experiment,
    Params,
    RunResult,
    RunSpec,
    _fit_beta,
    add_event,
    eng,
    finish_table,
    fit_kww,
    mean_of,
    read_pulse,
    require_finite,
)
from .registry import grouped, num_field, text_field

__all__ = ["SynapseEPSC"]


class SynapseEPSC(Experiment):
    """One stimulus pulse, then a log-spaced low-bias read train."""

    key = "epsc"

    @property
    def name(self) -> str:
        return "Synapse  single-pulse EPSC"

    def fields(self) -> List[dict]:
        return [
            *grouped(
                "Stimulus pulse",
                text_field("epsc_v_p", "Amplitude(s) (V)", "0.7",
                           "one run per value; comma list or lin(start,stop,n)"),
                text_field("epsc_w_p", "Width(s) (s)", "1u",
                           "one run per amplitude/width pair"),
                num_field("epsc_pulse_samples", "Current samples in pulse", "12",
                          "2-200 averaged samples across the flat top"),
                num_field("epsc_pause", "Pause between runs (s)", "0.2"),
            ),
            *grouped(
                "Baseline and EPSC read train",
                num_field("epsc_v_read", "Read amplitude (V)", "0.1",
                          "low-bias read; typically 0.05-0.2 V"),
                num_field("epsc_t_read", "Read width (s)", "1u"),
                num_field("epsc_n_pre", "Baseline reads", "3"),
                num_field("epsc_t_rest", "Baseline spacing (s)", "1u"),
                num_field("epsc_t_first", "First post-pulse read (s)", "10u"),
                num_field("epsc_t_last", "Last post-pulse read (s)", "100m"),
                num_field("epsc_n_reads", "Post-pulse reads", "24"),
            ),
        ]

    def pause_between_runs(self, p: Params) -> float:
        return p.num("epsc_pause")

    # --------------------------------------------------------------- build

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        amplitudes = parse_list(p.str("epsc_v_p"))
        widths = parse_list(p.str("epsc_w_p"))
        require_finite(amplitudes, "EPSC amplitude")
        require_finite(widths, "EPSC pulse width")
        if len(amplitudes) * len(widths) > 32:
            raise ValueError("EPSC amplitude/width sweep is limited to 32 runs")
        return [self.one(p, hw, v, w) for v in amplitudes for w in widths]

    def one(self, p: Params, hw: Hardware, vp: float, width: float) -> RunSpec:
        r = RunSpec()
        r.plan.hint = "synapse"
        r.label = f"Vp = {eng(vp, 'V', 3)}, width = {eng(width, 's', 3)}"
        W = Wave(r.plan)

        edge = ticks(hw.edge)
        read_width = ticks(p.num("epsc_t_read"))
        pulse_width = ticks(width)
        vr = p.num("epsc_v_read")
        pre_reads = p.integer("epsc_n_pre")
        samples = p.integer("epsc_pulse_samples")
        read_count = p.integer("epsc_n_reads")

        if pre_reads < 1 or read_count < 2:
            raise ValueError(
                "EPSC needs at least one baseline read and two post-pulse reads"
            )
        if samples < 2 or samples > 200:
            raise ValueError("pulse samples must be between 2 and 200")
        if pulse_width // samples < 1:
            raise ValueError(
                "pulse width is too short for the requested current samples"
            )

        delays = parse_list(
            f"log({p.str('epsc_t_first')},{p.str('epsc_t_last')},{read_count})"
        )
        require_finite(delays, "EPSC read delay")

        r.v["vr"] = vr
        r.v["vReadDevice"] = vr - hw.sink_bias
        r.v["vp"] = vp
        r.v["width"] = secs(pulse_width)

        for i in range(pre_reads):
            read_pulse(r, W, hw, vr, read_width, f"baseline{i}", 0, i, 0.0, 0.0)
            W.hold(ticks(p.num("epsc_t_rest")))

        W.to(vp, edge)
        pulse_start = W.now()
        W.hold(pulse_width)
        pulse_end = W.now()
        W.to(0.0, edge)
        pulse_interval = max(1, (pulse_end - pulse_start) // samples)
        pulse_points = (pulse_end - pulse_start) // pulse_interval
        add_event(
            r, "stimulus", pulse_start, pulse_points, pulse_interval,
            pulse_interval, 1, 0, vp, secs(pulse_width), vp - hw.sink_bias,
        )

        end_of_pulse = pulse_end + edge
        for i in range(read_count):
            target = end_of_pulse + ticks(delays[i])
            if target < W.now():
                raise ValueError(
                    "post-pulse read delays are too close for the selected read width"
                )
            if target > W.now():
                W.hold(target - W.now())
            read_pulse(r, W, hw, vr, read_width, f"post{i}", 2, i, vp, delays[i])

        W.hold(ticks(1e-6))
        r.v["pulseEnd"] = secs(end_of_pulse)
        return r

    # ------------------------------------------------------------- analyze

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        current = res.current(hw)
        t, ei = r.plan.samples()
        n = min(len(current), len(t))
        R = RunResult(label=r.label)
        if n < len(t):
            R.notes.append(f"partial data: {n}/{len(t)} points")

        I = [0.0] * n
        G = [NaN] * n
        time_after = [NaN] * n
        baseline_g: List[float] = []
        pulse_time: List[float] = []
        pulse_current: List[float] = []
        energy = 0.0

        for i in range(n):
            e = int(ei[i])
            I[i] = abs(current[i])
            measure_v = abs(r.ev_measure_v[e])
            G[i] = I[i] / measure_v if measure_v > 0.0 else NaN
            phase = r.ev_phase[e]
            if phase == 0:
                baseline_g.append(G[i])
            elif phase == 1:
                pulse_time.append(t[i] - r.v["pulseEnd"] + r.v["width"])
                pulse_current.append(I[i])
                interval = secs(r.plan.ev[e].interval)
                energy += I[i] * measure_v * interval
            elif phase == 2:
                time_after[i] = t[i] - r.v["pulseEnd"]

        g0 = mean_of(baseline_g)
        if baseline_g:
            lo, hi = min(baseline_g), max(baseline_g)
            baseline_spread = (hi - lo) / g0 * 100.0 if g0 > 0.0 else NaN
        else:
            baseline_spread = NaN

        peak_delta_g = -math.inf
        response_t: List[float] = []
        response_g: List[float] = []
        fit_t: List[float] = []
        fit_g: List[float] = []
        for i in range(n):
            if r.ev_phase[int(ei[i])] != 2:
                continue
            delta = G[i] - g0
            response_t.append(time_after[i])
            response_g.append(delta)
            peak_delta_g = max(peak_delta_g, delta)
            if time_after[i] > 0.0 and math.isfinite(delta):
                fit_t.append(time_after[i])
                fit_g.append(max(delta, 0.0))
        if not math.isfinite(peak_delta_g):
            peak_delta_g = NaN

        exponential = _fit_beta(fit_t, fit_g, 1.0)
        stretched = fit_kww(fit_t, fit_g)

        R.a["t"] = response_t
        R.a["deltaG"] = response_g
        R.a["pulseT"] = pulse_time
        R.a["pulseI"] = pulse_current

        R.summary.append(("G baseline (S)", g0))
        R.summary.append(("baseline read spread (%)", baseline_spread))
        R.summary.append(("EPSC peak delta G (S)", peak_delta_g))
        R.summary.append(
            ("EPSC peak delta I (A)", peak_delta_g * abs(r.v["vReadDevice"]))
        )
        R.summary.append(("tau exponential (s)", exponential.tau if exponential.ok else NaN))
        R.summary.append(("tau stretched (s)", stretched.tau if stretched.ok else NaN))
        R.summary.append(("beta stretched", stretched.beta if stretched.ok else NaN))
        R.summary.append(("pulse energy estimate (J)", energy))
        if math.isfinite(baseline_spread) and baseline_spread > 5.0:
            R.notes.append(
                "baseline reads vary by more than 5%; verify read "
                "non-disturbance and measurement noise"
            )

        if exponential.ok and fit_t and fit_t[0] > 0:
            fit_time: List[float] = []
            fit_response: List[float] = []
            for i in range(101):
                tv = fit_t[0] * (fit_t[-1] / fit_t[0]) ** (i / 100.0)
                fit_time.append(tv)
                fit_response.append(
                    exponential.ginf + exponential.A * math.exp(-tv / exponential.tau)
                )
            R.a["fitT"] = fit_time
            R.a["fitG"] = fit_response
        else:
            R.a["fitT"] = []
            R.a["fitG"] = []

        finish_table(R, r, t, ei, n, I, time_after, "t_since_pulse_s")
        delta_g = [NaN] * n
        for i in range(n):
            if r.ev_phase[int(ei[i])] == 2:
                delta_g[i] = G[i] - g0
        R.cols.append("G_S")
        R.data.append(G)
        R.cols.append("delta_G_S")
        R.data.append(delta_g)
        return R

    # ---------------------------------------------------------------- draw

    def draw(self, results: List[RunResult]) -> PanelGrid:
        response = Panel(
            title="EPSC response after stimulus",
            xlabel="time after pulse (s)",
            ylabel="delta G (uS)",
            logx=True, row=0, col=0,
        )
        pulse = Panel(
            title="Current during stimulus",
            xlabel="time into pulse (s)",
            ylabel="|I| (uA)",
            row=0, col=1,
        )
        summary = Panel(title="EPSC metrics", row=1, col=0)
        protocol = Panel(
            title="Protocol",
            text=[
                "One stimulus pulse, followed by low-bias reads.",
                "Decay fits use baseline-subtracted conductance.",
            ],
            row=1, col=1,
        )

        for i, R in enumerate(results):
            col = palette(i)
            response.series.append(
                Series(name=R.label, x=R.a["t"],
                       y=[g * 1e6 for g in R.a["deltaG"]],
                       color=col, style="line_markers")
            )
            if R.a.get("fitT"):
                response.series.append(
                    Series(name="exponential fit", x=R.a["fitT"],
                           y=[g * 1e6 for g in R.a["fitG"]],
                           color=col, dashed=True)
                )
            pulse.series.append(
                Series(name=R.label, x=R.a["pulseT"],
                       y=[v * 1e6 for v in R.a["pulseI"]],
                       color=col, style="line_markers")
            )

        summary.text = self.summary_lines(results)
        return PanelGrid(panels=[response, pulse, summary, protocol], rows=2, cols=2)
