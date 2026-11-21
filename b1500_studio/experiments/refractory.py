"""Neuron: refractory period from paired firing spikes.

Port of ``class NeuronRefractory`` in ``pkg/src/experiments.cpp``.
Two supra-threshold pulses separated by dt; extracts the absolute refractory
period (no second spike) and the relative refractory period (90 % recovery).
"""

from __future__ import annotations

import math
from typing import Dict, List

from ..core.driver import Hardware, Results
from ..core.plan import Wave, parse_list, secs, ticks
from ..core.plotting import Panel, PanelGrid, Series, palette
from .base import (
    NaN,
    Experiment,
    Params,
    RunResult,
    RunSpec,
    add_event,
    finish_table,
    mean_of,
    require_finite,
)
from .registry import grouped, num_field, text_field

__all__ = ["NeuronRefractory"]


class NeuronRefractory(Experiment):
    """Paired supra-threshold pulses swept over their separation."""

    key = "refractory"

    @property
    def name(self) -> str:
        return "Neuron  refractory period (paired firing spikes)"

    def fields(self) -> List[dict]:
        return [
            *grouped(
                "Paired-pulse stimulus (Refractory)",
                num_field("ref_v_p", "Spike pulse amplitude (V)", "0.8",
                          "supra-threshold pulse amplitude to trigger firing"),
                num_field("ref_w_p", "Pulse width (s)", "2u",
                          "width of Pulse 1 and Pulse 2"),
                text_field("ref_dt_list", "Inter-pulse interval(s) dt (s)",
                           "log(2u,200u,8)",
                           "list or log(start,stop,n): swept delta t"),
                num_field("ref_samples", "Samples across pulse", "20",
                          "time-resolved measurement points during the flat top"),
            ),
            *grouped(
                "Spike detection & recovery",
                num_field("ref_i_th", "Spike threshold |I| (A)", "10u",
                          "threshold current for firing event confirmation"),
                num_field("ref_t_rest", "Rest between pairs (s)", "0.5",
                          "complete relaxation time between pair tests"),
                num_field("ref_pause", "Pause between runs (s)", "0.5"),
            ),
        ]

    def pause_between_runs(self, p: Params) -> float:
        return p.num("ref_pause")

    # --------------------------------------------------------------- build

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        dts = parse_list(p.str("ref_dt_list"))
        require_finite(dts, "refractory interval")
        return [self.one(p, hw, dts)]

    def one(self, p: Params, hw: Hardware, dts: List[float]) -> RunSpec:
        r = RunSpec()
        r.plan.hint = "neuron"
        r.label = "Refractory sweep"
        W = Wave(r.plan)

        e = ticks(hw.edge)
        vp = p.num("ref_v_p")
        wp = ticks(p.num("ref_w_p"))
        S = max(2, min(64, p.integer("ref_samples")))
        rest = ticks(p.num("ref_t_rest"))

        r.v["vp"] = vp
        r.v["wp"] = secs(wp)
        r.v["ith"] = p.num("ref_i_th")

        for i, dt in enumerate(dts):
            dt_ticks = ticks(dt)

            # Pulse 1.
            W.to(vp, e)
            fs1 = W.now()
            W.hold(wp)
            fe1 = W.now()
            W.to(0.0, e)
            iv1 = max(1, (fe1 - fs1) // S)
            add_event(r, f"p1_{i}", fs1, (fe1 - fs1) // iv1, iv1, iv1,
                      1, i, vp, dt, vp - hw.sink_bias)

            # Inter-pulse interval.
            W.hold(dt_ticks)

            # Pulse 2.
            W.to(vp, e)
            fs2 = W.now()
            W.hold(wp)
            fe2 = W.now()
            W.to(0.0, e)
            iv2 = max(1, (fe2 - fs2) // S)
            add_event(r, f"p2_{i}", fs2, (fe2 - fs2) // iv2, iv2, iv2,
                      2, i, vp, dt, vp - hw.sink_bias)

            W.hold(rest)
        return r

    # ------------------------------------------------------------- analyze

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        cur = res.current(hw)
        t, ei = r.plan.samples()
        n = min(len(cur), len(t))
        R = RunResult(label=r.label)
        if n < len(t):
            R.notes.append(f"partial data: {n}/{len(t)} points")
        ith = r.v["ith"]

        p1_max: Dict[int, float] = {}
        p2_max: Dict[int, float] = {}
        dt_map: Dict[int, float] = {}
        I = [0.0] * n

        for i in range(n):
            e = int(ei[i])
            I[i] = abs(cur[i])
            idx = int(r.ev_idx[e])
            ph = r.ev_phase[e]
            dt_map[idx] = r.ev_width[e]
            if ph == 1:
                p1_max[idx] = max(p1_max.get(idx, 0.0), I[i])
            elif ph == 2:
                p2_max[idx] = max(p2_max.get(idx, 0.0), I[i])

        dt_arr: List[float] = []
        i1_arr: List[float] = []
        i2_arr: List[float] = []
        ratio_arr: List[float] = []
        fired_arr: List[float] = []
        t_abs = NaN
        t_rel = NaN

        for idx in sorted(dt_map):
            dt = dt_map[idx]
            i1 = p1_max.get(idx, 0.0)
            i2 = p2_max.get(idx, 0.0)
            ratio = (i2 / i1) * 100.0 if i1 > 1e-12 else 0.0
            fired = i2 >= ith

            dt_arr.append(dt)
            i1_arr.append(i1)
            i2_arr.append(i2)
            ratio_arr.append(ratio)
            fired_arr.append(1.0 if fired else 0.0)

            if not fired:
                t_abs = dt
            if math.isnan(t_rel) and ratio >= 90.0:
                t_rel = dt

        R.a["dt"] = dt_arr
        R.a["i1"] = i1_arr
        R.a["i2"] = i2_arr
        R.a["ratio"] = ratio_arr
        R.a["fired"] = fired_arr
        # The Qt version reads R.v["ith"] when drawing but never writes it.
        R.v["ith"] = ith

        R.summary.append(("I_peak1 mean (uA)", mean_of(i1_arr) * 1e6))
        R.summary.append(("Threshold I_th (uA)", ith * 1e6))
        R.summary.append(("Absolute refractory t_abs (s)", t_abs))
        R.summary.append(("Relative refractory t_rel 90% (s)", t_rel))
        if ratio_arr:
            R.summary.append(("Spike 2 recovery max (%)", max(ratio_arr)))

        finish_table(R, r, t, ei, n, I, [], "unused")
        R.cols.pop()
        R.data.pop()
        return R

    # ---------------------------------------------------------------- draw

    def draw(self, results: List[RunResult]) -> PanelGrid:
        a = Panel(
            title="Spike 2 Recovery Ratio vs delta t (interval)",
            xlabel="inter-spike interval delta t (s)",
            ylabel="Recovery: (I_peak2 / I_peak1) * 100 %",
            logx=True, row=0, col=0,
        )
        b = Panel(
            title="Peak Spike Currents: Pulse 1 vs Pulse 2",
            xlabel="inter-spike interval delta t (s)",
            ylabel="Peak |I| (uA)",
            logx=True, logy=True, row=0, col=1,
        )
        c = Panel(
            title="Second Spike Firing State (1 = fired, 0 = refractory)",
            xlabel="inter-spike interval delta t (s)",
            ylabel="Fired (I2 >= I_th)",
            logx=True, row=1, col=0,
        )
        d = Panel(title="Summary & Refractory Metrics", row=1, col=1)

        a.hlines.append((100.0, "#8793a8"))
        a.hlines.append((90.0, "#52d273"))

        ith = 0.0
        for i, R in enumerate(results):
            col = palette(i)
            a.series.append(
                Series(name=R.label, x=R.a["dt"], y=R.a["ratio"],
                       color=col, style="line_markers")
            )
            b.series.append(
                Series(name="I_peak1 (Pulse 1)", x=R.a["dt"],
                       y=[v * 1e6 for v in R.a["i1"]],
                       color=palette(0), style="line_markers")
            )
            b.series.append(
                Series(name="I_peak2 (Pulse 2)", x=R.a["dt"],
                       y=[v * 1e6 for v in R.a["i2"]],
                       color=palette(1), style="line_markers")
            )
            c.series.append(
                Series(name="Spike 2 Fired", x=R.a["dt"], y=R.a["fired"],
                       color=col, style="line_markers")
            )
            ith = R.v.get("ith", ith)

        if ith > 0.0:
            b.hlines.append((ith * 1e6, "#ffb703"))
        d.text = self.summary_lines(results)
        return PanelGrid(panels=[a, b, c, d], rows=2, cols=2)
