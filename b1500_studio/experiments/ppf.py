"""Synapse: paired-pulse facilitation / depression (PPF / PPD).

Port of ``class SynapsePPF`` in ``pkg/src/experiments.cpp``.
Measures the baseline-subtracted EPSC amplitude ratio ``(A2/A1) * 100 %``
against the inter-pulse interval.
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
    _fit_beta,
    finish_table,
    mean_of,
    read_pulse,
    require_finite,
)
from .registry import grouped, num_field, text_field

__all__ = ["SynapsePPF"]


class SynapsePPF(Experiment):
    """Two identical pulses separated by dt, swept over dt."""

    key = "ppf"

    @property
    def name(self) -> str:
        return "Synapse  PPF / PPD (paired-pulse plasticity)"

    def fields(self) -> List[dict]:
        return [
            *grouped(
                "Paired-pulse stimulus (PPF / PPD)",
                num_field("ppf_v_p", "Pulse amplitude (V)", "0.7",
                          "excitatory pulse amplitude V_p"),
                num_field("ppf_w_p", "Pulse width (s)", "1u",
                          "width of pulse 1 and pulse 2"),
                text_field("ppf_dt_list", "Inter-pulse interval(s) dt (s)",
                           "log(10u,100m,8)",
                           "list or log(start,stop,n): swept delta t"),
                num_field("ppf_pause", "Pause between runs (s)", "0.5"),
            ),
            *grouped(
                "Read & timing",
                num_field("ppf_v_read", "Read amplitude (V)", "0.1",
                          "non-disturbing read voltage (< 0.2 V)"),
                num_field("ppf_t_read", "Read width (s)", "1u"),
                num_field("ppf_t_gap", "Pulse -> read gap (s)", "200n"),
                num_field("ppf_t_rest", "Rest between pairs (s)", "1.0",
                          "recovery rest >= 10 tau to restore baseline"),
            ),
        ]

    def pause_between_runs(self, p: Params) -> float:
        return p.num("ppf_pause")

    # --------------------------------------------------------------- build

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        dts = parse_list(p.str("ppf_dt_list"))
        require_finite(dts, "inter-pulse interval")
        return [self.one(p, hw, dts)]

    def one(self, p: Params, hw: Hardware, dts: List[float]) -> RunSpec:
        r = RunSpec()
        r.plan.hint = "synapse"
        r.label = "PPF / PPD sweep"
        W = Wave(r.plan)

        e = ticks(hw.edge)
        tr = ticks(p.num("ppf_t_read"))
        gap = ticks(p.num("ppf_t_gap"))
        rest = ticks(p.num("ppf_t_rest"))
        vp = p.num("ppf_v_p")
        wp = ticks(p.num("ppf_w_p"))
        vr = p.num("ppf_v_read")
        r.v["vr"] = vr
        r.v["vp"] = vp
        r.v["wp"] = secs(wp)

        for i, dt in enumerate(dts):
            dt_ticks = ticks(dt)
            # Baseline read.
            read_pulse(r, W, hw, vr, tr, f"r0_{i}", 0, i, 0.0, dt)
            W.hold(rest)

            # Pulse 1 then read 1.
            W.to(vp, e)
            W.hold(wp)
            W.to(0.0, e)
            W.hold(gap)
            read_pulse(r, W, hw, vr, tr, f"p1_{i}", 1, i, vp, dt)

            # Inter-pulse interval.
            W.hold(dt_ticks)

            # Pulse 2 then read 2.
            W.to(vp, e)
            W.hold(wp)
            W.to(0.0, e)
            W.hold(gap)
            read_pulse(r, W, hw, vr, tr, f"p2_{i}", 2, i, vp, dt)

            # Recovery rest between pairs.
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

        g0_map: Dict[int, float] = {}
        g1_map: Dict[int, float] = {}
        g2_map: Dict[int, float] = {}
        dt_map: Dict[int, float] = {}
        I = [0.0] * n
        G = [NaN] * n

        for i in range(n):
            e = int(ei[i])
            I[i] = abs(cur[i])
            measure_v = abs(r.ev_measure_v[e])
            G[i] = I[i] / measure_v if measure_v > 0.0 else NaN
            idx = int(r.ev_idx[e])
            ph = r.ev_phase[e]
            dt_map[idx] = r.ev_width[e]
            if ph == 0:
                g0_map[idx] = G[i]
            elif ph == 1:
                g1_map[idx] = G[i]
            elif ph == 2:
                g2_map[idx] = G[i]

        dt_arr: List[float] = []
        ppf_arr: List[float] = []
        g0_arr: List[float] = []
        a1_arr: List[float] = []
        a2_arr: List[float] = []
        max_ppf = NaN

        for idx in sorted(dt_map):
            dt = dt_map[idx]
            if idx in g0_map and idx in g1_map and idx in g2_map:
                g0, g1, g2 = g0_map[idx], g1_map[idx], g2_map[idx]
                a1, a2 = g1 - g0, g2 - g0
                ppf_val = (a2 / a1) * 100.0 if a1 > 0.0 else NaN
                dt_arr.append(dt)
                ppf_arr.append(ppf_val)
                g0_arr.append(g0)
                a1_arr.append(a1)
                a2_arr.append(a2)
                if math.isfinite(ppf_val):
                    max_ppf = ppf_val if not math.isfinite(max_ppf) else max(max_ppf, ppf_val)

        R.a["dt"] = dt_arr
        R.a["ppf"] = ppf_arr
        R.a["g0"] = g0_arr
        R.a["g1"] = a1_arr
        R.a["g2"] = a2_arr

        fit_dt_in: List[float] = []
        fit_excess: List[float] = []
        for i, value in enumerate(ppf_arr):
            if math.isfinite(value):
                fit_dt_in.append(dt_arr[i])
                fit_excess.append(max(value - 100.0, 0.0))
        f_ppf = _fit_beta(fit_dt_in, fit_excess, 1.0)

        R.summary.append(("PPF max (%)", max_ppf))
        R.summary.append(("tau PPF (s)", f_ppf.tau if f_ppf.ok else NaN))
        R.summary.append(("Facilitation gain C (%)", f_ppf.A if f_ppf.ok else NaN))
        R.summary.append(("G_baseline mean (uS)", mean_of(g0_arr) * 1e6))
        R.summary.append(("EPSC A1 mean (uS)", mean_of(a1_arr) * 1e6))
        R.summary.append(("EPSC A2 mean (uS)", mean_of(a2_arr) * 1e6))

        fit_dt: List[float] = []
        fit_ppf: List[float] = []
        if f_ppf.ok and dt_arr and dt_arr[0] > 0 and dt_arr[-1] > dt_arr[0]:
            for i in range(101):
                tv = dt_arr[0] * (dt_arr[-1] / dt_arr[0]) ** (i / 100.0)
                fit_dt.append(tv)
                fit_ppf.append(
                    100.0 + f_ppf.ginf + f_ppf.A * math.exp(-tv / f_ppf.tau)
                )
        R.a["fit_dt"] = fit_dt
        R.a["fit_ppf"] = fit_ppf

        finish_table(R, r, t, ei, n, I, [], "unused")
        R.cols.pop()
        R.data.pop()
        R.cols.append("G_S")
        R.data.append(G)
        return R

    # ---------------------------------------------------------------- draw

    def draw(self, results: List[RunResult]) -> PanelGrid:
        a = Panel(
            title="PPF / PPD Index vs inter-pulse interval",
            xlabel="inter-pulse interval delta t (s)",
            ylabel="PPF Index: (A2 / A1) * 100 %",
            logx=True, row=0, col=0,
        )
        b = Panel(
            title="Baseline-corrected EPSC amplitude",
            xlabel="inter-pulse interval delta t (s)",
            ylabel="G (uS)",
            logx=True, row=0, col=1,
        )
        c = Panel(
            title="Baseline Conductance G0 stability",
            xlabel="inter-pulse interval delta t (s)",
            ylabel="G0 (uS)",
            logx=True, row=1, col=0,
        )
        d = Panel(title="Summary & Figures of Merit", row=1, col=1)

        a.hlines.append((100.0, "#8793a8"))

        for i, R in enumerate(results):
            col = palette(i)
            a.series.append(
                Series(name=R.label, x=R.a["dt"], y=R.a["ppf"],
                       color=col, style="line_markers")
            )
            if R.a.get("fit_dt"):
                a.series.append(
                    Series(name="fit (exp)", x=R.a["fit_dt"], y=R.a["fit_ppf"],
                           color=col, dashed=True)
                )
            b.series.append(
                Series(name="A1 (pulse 1)", x=R.a["dt"],
                       y=[g * 1e6 for g in R.a["g1"]],
                       color=palette(0), style="line_markers")
            )
            b.series.append(
                Series(name="A2 (pulse 2)", x=R.a["dt"],
                       y=[g * 1e6 for g in R.a["g2"]],
                       color=palette(1), style="line_markers")
            )
            c.series.append(
                Series(name="G0 (baseline)", x=R.a["dt"],
                       y=[g * 1e6 for g in R.a["g0"]],
                       color=palette(2), style="line_markers")
            )

        d.text = self.summary_lines(results)
        return PanelGrid(panels=[a, b, c, d], rows=2, cols=2)
