"""Neuron: transient turn-off and relaxation (LRS -> HRS).

Port of ``class Relaxation`` in ``pkg/src/experiments.cpp``.
Fits ``G(t) = Ginf + A exp(-(t/tau)^beta)`` as both a single exponential and
a stretched (KWW) exponential.
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
    add_event,
    cross_time,
    eng,
    finish_table,
    _fit_beta,
    fit_kww,
    mean_of,
    read_pulse,
    require_finite,
    t_unit,
)
from .registry import check_field, combo_field, grouped, num_field, text_field

__all__ = ["Relaxation"]


class Relaxation(Experiment):
    """One switching pulse, then a log-spaced read train watching it decay."""

    key = "relaxation"

    @property
    def name(self) -> str:
        return "Neuron  relaxation (LRS -> HRS)"

    def fields(self) -> List[dict]:
        return [
            *grouped(
                "Switching pulse",
                text_field(
                    "rel_v_sw",
                    "Amplitude(s) (V)",
                    "1.0",
                    "list or lin(0.7,1,4): one run per amplitude",
                ),
                num_field("rel_w_sw", "Width (s)", "1u"),
                check_field("rel_capture", "Capture current during the pulse", True),
                num_field("rel_samples", "Samples across the pulse", "20"),
            ),
            *grouped(
                "Read pulses (LRS -> HRS monitor)",
                num_field("rel_v_read", "Read amplitude (V)", "0.1"),
                num_field("rel_t_read", "Read width (s)", "1u"),
                num_field("rel_n_pre", "# baseline reads (HRS)", "3"),
                num_field("rel_pre_gap", "Baseline spacing (s)", "5u"),
            ),
            *grouped(
                "Relaxation sampling",
                num_field("rel_t_first", "First read after pulse (s)", "2u"),
                num_field("rel_t_last", "Last read after pulse (s)", "1m"),
                num_field("rel_n_reads", "# reads", "40"),
                combo_field("rel_spacing", "Spacing", ["Log", "Linear"], 0),
                num_field("rel_pause", "Pause between runs (s)", "0.5"),
            ),
        ]

    def pause_between_runs(self, p: Params) -> float:
        return p.num("rel_pause")

    # --------------------------------------------------------------- build

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        values = parse_list(p.str("rel_v_sw"))
        require_finite(values, "switching amplitude")
        return [self.one(p, hw, v) for v in values]

    def one(self, p: Params, hw: Hardware, vsw: float) -> RunSpec:
        r = RunSpec()
        r.plan.hint = "neuron"
        r.label = "Vsw = " + eng(vsw, "V", 3)
        W = Wave(r.plan)

        e = ticks(hw.edge)
        tr = ticks(p.num("rel_t_read"))
        vr = p.num("rel_v_read")
        r.v["vr"] = vr

        for i in range(p.integer("rel_n_pre")):
            read_pulse(r, W, hw, vr, tr, f"pre{i}", 0, i, 0.0, 0.0)
            W.hold(ticks(p.num("rel_pre_gap")))

        W.to(vsw, e)
        fs = W.now()
        W.hold(ticks(p.num("rel_w_sw")))
        fe = W.now()
        t_end = W.to(0.0, e)

        if p.flag("rel_capture"):
            S = max(1, min(200, p.integer("rel_samples")))
            iv = max(1, (fe - fs) // S)
            add_event(
                r, "sw", fs, (fe - fs) // iv, iv, iv, 2, 0,
                vsw, p.num("rel_w_sw"), vsw - hw.sink_bias,
            )

        n = max(1, p.integer("rel_n_reads"))
        t0, t1 = p.num("rel_t_first"), p.num("rel_t_last")
        if t0 <= 0 or t1 <= t0:
            raise ValueError("need 0 < first read < last read")

        # Read start times in ticks after the switching pulse. Consecutive
        # reads are pushed apart by one read pulse (2 edges + width) so dense
        # log spacing at early times stays physically realisable.
        tt: List[int] = []
        log_spacing = p.integer("rel_spacing") == 0
        for k in range(n):
            f = 0.0 if n == 1 else k / (n - 1)
            raw = t0 * (t1 / t0) ** f if log_spacing else t0 + (t1 - t0) * f
            t = ticks(raw)
            if k > 0:
                t = max(t, tt[-1] + 2 * e + tr)
            tt.append(t)

        for k in range(n):
            target = t_end + tt[k]
            if target > W.now():
                W.hold(target - W.now())
            read_pulse(r, W, hw, vr, tr, f"rd{k}", 1, k, vsw, secs(tt[k]))

        W.hold(ticks(1e-6))
        r.v["tEnd"] = secs(t_end)
        r.v["fs"] = secs(fs)
        r.v["vsw"] = vsw
        return r

    # ------------------------------------------------------------- analyze

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        cur = res.current(hw)
        t, ei = r.plan.samples()
        n = min(len(cur), len(t))
        R = RunResult(label=r.label)
        if n < len(t):
            R.notes.append("partial data")

        I = [0.0] * n
        pre_g: List[float] = []
        rt: List[float] = []
        rg: List[float] = []
        swt: List[float] = []
        swi: List[float] = []

        for i in range(n):
            e = int(ei[i])
            I[i] = abs(cur[i])
            ph = r.ev_phase[e]
            measure_v = abs(r.ev_measure_v[e])
            conductance = I[i] / measure_v if measure_v > 0.0 else NaN
            if ph == 0:
                pre_g.append(conductance)
            elif ph == 1:
                rt.append(t[i] - r.v["tEnd"])
                rg.append(conductance)
            elif ph == 2:
                swt.append(t[i] - r.v["fs"])
                swi.append(I[i])

        ghrs = (rg[-1] if rg else NaN) if not pre_g else mean_of(pre_g)
        R.a["rt"], R.a["rg"] = rt, rg
        R.a["swt"], R.a["swi"] = swt, swi
        R.v["ghrs"] = ghrs

        g0 = rg[0] if rg else NaN
        R.summary.append(("G_HRS before (S)", ghrs))
        R.summary.append(("G first read (S)", g0))
        R.summary.append(("G first / G_HRS", g0 / ghrs if ghrs else NaN))

        fe_fit = _fit_beta(rt, rg, 1.0)
        fk_fit = fit_kww(rt, rg)
        R.summary.append(("tau exp (s)", fe_fit.tau if fe_fit.ok else NaN))
        R.summary.append(("tau KWW (s)", fk_fit.tau if fk_fit.ok else NaN))
        R.summary.append(("beta KWW", fk_fit.beta if fk_fit.ok else NaN))
        R.summary.append(("G_inf exp fit (S)", fe_fit.ginf if fe_fit.ok else NaN))

        dg = [g - ghrs for g in rg]
        d0 = dg[0] if dg else NaN
        R.summary.append(("t(1/e) (s)", cross_time(rt, dg, d0 / math.e)))
        R.summary.append(("t(90% recovered) (s)", cross_time(rt, dg, d0 * 0.1)))

        if swi:
            pk = max(swi)
            t50 = NaN
            for i in range(len(swi)):
                if swi[i] >= 0.5 * pk:
                    t50 = swt[i]
                    break
            R.summary.append(("I peak in pulse (A)", pk))
            R.summary.append(("delay to 50% I (s)", t50))

        # Smooth fit curves for plotting.
        for k, fit in enumerate((fe_fit, fk_fit)):
            ft: List[float] = []
            fg: List[float] = []
            if fit.ok and rt and rt[0] > 0 and rt[-1] > 0:
                for i in range(121):
                    tv = rt[0] * (rt[-1] / rt[0]) ** (i / 120.0)
                    ft.append(tv)
                    try:
                        fg.append(
                            fit.ginf + fit.A * math.exp(-((tv / fit.tau) ** fit.beta))
                        )
                    except (OverflowError, ValueError):
                        fg.append(fit.ginf)
            R.a["fe_t" if k == 0 else "fk_t"] = ft
            R.a["fe_g" if k == 0 else "fk_g"] = fg

        R.v["ginf"] = fe_fit.ginf if fe_fit.ok else NaN

        finish_table(R, r, t, ei, n, I, [], "unused")
        R.cols.pop()
        R.data.pop()
        G = []
        for i in range(n):
            measure_v = abs(r.ev_measure_v[int(ei[i])])
            G.append(I[i] / measure_v if measure_v > 0.0 else NaN)
        R.cols.append("G_S")
        R.data.append(G)
        return R

    # ---------------------------------------------------------------- draw

    def draw(self, results: List[RunResult]) -> PanelGrid:
        a = Panel(
            title="Relaxation after switching pulse (read pulses)",
            xlabel="time after pulse (s)",
            ylabel="G (uS)",
            logx=True,
            logy=True,
            row=0,
            col=0,
        )
        b = Panel(
            title="Normalised decay",
            xlabel="time after pulse",
            ylabel="(G - G_HRS)/(G0 - G_HRS)",
            logy=True,
            row=0,
            col=1,
        )
        c = Panel(
            title="Current during switching pulse",
            xlabel="t into pulse",
            ylabel="|I| (uA)",
            row=1,
            col=0,
        )
        d = Panel(title="Summary", row=1, col=1)

        tmax = swmax = 0.0
        for R in results:
            if R.a.get("rt"):
                tmax = max(tmax, R.a["rt"][-1])
            if R.a.get("swt"):
                swmax = max(swmax, R.a["swt"][-1])
        tu_f, tu_n = t_unit(tmax if tmax > 0 else 1e-3)
        su_f, su_n = t_unit(swmax if swmax > 0 else 1e-6)
        b.xlabel += f" ({tu_n})"
        c.xlabel += f" ({su_n})"

        for i, R in enumerate(results):
            col = palette(i)
            a.series.append(
                Series(
                    name=R.label,
                    x=R.a["rt"],
                    y=[g * 1e6 for g in R.a["rg"]],
                    color=col,
                    style="line_markers",
                )
            )
            if R.a.get("fe_t"):
                a.series.append(
                    Series(
                        x=R.a["fe_t"],
                        y=[g * 1e6 for g in R.a["fe_g"]],
                        color=col,
                        dashed=True,
                    )
                )
            if R.v.get("ghrs", 0) and R.v["ghrs"] > 0:
                a.hlines.append((R.v["ghrs"] * 1e6, "#52d273"))

            rg, ghrs = R.a["rg"], R.v["ghrs"]
            if rg:
                d0 = rg[0] - ghrs
                if d0:
                    b.series.append(
                        Series(
                            x=[t / tu_f for t in R.a["rt"]],
                            y=[(g - ghrs) / d0 for g in rg],
                            color=col,
                            style="line_markers",
                        )
                    )
            if R.a.get("swt"):
                c.series.append(
                    Series(
                        name=R.label,
                        x=[t / su_f for t in R.a["swt"]],
                        y=[v * 1e6 for v in R.a["swi"]],
                        color=col,
                        style="line_markers",
                    )
                )

        b.hlines.append((math.exp(-1.0), "#ffb703"))
        d.text = self.summary_lines(results)
        return PanelGrid(panels=[a, b, c, d], rows=2, cols=2)
