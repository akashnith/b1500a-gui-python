"""Synapse: LTP / LTD pulse trains plus retention.

Port of ``class Synapse`` in ``pkg/src/experiments.cpp``.
"""

from __future__ import annotations

import math
from typing import List

from ..core.driver import Hardware, Results
from ..core.plan import GRID, Wave, parse_list, secs, ticks
from ..core.plotting import Panel, PanelGrid, Series, palette
from .base import (
    NaN,
    Experiment,
    Params,
    RunResult,
    RunSpec,
    add_event,
    eng,
    finish_table,
    linfit,
    read_pulse,
    require_finite,
)
from .registry import check_field, combo_field, grouped, num_field, text_field

__all__ = ["Synapse", "SCHEMES"]

SCHEMES = ("Amplitude increment", "Width increment", "Constant pulses")


class Synapse(Experiment):
    """Potentiation / depression pulse trains with optional retention reads."""

    key = "synapse"

    @property
    def name(self) -> str:
        return "Synapse  LTP / LTD + retention"

    def fields(self) -> List[dict]:
        return [
            *grouped(
                "Stimulus",
                combo_field(
                    "syn_scheme",
                    "Stimulus",
                    [*SCHEMES, "All three (sequential)"],
                    0,
                    "how successive pulses change",
                ),
                num_field("syn_pause", "Pause between runs (s)", "1"),
                check_field(
                    "syn_geometric",
                    "Width grows geometrically (dw = ratio)",
                    False,
                ),
            ),
            *grouped(
                "Potentiation",
                num_field("syn_n_pot", "# pulses", "30"),
                num_field("syn_pot_v0", "Start amplitude V0 (V)", "0.7"),
                num_field("syn_pot_dv", "Amplitude step dV (V)", "10m"),
                num_field("syn_pot_w0", "Start width w0 (s)", "1u"),
                num_field("syn_pot_dw", "Width step dw (s | ratio)", "200n"),
            ),
            *grouped(
                "Depression",
                num_field("syn_n_dep", "# pulses", "30"),
                num_field("syn_dep_v0", "Start amplitude V0 (V)", "-0.7"),
                num_field("syn_dep_dv", "Amplitude step dV (V)", "-10m"),
                num_field("syn_dep_w0", "Start width w0 (s)", "1u"),
                num_field("syn_dep_dw", "Width step dw (s | ratio)", "200n"),
            ),
            *grouped(
                "Read",
                num_field(
                    "syn_v_read",
                    "Read amplitude (V)",
                    "0.1",
                    "tutorial recommends 0.05 - 0.2 V to prevent state disturb",
                ),
                num_field("syn_t_read", "Read width (s)", "1u"),
                num_field("syn_t_gap", "Pulse -> read gap (s)", "200n"),
                num_field("syn_t_rest", "Rest after read (s)", "1u"),
                check_field("syn_pre_read", "Read before the first pulse", True),
            ),
            *grouped(
                "Retention",
                combo_field(
                    "syn_ret_mode",
                    "Measure retention",
                    ["None", "After LTP", "After LTD", "After LTP and LTD"],
                    3,
                ),
                num_field("syn_ret_t0", "First read at (s)", "1m"),
                num_field("syn_ret_t1", "Last read at (s)", "10"),
                num_field("syn_ret_n", "# reads (log spaced)", "12"),
            ),
        ]

    def pause_between_runs(self, p: Params) -> float:
        return p.num("syn_pause")

    # --------------------------------------------------------------- build

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        sch = p.integer("syn_scheme")
        schemes = [0, 1, 2] if sch == 3 else [sch]
        return [self.one(p, hw, s) for s in schemes]

    def one(self, p: Params, hw: Hardware, sch: int) -> RunSpec:
        r = RunSpec()
        r.plan.hint = "synapse"
        r.label = SCHEMES[sch]
        W = Wave(r.plan)

        e = ticks(hw.edge)
        tr = ticks(p.num("syn_t_read"))
        gap = ticks(p.num("syn_t_gap"))
        rest = ticks(p.num("syn_t_rest"))
        vr = p.num("syn_v_read")
        n_pot = p.integer("syn_n_pot")
        n_dep = p.integer("syn_n_dep")
        geo = p.flag("syn_geometric")
        ret = p.integer("syn_ret_mode")  # 0 none, 1 LTP, 2 LTD, 3 both

        if n_pot < 0 or n_dep < 0 or n_pot + n_dep < 1:
            raise ValueError("need at least one pulse")
        r.v["vr"] = vr

        def vals(pot: bool, k: int):
            pre = "syn_pot_" if pot else "syn_dep_"
            v0 = p.num(pre + "v0")
            dv = p.num(pre + "dv")
            w0 = p.num(pre + "w0")
            dw = p.num(pre + "dw")
            V = v0 + (k * dv if sch == 0 else 0.0)
            w = w0
            if sch == 1:
                w = w0 * (dw ** k) if geo else w0 + k * dw
            if w < GRID:
                raise ValueError("pulse width falls below 10 ns")
            return V, w

        def block(pot: bool, n: int, phase: int) -> int:
            pulse_end = 0
            tag = "pot" if pot else "dep"
            for k in range(n):
                V, w = vals(pot, k)
                W.to(V, e)
                W.hold(ticks(w))
                pulse_end = W.to(0.0, e)
                W.hold(gap)
                read_pulse(r, W, hw, vr, tr, f"{tag}{k}", phase, k, V, w)
                W.hold(rest)
            return pulse_end

        def retention(phase: int, pulse_end: int, tag: str) -> None:
            ts = parse_list(
                f"log({p.str('syn_ret_t0')},{p.str('syn_ret_t1')},"
                f"{p.integer('syn_ret_n')})"
            )
            require_finite(ts, "retention time")
            for i, t_ret in enumerate(ts):
                target = pulse_end + ticks(t_ret)
                if target < W.now():
                    raise ValueError(
                        f"retention time {eng(t_ret, 's')} is too short: the "
                        "first retention read must start at least "
                        f"{eng(secs(W.now() - pulse_end), 's')} after the last "
                        "pulse (gap + read + rest)"
                    )
                if target > W.now():
                    W.hold(target - W.now())
                read_pulse(r, W, hw, vr, tr, f"{tag}{i}", phase, i, 0.0, t_ret)
            W.hold(rest)

        if p.flag("syn_pre_read"):
            read_pulse(r, W, hw, vr, tr, "r0", 0, 0, 0.0, 0.0)
            W.hold(rest)

        p_end_pot = block(True, n_pot, 1)
        if n_pot > 0 and (ret & 1):
            retention(2, p_end_pot, "retP")
        p_end_dep = block(False, n_dep, 3)
        if n_dep > 0 and (ret & 2):
            retention(4, p_end_dep, "retD")

        r.v["pEndPot"] = secs(p_end_pot)
        r.v["pEndDep"] = secs(p_end_dep)
        r.v["nPot"] = float(n_pot)
        r.v["pre"] = 1.0 if p.flag("syn_pre_read") else 0.0
        return r

    # ------------------------------------------------------------- analyze

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        cur = res.current(hw)
        t, ei = r.plan.samples()
        n = min(len(cur), len(t))

        R = RunResult(label=r.label)
        if n < len(t):
            R.notes.append(f"partial data: {n}/{len(t)} points")

        I = [0.0] * n
        G = [NaN] * n
        tsince = [NaN] * n
        px: List[float] = []
        pg: List[float] = []
        pot_x: List[float] = []
        pot_g: List[float] = []
        dep_x: List[float] = []
        dep_g: List[float] = []
        rP_t: List[float] = []
        rP_g: List[float] = []
        rD_t: List[float] = []
        rD_g: List[float] = []
        counter = 0

        for i in range(n):
            e = int(ei[i])
            I[i] = abs(cur[i])
            ph = r.ev_phase[e]
            measure_v = abs(r.ev_measure_v[e])
            G[i] = I[i] / measure_v if measure_v > 0.0 else NaN
            if ph in (0, 1, 3):
                px.append(float(counter))
                pg.append(G[i])
                if ph == 1:
                    pot_x.append(float(counter))
                    pot_g.append(G[i])
                if ph == 3:
                    dep_x.append(float(counter))
                    dep_g.append(G[i])
                counter += 1
            elif ph == 2:
                tsince[i] = t[i] - r.v["pEndPot"]
                rP_t.append(tsince[i])
                rP_g.append(G[i])
            elif ph == 4:
                tsince[i] = t[i] - r.v["pEndDep"]
                rD_t.append(tsince[i])
                rD_g.append(G[i])

        R.v["boundary"] = r.v["pre"] + r.v["nPot"] - 0.5
        R.a["px"] = px
        R.a["pg"] = pg
        step = [NaN] * len(px)
        for i in range(1, len(pg)):
            step[i] = pg[i] - pg[i - 1]
        R.a["step"] = step
        R.a["rP_t"] = rP_t
        R.a["rP_g"] = rP_g
        R.a["rD_t"] = rD_t
        R.a["rD_g"] = rD_g

        if pg:
            gmin, gmax = min(pg), max(pg)
            R.summary.append(("G_min (S)", gmin))
            R.summary.append(("G_max (S)", gmax))
            R.summary.append(("Gmax/Gmin", gmax / max(gmin, 1e-30)))

        lp = linfit(pot_x, pot_g)
        ld = linfit(dep_x, dep_g)
        R.summary.append(("LTP slope (S/pulse)", lp.slope))
        R.summary.append(("LTP linearity R2", lp.r2))
        R.summary.append(("LTD slope (S/pulse)", ld.slope))
        R.summary.append(("LTD linearity R2", ld.r2))

        def ret_stats(tag: str, rt: List[float], rg: List[float]) -> None:
            if len(rg) < 2:
                return
            R.summary.append((f"ret {tag} G(t0) (S)", rg[0]))
            R.summary.append(
                (f"ret {tag} G(end)/G(t0)", rg[-1] / max(rg[0], 1e-30))
            )
            lx, ly = [], []
            for i in range(len(rt)):
                if rt[i] > 0 and rg[i] > 0:
                    lx.append(math.log10(rt[i]))
                    ly.append(math.log10(rg[i]))
            R.summary.append((f"ret {tag} power-law exp.", linfit(lx, ly).slope))

        ret_stats("LTP", rP_t, rP_g)
        ret_stats("LTD", rD_t, rD_g)

        finish_table(R, r, t, ei, n, I, tsince, "t_since_last_pulse_s")
        R.cols.append("G_S")
        R.data.append(G)
        return R

    # ---------------------------------------------------------------- draw

    def draw(self, results: List[RunResult]) -> PanelGrid:
        a = Panel(
            title="Conductance vs pulse number",
            xlabel="read #  (0 = pre-read)",
            ylabel="G (uS)",
            row=0,
            col=0,
        )
        b = Panel(
            title="Update step per pulse",
            xlabel="read #",
            ylabel="dG (uS)",
            row=0,
            col=1,
        )
        c = Panel(
            title="Retention",
            xlabel="time since last pulse (s)",
            ylabel="G / G(first read)",
            logx=True,
            row=1,
            col=0,
        )
        d = Panel(title="Summary", row=1, col=1)

        for i, R in enumerate(results):
            col = palette(i)
            a.series.append(
                Series(
                    name=R.label,
                    x=R.a["px"],
                    y=[g * 1e6 for g in R.a["pg"]],
                    color=col,
                    style="line_markers",
                )
            )
            a.vlines.append((R.v["boundary"], "#4a5878"))
            b.series.append(
                Series(
                    x=R.a["px"],
                    y=[g * 1e6 for g in R.a["step"]],
                    color=col,
                    style="line_markers",
                )
            )
            for k, (tk, gk, suffix) in enumerate(
                (
                    ("rP_t", "rP_g", " after LTP"),
                    ("rD_t", "rD_g", " after LTD"),
                )
            ):
                rt, rg = R.a[tk], R.a[gk]
                if not rg:
                    continue
                c.series.append(
                    Series(
                        name=R.label + suffix,
                        x=rt,
                        y=[g / rg[0] for g in rg],
                        color=col,
                        style="line_markers",
                        dashed=(k == 1),
                    )
                )

        c.hlines.append((1.0, "#4a5878"))
        d.text = self.summary_lines(results)
        return PanelGrid(panels=[a, b, c, d], rows=2, cols=2)
