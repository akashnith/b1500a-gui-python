"""Synapse: spike-timing-dependent plasticity.

Port of ``class SynapseSTDP`` in ``pkg/src/experiments.cpp``.
A one-channel composite waveform approximates ``Vpre(t) - Vpost(t + dt)``;
each timing point starts with a reset pulse and is read before and after.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

from ..core.driver import Hardware, Results
from ..core.plan import GRID, Wave, parse_list, ticks
from ..core.plotting import Panel, PanelGrid, Series, palette
from .base import (
    NaN,
    Experiment,
    Params,
    RunResult,
    RunSpec,
    finish_table,
    read_pulse,
    require_finite,
)
from .registry import grouped, num_field, text_field

__all__ = ["SynapseSTDP"]


class SynapseSTDP(Experiment):
    """Sweep the pre/post spike offset and measure the weight change."""

    key = "stdp"

    @property
    def name(self) -> str:
        return "Synapse  spike-timing-dependent plasticity"

    def fields(self) -> List[dict]:
        return [
            *grouped(
                "Pre/post spike waveform",
                num_field("stdp_v_p", "Pre-spike amplitude (V)", "0.45"),
                num_field("stdp_v_post", "Post-spike amplitude ratio", "1.0",
                          "post amplitude is this multiplier times the pre amplitude"),
                num_field("stdp_w_p", "Positive spike width (s)", "1u"),
                num_field("stdp_tail_ratio", "Negative tail ratio", "0.2",
                          "tail magnitude / positive spike amplitude; must be 0-1"),
                num_field("stdp_tail_w", "Tail width (s)", "5u"),
                text_field("stdp_dt_list", "dt = t_post - t_pre (s)",
                           "lin(-100u,100u,21)",
                           "signed list or lin(start,stop,n); positive means post follows pre"),
            ),
            *grouped(
                "Reset and readback",
                num_field("stdp_v_reset", "Reset pulse amplitude (V)", "-0.7"),
                num_field("stdp_reset_w", "Reset pulse width (s)", "2u"),
                num_field("stdp_reset_gap", "Reset-to-read gap (s)", "2u"),
                num_field("stdp_pair_gap", "Read-to-spike-pair gap (s)", "2u"),
                num_field("stdp_v_read", "Read amplitude (V)", "0.1"),
                num_field("stdp_t_read", "Read width (s)", "1u"),
                num_field("stdp_read_delay", "Pair-to-read delay (s)", "10u"),
                num_field("stdp_t_rest", "Rest between timing points (s)", "1m"),
                num_field("stdp_pause", "Pause between runs (s)", "0.2"),
            ),
        ]

    def pause_between_runs(self, p: Params) -> float:
        return p.num("stdp_pause")

    # --------------------------------------------------------------- build

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        deltas = parse_list(p.str("stdp_dt_list"))
        require_finite(deltas, "STDP timing")
        if len(deltas) > 201:
            raise ValueError("STDP timing sweep is limited to 201 points")
        return [self.one(p, hw, deltas)]

    def one(self, p: Params, hw: Hardware, deltas: List[float]) -> RunSpec:
        r = RunSpec()
        r.plan.hint = "synapse"
        r.label = "STDP timing sweep"
        W = Wave(r.plan)

        edge = ticks(hw.edge)
        reset_width = ticks(p.num("stdp_reset_w"))
        pulse_width = ticks(p.num("stdp_w_p"))
        tail_width = ticks(p.num("stdp_tail_w"))
        read_width = ticks(p.num("stdp_t_read"))
        vr = p.num("stdp_v_read")
        vp = p.num("stdp_v_p")
        post_v = p.num("stdp_v_post")
        tail = p.num("stdp_tail_ratio")
        reset_v = p.num("stdp_v_reset")

        if vp == 0.0 or post_v == 0.0:
            raise ValueError("pre- and post-spike amplitudes must be non-zero")
        if post_v < 0.0:
            raise ValueError("post-spike amplitude ratio must be positive")
        if tail < 0.0 or tail > 1.0:
            raise ValueError("tail ratio must be between 0 and 1")

        r.v["vr"] = vr
        r.v["vp"] = vp

        for i, dt in enumerate(deltas):
            if abs(dt) > 1.0:
                raise ValueError("STDP timing offsets are limited to +/-1 s")
            dt_ticks = int(round(dt / GRID))

            W.to(reset_v, edge)
            W.hold(reset_width)
            W.to(0.0, edge)
            W.hold(ticks(p.num("stdp_reset_gap")))
            read_pulse(r, W, hw, vr, read_width, f"before{i}", 0, i, 0.0, dt)
            W.hold(ticks(p.num("stdp_pair_gap")))

            base = W.now() + max(edge * 2, abs(dt_ticks) + edge)
            pre_start = base + max(0, -dt_ticks)
            post_start = base + max(0, dt_ticks)

            changes: Dict[int, float] = defaultdict(float)

            def add_spike(start: int, polarity: float) -> None:
                changes[start] += polarity * vp
                changes[start + pulse_width] -= polarity * vp
                changes[start + pulse_width] -= polarity * vp * tail
                changes[start + pulse_width + tail_width] += polarity * vp * tail

            add_spike(pre_start, 1.0)
            add_spike(post_start, -post_v)

            level = 0.0
            keys = sorted(changes)
            for j, at in enumerate(keys):
                if at > W.now():
                    W.hold(at - W.now())
                level += changes[at]
                next_at = keys[j + 1] if j + 1 < len(keys) else at + edge
                span = max(1, next_at - at)
                ramp = min(edge, span)
                W.to(level, ramp)
                if span > ramp:
                    W.hold(span - ramp)

            W.hold(ticks(p.num("stdp_read_delay")))
            read_pulse(r, W, hw, vr, read_width, f"after{i}", 1, i, vp, dt)
            W.hold(ticks(p.num("stdp_t_rest")))
        return r

    # ------------------------------------------------------------- analyze

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        current = res.current(hw)
        t, ei = r.plan.samples()
        n = min(len(current), len(t))
        R = RunResult(label=r.label)

        before: Dict[int, float] = {}
        after: Dict[int, float] = {}
        dt_map: Dict[int, float] = {}
        I = [0.0] * n

        for i in range(n):
            e = int(ei[i])
            I[i] = abs(current[i])
            v_read = abs(r.ev_measure_v[e])
            conductance = I[i] / v_read if v_read > 0.0 else NaN
            idx = int(r.ev_idx[e])
            dt_map[idx] = r.ev_width[e]
            if r.ev_phase[e] == 0:
                before[idx] = conductance
            elif r.ev_phase[e] == 1:
                after[idx] = conductance

        delta_t: List[float] = []
        delta_w: List[float] = []
        g_before: List[float] = []
        g_after: List[float] = []
        for idx in sorted(dt_map):
            if idx not in before or idx not in after:
                continue
            delta_t.append(dt_map[idx])
            g_before.append(before[idx])
            g_after.append(after[idx])
            delta_w.append(
                100.0 * (after[idx] - before[idx]) / before[idx]
                if before[idx] > 0.0
                else NaN
            )

        R.a["dt"] = delta_t
        R.a["dw"] = delta_w
        R.a["before"] = g_before
        R.a["after"] = g_after

        finite_dw = [v for v in delta_w if math.isfinite(v)]
        if finite_dw:
            R.summary.append(("maximum potentiation (%)", max(finite_dw)))
            R.summary.append(("maximum depression (%)", min(finite_dw)))

        finish_table(R, r, t, ei, n, I, [], "unused")
        R.cols.pop()
        R.data.pop()
        G = [NaN] * n
        for i in range(n):
            v_read = abs(r.ev_measure_v[int(ei[i])])
            if v_read > 0.0:
                G[i] = I[i] / v_read
        R.cols.append("G_S")
        R.data.append(G)
        return R

    # ---------------------------------------------------------------- draw

    def draw(self, results: List[RunResult]) -> PanelGrid:
        timing = Panel(
            title="STDP learning window",
            xlabel="delta t = t_post - t_pre (s)",
            ylabel="weight change delta W (%)",
            row=0, col=0,
        )
        conductance = Panel(
            title="Conductance before and after spike pair",
            xlabel="delta t (s)",
            ylabel="G (uS)",
            row=0, col=1,
        )
        summary = Panel(title="STDP metrics", row=1, col=0)
        convention = Panel(
            title="Timing convention",
            text=[
                "Positive delta t: post-spike follows pre-spike.",
                "Negative delta t: post-spike precedes pre-spike.",
                "Each timing point begins with the configured reset pulse.",
            ],
            row=1, col=1,
        )

        for i, R in enumerate(results):
            timing.series.append(
                Series(name=R.label, x=R.a["dt"], y=R.a["dw"],
                       color=palette(i), style="line_markers")
            )
            conductance.series.append(
                Series(name="before", x=R.a["dt"],
                       y=[g * 1e6 for g in R.a["before"]],
                       color=palette(0), style="line_markers")
            )
            conductance.series.append(
                Series(name="after", x=R.a["dt"],
                       y=[g * 1e6 for g in R.a["after"]],
                       color=palette(1), style="line_markers")
            )

        timing.hlines.append((0.0, "#8793a8"))
        summary.text = self.summary_lines(results)
        return PanelGrid(
            panels=[timing, conductance, summary, convention], rows=2, cols=2
        )
