"""Waveform preview: the force voltage with idle 0 V gaps compressed.

Port of ``drawPreview()`` in ``pkg/src/experiments.cpp``.  Long rests between
pulses would otherwise squash the pulses themselves into invisible spikes, so
any idle run of 0 V longer than ``cmax`` is drawn at a fixed compressed
width. Measure-window centres are overlaid as markers.
"""

from __future__ import annotations

from typing import List, Sequence

from ..core.plan import GRID
from ..core.plotting import Panel, PanelGrid, Series, palette
from .base import RunSpec, t_unit

__all__ = ["draw_preview"]

#: At most this many runs are drawn, matching the Qt version.
MAX_RUNS = 6


def draw_preview(runs: Sequence[RunSpec], title: str = "Waveform preview") -> PanelGrid:
    """Build a one-panel preview of the force waveform(s)."""
    panel = Panel(title=title, ylabel="force voltage (V)")
    tmax_disp = 0.0
    all_series: List[Series] = []

    for ri, run in enumerate(runs[:MAX_RUNS]):
        pl = run.plan
        wmax = 0
        for v in pl.vec:
            if v.v != 0.0:
                wmax = max(wmax, v.n)
        cmax = max(2 * wmax, 200)  # >= 2 us of compressed idle

        # segs maps real ticks -> displayed ticks, piecewise linearly.
        segs: List[tuple] = []  # (r0, r1, d0, d1)
        s = Series(
            name=run.label, color=palette(ri), style="line", x=[0.0], y=[0.0]
        )
        rt = dt = v0 = 0.0
        for v in pl.vec:
            idle = v.v == 0.0 and v0 == 0.0 and v.n > cmax
            dn = float(cmax if idle else v.n)
            segs.append((rt, rt + float(v.n), dt, dt + dn))
            rt += float(v.n)
            dt += dn
            v0 = v.v
            s.x.append(dt * GRID)
            s.y.append(v.v)

        def disp(tick: float, segs=segs, dt=dt) -> float:
            for r0, r1, d0, d1 in segs:
                if tick <= r1:
                    span = max(r1 - r0, 1.0)
                    return (d0 + (tick - r0) / span * (d1 - d0)) * GRID
            return dt * GRID

        def v_at(tick: float, pl=pl) -> float:
            acc = 0.0
            vp = 0.0
            for v in pl.vec:
                if tick <= acc + float(v.n):
                    return vp + (v.v - vp) * (tick - acc) / float(v.n)
                acc += float(v.n)
                vp = v.v
            return vp

        m = Series(
            style="markers",
            color="#f72585",
            name="measure windows" if ri == 0 else "",
        )
        for ev in pl.ev:
            for k in range(ev.points):
                c = float(ev.t + k * ev.interval) + ev.avg / 2.0
                m.x.append(disp(c))
                m.y.append(v_at(c))

        tmax_disp = max(tmax_disp, dt * GRID)
        all_series.append(s)
        all_series.append(m)

    tu_f, tu_n = t_unit(tmax_disp if tmax_disp > 0 else 1e-6)
    for s in all_series:
        s.x = [x / tu_f for x in s.x]
    panel.xlabel = f"time ({tu_n}, idle 0 V gaps compressed)"
    panel.series = all_series
    panel.legend = len(runs) <= MAX_RUNS
    return PanelGrid(panels=[panel], rows=1, cols=1)
