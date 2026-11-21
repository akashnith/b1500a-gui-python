"""Neuron: leaky integrate-and-fire.

Port of ``class IntegrateFire`` in ``pkg/src/experiments.cpp``.
A shorter period leaves less time to leak, so fewer pulses are needed to fire.
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
    eng,
    finish_table,
    require_finite,
)
from .registry import grouped, num_field, text_field

__all__ = ["IntegrateFire"]


class IntegrateFire(Experiment):
    """A train of sub-threshold pulses that integrate until the device fires."""

    key = "integrate_fire"

    @property
    def name(self) -> str:
        return "Neuron  integrate-and-fire (leaky)"

    def fields(self) -> List[dict]:
        return [
            *grouped(
                "Pulse train",
                num_field(
                    "if_v_p",
                    "Amplitude (V)",
                    "0.65",
                    "sub-threshold pulses that integrate",
                ),
                num_field("if_w_p", "Width (s)", "2u"),
                text_field(
                    "if_periods",
                    "Period(s) (s)",
                    "log(4u,16u,7)",
                    "list or log(a,b,n): one run per period",
                ),
                num_field("if_n", "# pulses", "30"),
            ),
            *grouped(
                "Spike detection",
                num_field(
                    "if_i_th",
                    "Spike threshold |I| (A)",
                    "10u",
                    "a pulse whose current exceeds this fired",
                ),
                num_field("if_samples", "Samples per pulse", "4"),
                num_field("if_pause", "Pause between runs (s)", "0.5"),
            ),
        ]

    def pause_between_runs(self, p: Params) -> float:
        return p.num("if_pause")

    # --------------------------------------------------------------- build

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        periods = parse_list(p.str("if_periods"))
        require_finite(periods, "pulse period")
        return [self.one(p, hw, T) for T in periods]

    def one(self, p: Params, hw: Hardware, T: float) -> RunSpec:
        r = RunSpec()
        r.plan.hint = "neuron"
        r.label = "T = " + eng(T, "s", 3)
        W = Wave(r.plan)

        e = ticks(hw.edge)
        w = ticks(p.num("if_w_p"))
        per = ticks(T)
        vp = p.num("if_v_p")
        N = p.integer("if_n")
        if N < 1:
            raise ValueError("need at least one pulse")
        if per < w + 3 * e:
            raise ValueError(
                f"period {eng(T, 's')} must exceed width + 3 edges"
            )
        S = max(1, min(64, p.integer("if_samples")))

        for k in range(N):
            start = W.now()
            W.to(vp, e)
            fs = W.now()
            W.hold(w)
            fe = W.now()
            W.to(0.0, e)
            W.hold(start + per - W.now())
            iv = max(1, (fe - fs) // S)
            add_event(
                r, f"p{k}", fs, (fe - fs) // iv, iv, iv, 1, k,
                vp, p.num("if_w_p"), vp - hw.sink_bias,
            )

        r.v["T"] = T
        r.v["N"] = float(N)
        r.v["per"] = secs(per)
        r.v["e"] = secs(e)
        r.v["ith"] = p.num("if_i_th")
        return r

    # ------------------------------------------------------------- analyze

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        cur = res.current(hw)
        t, ei = r.plan.samples()
        n = min(len(cur), len(t))
        R = RunResult(label=r.label)
        if n < len(t):
            R.notes.append("partial data")

        N = int(r.v["N"])
        ith = r.v["ith"]
        per = r.v["per"]
        e = r.v["e"]

        I = [0.0] * n
        ipk = [0.0] * N
        delay = [NaN] * N

        for i in range(n):
            idx = int(ei[i])
            I[i] = abs(cur[i])
            k = int(r.ev_idx[idx])
            ipk[k] = max(ipk[k], I[i])
            if I[i] > ith and math.isnan(delay[k]):
                delay[k] = t[i] - (k * per + e)

        spikes = 0
        first = 0
        pulse: List[float] = []
        fired: List[float] = []
        dsum = 0.0
        for k in range(N):
            pulse.append(float(k + 1))
            f = ipk[k] > ith
            fired.append(1.0 if f else 0.0)
            if f:
                spikes += 1
                if not first:
                    first = k + 1
                if math.isfinite(delay[k]):
                    dsum += delay[k]

        R.a["pulse"] = pulse
        R.a["ipk"] = ipk
        R.a["fired"] = fired
        R.v["T"] = r.v["T"]
        R.v["nfire"] = float(first) if first else NaN
        R.v["spikes"] = float(spikes)
        R.v["delay1"] = delay[first - 1] if first else NaN
        # The Qt version reads R.v["ith"] when drawing the threshold line but
        # never writes it, so the line never appeared. Publish it here.
        R.v["ith"] = ith

        R.summary.append(("period (s)", r.v["T"]))
        R.summary.append(("pulses to first spike", float(first) if first else NaN))
        R.summary.append(("# spikes", float(spikes)))
        R.summary.append(("spike prob.", spikes / N if N else NaN))
        R.summary.append(("first-spike delay (s)", R.v["delay1"]))
        R.summary.append(
            ("mean spike delay (s)", dsum / spikes if spikes else NaN)
        )

        finish_table(R, r, t, ei, n, I, [], "unused")
        R.cols.pop()
        R.data.pop()
        return R

    # ---------------------------------------------------------------- draw

    def draw(self, results: List[RunResult]) -> PanelGrid:
        a = Panel(
            title="Peak current per pulse",
            xlabel="pulse #",
            ylabel="peak |I| (uA)",
            logy=True,
            row=0,
            col=0,
        )
        b = Panel(
            title="Pulses to first spike vs period",
            xlabel="pulse period (s)",
            ylabel="N to fire",
            logx=True,
            row=0,
            col=1,
        )
        c = Panel(
            title="First-spike delay",
            xlabel="pulse period (s)",
            ylabel="delay (us)",
            logx=True,
            legend=False,
            row=1,
            col=0,
        )
        d = Panel(title="Summary", row=1, col=1)

        nf = Series(name="first spike", color=palette(0), style="line_markers")
        sp = Series(name="# spikes", color=palette(1), style="line_markers")
        dl = Series(color=palette(2), style="line_markers")

        ith = 0.0
        txt: List[str] = []
        for i, R in enumerate(results):
            a.series.append(
                Series(
                    name=R.label,
                    x=R.a["pulse"],
                    y=[v * 1e6 for v in R.a["ipk"]],
                    color=palette(i),
                    style="line_markers",
                )
            )
            nf.x.append(R.v["T"])
            nf.y.append(R.v["nfire"])
            sp.x.append(R.v["T"])
            sp.y.append(R.v["spikes"])
            dl.x.append(R.v["T"])
            dl.y.append(R.v["delay1"] * 1e6)

            txt.append(R.label)
            # The first summary row is the period, already in the label.
            for key, value in R.summary[1:]:
                txt.append(f"  {key} = {value:.4g}")
            for note in R.notes:
                txt.append(f"  ! {note}")
            txt.append("")
            ith = R.v.get("ith", ith)

        if ith > 0:
            a.hlines.append((ith * 1e6, "#ffb703"))
        b.series.extend([nf, sp])
        c.series.append(dl)
        d.text = txt
        return PanelGrid(panels=[a, b, c, d], rows=2, cols=2)
