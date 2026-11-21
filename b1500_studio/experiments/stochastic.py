"""Neuron: stochastic firing probability.

Port of ``class NeuronStochastic`` in ``pkg/src/experiments.cpp``.
Repeats identical threshold pulses and estimates ``P(fire)`` with Wilson
95 % confidence bounds.
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
    eng,
    finish_table,
    require_finite,
)
from .registry import grouped, num_field, text_field

__all__ = ["NeuronStochastic"]


class NeuronStochastic(Experiment):
    """Repeat identical pulses at each amplitude and count the spikes."""

    key = "stochastic"

    @property
    def name(self) -> str:
        return "Neuron  stochastic firing probability"

    def fields(self) -> List[dict]:
        return [
            *grouped(
                "Repeated threshold stimulus",
                text_field("stoch_v_list", "Amplitude(s) (V)", "lin(0.45,0.9,10)",
                           "one run per amplitude; comma list or lin(start,stop,n)"),
                num_field("stoch_w_p", "Pulse width (s)", "2u"),
                num_field("stoch_trials", "Trials per amplitude", "100",
                          "repeat identical pulses; maximum 10000"),
                num_field("stoch_samples", "Current samples per trial", "10",
                          "2-64 samples across each pulse"),
            ),
            *grouped(
                "Firing classification and recovery",
                num_field("stoch_i_th", "Spike threshold |I| (A)", "10u",
                          "a trial fired when peak current >= threshold"),
                num_field("stoch_t_rest", "Rest between trials (s)", "1m",
                          "allow volatile state to recover before each trial"),
                num_field("stoch_pause", "Pause between amplitudes (s)", "0.2"),
            ),
        ]

    def pause_between_runs(self, p: Params) -> float:
        return p.num("stoch_pause")

    # --------------------------------------------------------------- build

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        amplitudes = parse_list(p.str("stoch_v_list"))
        require_finite(amplitudes, "stochastic stimulus amplitude")
        if len(amplitudes) > 100:
            raise ValueError(
                "stochastic amplitude sweep is limited to 100 values"
            )
        return [self.one(p, hw, v) for v in amplitudes]

    def one(self, p: Params, hw: Hardware, voltage: float) -> RunSpec:
        r = RunSpec()
        r.plan.hint = "neuron"
        r.label = "Vp = " + eng(voltage, "V", 3)
        W = Wave(r.plan)

        trials = p.integer("stoch_trials")
        samples = p.integer("stoch_samples")
        edge = ticks(hw.edge)
        width = ticks(p.num("stoch_w_p"))
        rest = ticks(p.num("stoch_t_rest"))

        if trials < 1 or trials > 10000:
            raise ValueError("trial count must be between 1 and 10000")
        if samples < 2 or samples > 64:
            raise ValueError("samples per trial must be between 2 and 64")
        if width // samples < 1:
            raise ValueError(
                "pulse width is too short for the requested current samples"
            )

        r.v["voltage"] = voltage
        r.v["threshold"] = p.num("stoch_i_th")
        r.v["trials"] = float(trials)

        for trial in range(trials):
            W.to(voltage, edge)
            start = W.now()
            W.hold(width)
            end = W.now()
            W.to(0.0, edge)
            interval = max(1, (end - start) // samples)
            points = (end - start) // interval
            add_event(r, f"trial{trial}", start, points, interval, interval,
                      1, trial, voltage, secs(width), voltage - hw.sink_bias)
            W.hold(rest)
        return r

    # ------------------------------------------------------------- analyze

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        current = res.current(hw)
        t, ei = r.plan.samples()
        n = min(len(current), len(t))
        R = RunResult(label=r.label)

        peak: Dict[int, float] = {}
        total: Dict[int, float] = {}
        count: Dict[int, int] = {}
        for i in range(n):
            idx = int(r.ev_idx[int(ei[i])])
            value = abs(current[i])
            peak[idx] = max(peak.get(idx, 0.0), value)
            total[idx] = total.get(idx, 0.0) + value
            count[idx] = count.get(idx, 0) + 1

        trials = int(r.v["trials"])
        fired = 0
        current_sum = 0.0
        trial_number: List[float] = []
        peak_current: List[float] = []
        mean_current: List[float] = []
        outcome: List[float] = []

        for trial in range(trials):
            if count.get(trial, 0) == 0:
                continue
            peak_i = peak[trial]
            did_fire = peak_i >= r.v["threshold"]
            fired += 1 if did_fire else 0
            current_sum += peak_i
            trial_number.append(float(trial + 1))
            peak_current.append(peak_i)
            mean_current.append(total[trial] / count[trial])
            outcome.append(1.0 if did_fire else 0.0)

        measured = len(trial_number)
        if measured != trials:
            R.notes.append(f"partial trials: {measured}/{trials}")
        if measured < 30:
            R.notes.append(
                "fewer than 30 trials; firing-probability confidence interval is broad"
            )

        probability = fired / measured if measured > 0 else NaN
        z = 1.96
        ci_low = ci_high = NaN
        if measured > 0:
            sample_count = float(measured)
            denom = 1.0 + z * z / sample_count
            center = (probability + z * z / (2.0 * sample_count)) / denom
            half = (
                z
                * math.sqrt(
                    probability * (1.0 - probability) / sample_count
                    + z * z / (4.0 * sample_count * sample_count)
                )
                / denom
            )
            ci_low = max(0.0, center - half)
            ci_high = min(1.0, center + half)

        R.a["trial"] = trial_number
        R.a["peakI"] = peak_current
        R.a["meanI"] = mean_current
        R.a["outcome"] = outcome
        R.v["voltage"] = r.v["voltage"]
        R.v["threshold"] = r.v["threshold"]
        R.v["probability"] = probability
        R.v["ciLow"] = ci_low
        R.v["ciHigh"] = ci_high

        R.summary.append(("stimulus amplitude (V)", r.v["voltage"]))
        R.summary.append(("trials measured", float(measured)))
        R.summary.append(("spikes detected", float(fired)))
        R.summary.append(("firing probability", probability))
        R.summary.append(("Wilson 95% lower", ci_low))
        R.summary.append(("Wilson 95% upper", ci_high))
        R.summary.append(
            ("mean peak |I| (A)", current_sum / measured if measured > 0 else NaN)
        )

        finish_table(R, r, t, ei, n, list(current[:n]), [], "unused")
        R.cols.pop()
        R.data.pop()
        return R

    # ---------------------------------------------------------------- draw

    def draw(self, results: List[RunResult]) -> PanelGrid:
        probability = Panel(
            title="Firing probability vs pulse amplitude",
            xlabel="pulse amplitude (V)",
            ylabel="P(fire)",
            row=0, col=0,
        )
        currents = Panel(
            title="Peak current by trial",
            xlabel="trial #",
            ylabel="peak |I| (uA)",
            row=0, col=1,
        )
        outcomes = Panel(
            title="Trial outcomes (1 = fired)",
            xlabel="trial #",
            ylabel="firing state",
            row=1, col=0,
        )
        summary = Panel(title="Stochastic neuron metrics", row=1, col=1)

        p = Series(name="P(fire)", color=palette(0), style="line_markers")
        low = Series(name="95% Wilson lower", color=palette(1), style="markers")
        high = Series(name="95% Wilson upper", color=palette(2), style="markers")

        for i, R in enumerate(results):
            color = palette(i)
            p.x.append(R.v["voltage"])
            p.y.append(R.v["probability"])
            low.x.append(R.v["voltage"])
            low.y.append(R.v["ciLow"])
            high.x.append(R.v["voltage"])
            high.y.append(R.v["ciHigh"])
            currents.series.append(
                Series(name=R.label, x=R.a["trial"],
                       y=[v * 1e6 for v in R.a["peakI"]],
                       color=color, style="line_markers")
            )
            outcomes.series.append(
                Series(name=R.label, x=R.a["trial"], y=R.a["outcome"],
                       color=color, style="line_markers")
            )

        probability.series = [p, low, high]
        if results:
            currents.hlines.append((results[0].v["threshold"] * 1e6, "#ffb703"))
        summary.text = self.summary_lines(results)
        return PanelGrid(
            panels=[probability, currents, outcomes, summary], rows=2, cols=2
        )
