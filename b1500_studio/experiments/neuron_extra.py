"""Neuron protocols not already covered by the integrate-and-fire family.

Ports of ``neuron.fi``, ``neuron.leak``, ``neuron.iv`` and
``neuron.oscillator`` from the Qt tree.

The firing-rate measurements here count spikes the way the Qt implementation
does: threshold crossings on the RISING edge, with the threshold taken from
the record itself -- halfway between its quiet level and its peak -- rather
than from a number typed in beforehand. A fixed threshold either misses every
spike on a quiet device or counts noise on a loud one.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from ..core.driver import Hardware, Results
from ..core.plan import Wave, secs, ticks
from ..core.plotting import Panel, PanelGrid, Series, palette
from .base import (
    NaN,
    Experiment,
    Params,
    RunResult,
    RunSpec,
    finish_table,
    linfit,
    read_pulse,
)
from .registry import grouped, num_field
from .synapse_sweeps import (
    PH_READ,
    conductances,
    drive,
    lin_points,
    log_points,
    summarise_conductance,
)

__all__ = ["NeuronFi", "NeuronLeak", "NeuronIv", "NeuronOscillator"]


def count_spikes(values: Sequence[float]) -> Tuple[int, float]:
    """Rising-edge crossings of a self-scaled threshold, and that threshold.

    The threshold is halfway between the record's quiet level (its median)
    and its peak. Taking it from the data is what lets the same analysis work
    on a device whose firing current is nanoamps and one whose is milliamps.
    """
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 3:
        return 0, NaN
    ordered = sorted(finite)
    quiet = ordered[len(ordered) // 2]
    peak = ordered[-1]
    if not peak > quiet:
        return 0, NaN
    threshold = quiet + 0.5 * (peak - quiet)
    spikes = 0
    above = False
    for v in values:
        if not math.isfinite(v):
            continue
        if not above and v >= threshold:
            spikes += 1
            above = True
        elif above and v < threshold:
            above = False
    return spikes, threshold


class NeuronFi(Experiment):
    """f-I curve: firing rate against input amplitude (``neuron.fi``)."""

    key = "fi"

    @property
    def name(self) -> str:
        return "f-I / firing-rate"

    def fields(self) -> List[dict]:
        return grouped(
            "f-I curve",
            num_field("amp_start", "First input amplitude [V]", "0.5"),
            num_field("amp_stop", "Last input amplitude [V]", "0.95"),
            num_field("amp_points", "Amplitude points", "8"),
            num_field("input_width", "Input pulse width [s]", "500n"),
            num_field("input_f", "Input pulse rate [Hz]", "200k"),
            num_field("train_duration", "Train duration per point [s]", "200u"),
            num_field("samples_per_point", "Observation samples", "40"),
            num_field("read_v", "Observation bias [V]", "0.2"),
            num_field("read_width", "Observation window [s]", "500n"),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "neuron"
        r.label = self.name
        W = Wave(r.plan)

        amps = lin_points(p.num("amp_start"), p.num("amp_stop"),
                          int(p.num("amp_points")))
        width = ticks(p.num("input_width"))
        rate = max(p.num("input_f"), 1e-9)
        period = ticks(1.0 / rate)
        pulses = max(1, int(round(p.num("train_duration") * rate)))
        read_v = p.num("read_v")
        tread = ticks(p.num("read_width"))
        edge = ticks(hw.edge)

        # Observe DURING the train, not after it.
        #
        # A neuron fires while it is being driven; once the drive stops there
        # is nothing left to count but the relaxation, which is a single
        # monotonic decay. Reading only afterwards reported exactly one
        # "spike" at every amplitude and produced an f-I curve with a
        # negative slope -- the measurement ran and meant nothing.
        self._samples = pulses
        for k, amplitude in enumerate(amps):
            for pulse in range(pulses):
                drive(W, hw, amplitude, width)
                read_pulse(r, W, hw, read_v, tread, f"a{k}p{pulse}", PH_READ, k,
                           amplitude, amplitude)
                if pulse + 1 < pulses:
                    W.hold(max(0, period - width - tread - 4 * edge))
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        xs, g, i_read, _ = conductances(r, res, hw)
        samples = getattr(self, "_samples", 1)

        # The observation span comes from the sample TIMES, not from the
        # averaging window. Each observation is its own read pulse with its
        # own edges, so the pulses are spaced much further apart than the
        # window is wide -- using the window gave rates hundreds of times too
        # high, and an f-I slope that came out negative.
        times, _ = r.plan.samples()
        amps: List[float] = []
        rates: List[float] = []
        for start in range(0, len(i_read) - samples + 1, samples):
            chunk = i_read[start:start + samples]
            spikes, _ = count_spikes(chunk)
            amps.append(xs[start])
            last = min(start + samples - 1, len(times) - 1)
            span = max(times[last] - times[start], 1e-12)
            rates.append(spikes / span)

        R.a["x"] = amps
        R.a["rate"] = rates
        R.a["g"] = g
        if rates:
            R.summary.append(("firing_rate_max_Hz", max(rates)))
            firing = [(a, f) for a, f in zip(amps, rates) if f > 0.0]
            if firing:
                # Rheobase: the smallest amplitude that produced any spike.
                R.summary.append(("rheobase_V", firing[0][0]))
            if len(firing) >= 3:
                fit = linfit([a for a, _ in firing], [f for _, f in firing])
                R.summary.append(("fi_gain_Hz_per_V", fit.slope))
                R.summary.append(("fi_linearity_R2", fit.r2))
            if not firing:
                R.notes.append(
                    "No spikes at any amplitude; raise the input amplitude or "
                    "the train duration past the device's threshold."
                )
        summarise_conductance(R, g)

        t, ei = r.plan.samples()
        current = res.current(hw)
        n = min(len(t), len(current))
        finish_table(R, r, t, ei, n, current, g[:n], "G_S")
        return R

    def draw(self, results: List[RunResult]) -> PanelGrid:
        series = [
            Series(x=R.a.get("x", []), y=R.a.get("rate", []), name=R.label,
                   color=palette(k), style="line_markers")
            for k, R in enumerate(results)
        ]
        return PanelGrid(
            rows=1, cols=2,
            panels=[
                Panel(title="f-I curve", xlabel="Input amplitude (V)",
                      ylabel="Firing rate (Hz)", series=series),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )


class NeuronLeak(Experiment):
    """Leaky integration (``neuron.leak``)."""

    key = "leak"

    @property
    def name(self) -> str:
        return "Leaky integration"

    def fields(self) -> List[dict]:
        return grouped(
            "Leaky integration",
            num_field("input_v", "Input amplitude [V]", "0.7"),
            num_field("input_width", "Input pulse width [s]", "500n"),
            num_field("pulses_per_train", "Pulses per train", "5"),
            num_field("interval_start", "Shortest inter-pulse interval [s]", "1u"),
            num_field("interval_stop", "Longest inter-pulse interval [s]", "1m"),
            num_field("interval_points", "Interval points", "8"),
            num_field("read_v", "Read voltage [V]", "0.2"),
            num_field("read_width", "Read pulse width [s]", "500n"),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "neuron"
        r.label = self.name
        W = Wave(r.plan)

        voltage = p.num("input_v")
        width = ticks(p.num("input_width"))
        train = max(1, int(p.num("pulses_per_train")))
        intervals = log_points(p.num("interval_start"), p.num("interval_stop"),
                               int(p.num("interval_points")))
        read_v = p.num("read_v")
        tread = ticks(p.num("read_width"))

        for k, interval in enumerate(intervals):
            # The same train every time; only the spacing changes. A leaky
            # integrator reaches a lower peak as the spacing grows, because
            # more of each contribution has leaked away before the next
            # arrives -- so this curve IS the leak time constant.
            for pulse in range(train):
                drive(W, hw, voltage, width)
                if pulse + 1 < train:
                    W.hold(ticks(interval))
            read_pulse(r, W, hw, read_v, tread, f"i{k}", PH_READ, k, voltage,
                       interval)
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        xs, g, _, _ = conductances(r, res, hw)
        R.a["x"], R.a["g"] = xs, g
        summarise_conductance(R, g)

        usable = [(x, y) for x, y in zip(xs, g)
                  if math.isfinite(y) and y > 0.0 and x > 0.0]
        if len(usable) >= 3:
            first, last = usable[0][1], usable[-1][1]
            if first > 0.0:
                R.summary.append(("integration_loss_percent",
                                  100.0 * (first - last) / first))
            # Where the response has fallen halfway: a direct read of the
            # leak time constant, in the units the operator set.
            half = last + 0.5 * (first - last)
            for (x0, y0), (x1, y1) in zip(usable, usable[1:]):
                if y0 >= half >= y1 and x0 > 0.0 and x1 > 0.0:
                    f = 0.0 if y1 == y0 else (y0 - half) / (y0 - y1)
                    R.summary.append(("leak_tau_s", x0 * (x1 / x0) ** f))
                    break

        t, ei = r.plan.samples()
        current = res.current(hw)
        n = min(len(t), len(current))
        finish_table(R, r, t, ei, n, current, g[:n], "G_S")
        return R

    def draw(self, results: List[RunResult]) -> PanelGrid:
        series = [
            Series(x=R.a.get("x", []), y=R.a.get("g", []), name=R.label,
                   color=palette(k), style="line_markers")
            for k, R in enumerate(results)
        ]
        return PanelGrid(
            rows=1, cols=2,
            panels=[
                Panel(title=self.name, xlabel="Inter-pulse interval (s)",
                      ylabel="Conductance (S)", logx=True, series=series),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )


class NeuronIv(Experiment):
    """Pulsed I-V (``neuron.iv``)."""

    key = "neuron_iv"

    @property
    def name(self) -> str:
        return "Pulsed I-V"

    def fields(self) -> List[dict]:
        return grouped(
            "Pulsed I-V",
            num_field("v_start", "Start voltage [V]", "-0.9"),
            num_field("v_stop", "Stop voltage [V]", "0.9"),
            num_field("v_points", "Points", "41"),
            num_field(
                "pulse_width",
                "Pulse width [s]",
                "500n",
                "Pulsed rather than DC so the device is biased only while it "
                "is being measured; a DC sweep would heat it and drift.",
            ),
            num_field("point_interval", "Interval between points [s]", "10u"),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "neuron"
        r.label = self.name
        W = Wave(r.plan)

        voltages = lin_points(p.num("v_start"), p.num("v_stop"),
                              int(p.num("v_points")))
        width = ticks(p.num("pulse_width"))
        gap = ticks(p.num("point_interval"))
        for k, voltage in enumerate(voltages):
            # The measurement IS the pulse: force and read at the same level.
            read_pulse(r, W, hw, voltage, width, f"v{k}", PH_READ, k, voltage,
                       voltage)
            W.hold(gap)
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        t, ei = r.plan.samples()
        current = res.current(hw)
        n = min(len(t), len(current))

        volts: List[float] = []
        amps: List[float] = []
        for i in range(n):
            volts.append(r.ev_amp[int(ei[i])])
            amps.append(current[i])

        R.a["v"], R.a["i"] = volts, amps
        if amps:
            R.summary.append(("I_max_A", max(amps)))
            R.summary.append(("I_min_A", min(amps)))
            peak = max(abs(x) for x in amps)
            R.summary.append(("peak_abs_current_A", peak))
            # Negative differential resistance: a stretch where the current
            # falls while the voltage rises. It is the feature that makes a
            # device able to oscillate at all, so it is worth naming.
            ndr = []
            for (v0, i0), (v1, i1) in zip(zip(volts, amps), list(zip(volts, amps))[1:]):
                if v1 > v0 and i1 < i0 and v0 > 0.0:
                    ndr.append(v0)
            if ndr:
                R.summary.append(("ndr_onset_V", min(ndr)))
                R.summary.append(("ndr_points", float(len(ndr))))
        finish_table(R, r, t, ei, n, current, volts, "drive_V")
        return R

    def draw(self, results: List[RunResult]) -> PanelGrid:
        series = [
            Series(x=R.a.get("v", []), y=R.a.get("i", []), name=R.label,
                   color=palette(k))
            for k, R in enumerate(results)
        ]
        return PanelGrid(
            rows=1, cols=2,
            panels=[
                Panel(title=self.name, xlabel="Applied voltage (V)",
                      ylabel="Current (A)", series=series),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )


class NeuronOscillator(Experiment):
    """Relaxation oscillator observation (``neuron.oscillator``)."""

    key = "oscillator"

    @property
    def name(self) -> str:
        return "Relaxation oscillator"

    def fields(self) -> List[dict]:
        return grouped(
            "Oscillator",
            num_field("supply_v", "Supply amplitude [V]", "0.9"),
            num_field("drive_width", "Drive pulse width [s]", "1u"),
            num_field("drive_f", "Drive frequency [Hz]", "100k"),
            num_field("observation", "Observation per trial [s]", "200u"),
            num_field("samples_per_trial", "Samples per trial", "60"),
            num_field("trials", "Trials", "3"),
            num_field("read_v", "Observation bias [V]", "0.2"),
            num_field("read_width", "Observation window [s]", "500n"),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "neuron"
        r.label = self.name
        W = Wave(r.plan)

        voltage = p.num("supply_v")
        width = ticks(p.num("drive_width"))
        frequency = max(p.num("drive_f"), 1e-9)
        duration = p.num("observation")
        pulses = max(1, int(round(duration * frequency)))
        trials = max(1, int(p.num("trials")))
        read_v = p.num("read_v")
        tread = ticks(p.num("read_width"))
        edge = ticks(hw.edge)
        period = ticks(1.0 / frequency)

        # Interleaved for the same reason as the f-I curve: the threshold
        # events happen while the supply is applied, so observing only
        # afterwards sees one relaxation and calls it one spike.
        self._samples = pulses
        for trial in range(trials):
            for pulse in range(pulses):
                drive(W, hw, voltage, width)
                read_pulse(r, W, hw, read_v, tread, f"t{trial}p{pulse}", PH_READ,
                           trial, voltage, float(trial + 1))
                if pulse + 1 < pulses:
                    W.hold(max(0, period - width - tread - 4 * edge))
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        xs, g, i_read, _ = conductances(r, res, hw)
        samples = getattr(self, "_samples", 1)

        trials: List[float] = []
        counts: List[float] = []
        for start in range(0, len(i_read) - samples + 1, samples):
            chunk = i_read[start:start + samples]
            spikes, _ = count_spikes(chunk)
            trials.append(xs[start])
            counts.append(float(spikes))

        R.a["x"], R.a["counts"] = trials, counts
        R.a["trace"] = i_read
        if counts:
            R.summary.append(("spikes_per_trial_mean", sum(counts) / len(counts)))
            R.summary.append(("spikes_per_trial_max", max(counts)))
            if len(counts) >= 2:
                mean = sum(counts) / len(counts)
                variance = sum((c - mean) ** 2 for c in counts) / (len(counts) - 1)
                R.summary.append(("spike_count_sd", math.sqrt(variance)))
        summarise_conductance(R, g)
        R.notes.append(
            "Illustrative only: a relaxation oscillator needs an external "
            "series resistor and load capacitor, which this instrument "
            "cannot supply. What is observed here is the device's own "
            "threshold behaviour, not a running oscillator."
        )

        t, ei = r.plan.samples()
        current = res.current(hw)
        n = min(len(t), len(current))
        finish_table(R, r, t, ei, n, current, g[:n], "G_S")
        return R

    def draw(self, results: List[RunResult]) -> PanelGrid:
        traces: List[Series] = []
        counts: List[Series] = []
        for k, R in enumerate(results):
            trace = R.a.get("trace", [])
            traces.append(Series(x=list(range(len(trace))), y=trace,
                                 name=R.label, color=palette(k)))
            counts.append(Series(x=R.a.get("x", []), y=R.a.get("counts", []),
                                 name="spikes", color=palette(k + 1),
                                 style="line_markers"))
        return PanelGrid(
            rows=1, cols=3,
            panels=[
                Panel(title="Observed current", xlabel="Sample",
                      ylabel="Current (A)", series=traces),
                Panel(title="Spikes per trial", xlabel="Trial",
                      ylabel="Spikes", series=counts),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )
