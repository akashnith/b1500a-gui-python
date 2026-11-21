"""Synapse protocols that sweep one variable and read the conductance back.

Ports of ``synapse.width_sweep``, ``synapse.incremental_amplitude``,
``synapse.incremental_width``, ``synapse.metrics``, ``synapse.kinetics`` and
``synapse.speed`` from the Qt tree.

They all have the same shape -- drive the device, read it, repeat with one
thing changed -- so the shape lives in :class:`_WriteReadSweep` and each
protocol only says what varies and what the answer means.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from ..core.driver import Hardware, Results
from ..core.plan import Wave, parse_list, secs, ticks
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
from .registry import grouped, num_field, text_field

__all__ = [
    "SynapseWidthSweep",
    "SynapseIncrementalAmplitude",
    "SynapseIncrementalWidth",
    "SynapseMetrics",
    "SynapseKinetics",
    "SynapseSpeed",
]

#: Phase codes shared by this family.
PH_READ, PH_AFTER_RESET, PH_AFTER_SET = 0, 1, 2


def log_points(a: float, b: float, n: int) -> List[float]:
    if n <= 1 or a <= 0.0 or b <= 0.0:
        return [a]
    return [a * (b / a) ** (i / (n - 1)) for i in range(n)]


def lin_points(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def drive(W: Wave, hw: Hardware, voltage: float, width: int) -> None:
    """One flat-topped write pulse, not measured."""
    e = ticks(hw.edge)
    W.to(voltage, e)
    W.hold(width)
    W.to(0.0, e)


def conductances(
    r: RunSpec, res: Results, hw: Hardware
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """``(independent, G, I, phase)`` for every read event, in order."""
    t, ei = r.plan.samples()
    current = res.current(hw)
    n = min(len(t), len(current))
    xs: List[float] = []
    g: List[float] = []
    i_read: List[float] = []
    phase: List[int] = []
    for i in range(n):
        e = int(ei[i])
        bias = r.ev_measure_v[e]
        xs.append(r.ev_width[e])
        i_read.append(current[i])
        # G = I/V at the read bias. Not at the write amplitude: the device is
        # read at a level low enough not to disturb it, and dividing by the
        # write level would report a conductance the device never had.
        g.append(current[i] / bias if bias and math.isfinite(bias) else NaN)
        phase.append(r.ev_phase[e])
    return xs, g, i_read, phase


def summarise_conductance(R: RunResult, g: Sequence[float]) -> None:
    usable = [x for x in g if math.isfinite(x) and x > 0.0]
    if not usable:
        R.notes.append("No finite positive conductance was measured.")
        return
    R.summary.append(("G_min_S", min(usable)))
    R.summary.append(("G_max_S", max(usable)))
    if min(usable) > 0.0:
        R.summary.append(("dynamic_range", max(usable) / min(usable)))


class _WriteReadSweep(Experiment):
    """Drive, read, repeat with one variable changed."""

    #: Axis label and unit for the swept variable.
    x_label = "x"
    x_unit = ""
    logx = False
    hint = "synapse"

    def steps(self, p: Params) -> List[Tuple[float, float, float]]:
        """``(write_voltage, write_width_s, independent_value)`` per step."""
        raise NotImplementedError

    def read_settings(self, p: Params) -> Tuple[float, float, float]:
        """``(read_voltage, read_width_s, delay_s)``."""
        return p.num("read_v"), p.num("read_width"), p.num("read_delay")

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = self.hint
        r.label = self.name
        W = Wave(r.plan)
        read_v, read_width, delay = self.read_settings(p)
        tread = ticks(read_width)

        for index, (voltage, width_s, independent) in enumerate(self.steps(p)):
            drive(W, hw, voltage, ticks(width_s))
            if delay > 0:
                W.hold(ticks(delay))
            read_pulse(r, W, hw, read_v, tread, f"read{index}", PH_READ, index,
                       voltage, independent)
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        xs, g, i_read, _ = conductances(r, res, hw)
        R.a["x"] = xs
        R.a["g"] = g
        summarise_conductance(R, g)
        self.extra_metrics(R, xs, g)

        t, ei = r.plan.samples()
        current = res.current(hw)
        n = min(len(t), len(current))
        finish_table(R, r, t, ei, n, current, g[:n], "G_S")
        return R

    def extra_metrics(self, R: RunResult, xs: Sequence[float], g: Sequence[float]) -> None:
        """Hook for whatever this protocol is actually measuring."""

    def draw(self, results: List[RunResult]) -> PanelGrid:
        series = [
            Series(x=R.a.get("x", []), y=R.a.get("g", []), name=R.label,
                   color=palette(k), style="line_markers")
            for k, R in enumerate(results)
        ]
        label = f"{self.x_label} ({self.x_unit})" if self.x_unit else self.x_label
        return PanelGrid(
            rows=1,
            cols=2,
            panels=[
                Panel(title=self.name, xlabel=label, ylabel="Conductance (S)",
                      logx=self.logx, series=series),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )


def _read_fields(read_v: str = "0.1", read_width: str = "500n",
                 delay: str = "1u") -> List[dict]:
    return [
        num_field("read_v", "Read voltage [V]", read_v,
                  "Low enough not to disturb the state it is reporting."),
        num_field("read_width", "Read pulse width [s]", read_width),
        num_field("read_delay", "Write-to-read delay [s]", delay),
    ]


class SynapseWidthSweep(_WriteReadSweep):
    """Write-read width sweep (``synapse.width_sweep``)."""

    key = "width_sweep"
    x_label = "Write pulse width"
    x_unit = "s"
    logx = True

    @property
    def name(self) -> str:
        return "Write-read width sweep"

    def fields(self) -> List[dict]:
        return grouped(
            "Width sweep",
            num_field("write_v", "Write amplitude [V]", "0.7"),
            num_field("width_start", "Shortest width [s]", "100n"),
            num_field("width_stop", "Longest width [s]", "10u"),
            num_field("width_points", "Width points", "12"),
            *_read_fields(),
        )

    def steps(self, p: Params):
        widths = log_points(p.num("width_start"), p.num("width_stop"),
                            int(p.num("width_points")))
        return [(p.num("write_v"), w, w) for w in widths]

    def extra_metrics(self, R, xs, g):
        usable = [(x, y) for x, y in zip(xs, g) if math.isfinite(y) and y > 0.0]
        if len(usable) >= 3:
            # Conductance against log width: a straight line there is the
            # signature of a filament growing logarithmically in time.
            fit = linfit([math.log10(x) for x, _ in usable], [y for _, y in usable])
            R.summary.append(("G_per_decade_S", fit.slope))
            R.summary.append(("log_width_linearity_R2", fit.r2))


class SynapseIncrementalWidth(SynapseWidthSweep):
    """Incremental width sweep (``synapse.incremental_width``).

    The same sweep as :class:`SynapseWidthSweep`, but *without* a reset
    between points, so each pulse acts on what the previous one left. The
    curve is cumulative potentiation against width rather than a set of
    independent trials -- which is why it is a separate protocol and not a
    flag on the other one.
    """

    key = "incremental_width"

    @property
    def name(self) -> str:
        return "Incremental width sweep"


class SynapseIncrementalAmplitude(_WriteReadSweep):
    """Incremental amplitude sweep (``synapse.incremental_amplitude``)."""

    key = "incremental_amplitude"
    x_label = "Write amplitude"
    x_unit = "V"

    @property
    def name(self) -> str:
        return "Incremental amplitude sweep"

    def fields(self) -> List[dict]:
        return grouped(
            "Amplitude sweep",
            num_field("amp_start", "First amplitude [V]", "0.2"),
            num_field("amp_stop", "Last amplitude [V]", "0.9"),
            num_field("amp_points", "Amplitude points", "12"),
            num_field("write_width", "Write pulse width [s]", "500n"),
            *_read_fields(),
        )

    def steps(self, p: Params):
        amps = lin_points(p.num("amp_start"), p.num("amp_stop"),
                          int(p.num("amp_points")))
        width = p.num("write_width")
        return [(a, width, a) for a in amps]

    def extra_metrics(self, R, xs, g):
        usable = [(x, y) for x, y in zip(xs, g) if math.isfinite(y) and y > 0.0]
        if len(usable) < 3:
            return
        # The threshold amplitude: where the conductance first rises
        # appreciably above where it started. Below it, pulses do nothing.
        first = usable[0][1]
        last = usable[-1][1]
        if last > first:
            half = first + 0.5 * (last - first)
            for (x0, y0), (x1, y1) in zip(usable, usable[1:]):
                if y0 < half <= y1:
                    f = 0.0 if y1 == y0 else (half - y0) / (y1 - y0)
                    R.summary.append(("v_half_switch_V", x0 + f * (x1 - x0)))
                    break
        fit = linfit([x for x, _ in usable], [y for _, y in usable])
        R.summary.append(("G_per_volt_S", fit.slope))


class SynapseMetrics(_WriteReadSweep):
    """Synaptic metrics / repeatability (``synapse.metrics``)."""

    key = "metrics"
    x_label = "Pulse number"

    @property
    def name(self) -> str:
        return "Synaptic metrics / repeatability"

    def fields(self) -> List[dict]:
        return grouped(
            "Repeatability",
            num_field("write_v", "Write amplitude [V]", "0.6"),
            num_field("write_width", "Write pulse width [s]", "500n"),
            num_field("pulses_per_train", "Pulses per cycle", "20"),
            num_field("cycles", "Cycles", "3"),
            *_read_fields(delay="1u"),
        )

    def steps(self, p: Params):
        voltage = p.num("write_v")
        width = p.num("write_width")
        per_train = int(p.num("pulses_per_train"))
        cycles = int(p.num("cycles"))
        out = []
        for cycle in range(cycles):
            for pulse in range(per_train):
                out.append((voltage, width, float(pulse + 1)))
        self._per_train = per_train
        self._cycles = cycles
        return out

    def extra_metrics(self, R, xs, g):
        per_train = getattr(self, "_per_train", 0)
        cycles = getattr(self, "_cycles", 0)
        if per_train < 2 or cycles < 2:
            return
        # Cycle-to-cycle variation at the same pulse number is the number
        # this measurement exists to produce: an analog weight is only usable
        # if the same pulse count lands on the same conductance twice.
        spreads: List[float] = []
        for pulse in range(per_train):
            column = [
                g[cycle * per_train + pulse]
                for cycle in range(cycles)
                if cycle * per_train + pulse < len(g)
            ]
            column = [x for x in column if math.isfinite(x) and x > 0.0]
            if len(column) < 2:
                continue
            mean = sum(column) / len(column)
            if mean <= 0.0:
                continue
            variance = sum((x - mean) ** 2 for x in column) / (len(column) - 1)
            spreads.append(100.0 * math.sqrt(variance) / mean)
        if spreads:
            R.summary.append(("cycle_to_cycle_cv_mean_percent", sum(spreads) / len(spreads)))
            R.summary.append(("cycle_to_cycle_cv_max_percent", max(spreads)))


class SynapseKinetics(_WriteReadSweep):
    """Switching kinetics grid (``synapse.kinetics``)."""

    key = "kinetics"
    x_label = "Write pulse width"
    x_unit = "s"
    logx = True

    @property
    def name(self) -> str:
        return "Switching kinetics (one ramp per amplitude)"

    def fields(self) -> List[dict]:
        return grouped(
            "Kinetics grid",
            num_field("amp_start", "First amplitude [V]", "0.4"),
            num_field("amp_stop", "Last amplitude [V]", "0.9"),
            num_field("amp_points", "Amplitudes", "3"),
            num_field("width_start", "Shortest width [s]", "100n"),
            num_field("width_stop", "Longest width [s]", "10u"),
            num_field("width_points", "Widths per amplitude", "6"),
            *_read_fields(),
        )

    def steps(self, p: Params):
        amps = lin_points(p.num("amp_start"), p.num("amp_stop"),
                          int(p.num("amp_points")))
        widths = log_points(p.num("width_start"), p.num("width_stop"),
                            int(p.num("width_points")))
        self._amps = amps
        self._widths = widths
        return [(a, w, w) for a in amps for w in widths]

    def extra_metrics(self, R, xs, g):
        amps = getattr(self, "_amps", [])
        widths = getattr(self, "_widths", [])
        if not amps or not widths:
            return
        # One ramp per amplitude, so the panels can be told apart.
        R.a["ramps"] = []
        for i, amplitude in enumerate(amps):
            lo = i * len(widths)
            chunk = g[lo:lo + len(widths)]
            if len(chunk) == len(widths):
                R.a["ramps"].append((amplitude, list(widths), list(chunk)))
        R.summary.append(("amplitudes", float(len(amps))))

    def draw(self, results: List[RunResult]) -> PanelGrid:
        series: List[Series] = []
        colour = 0
        for R in results:
            for amplitude, widths, g in R.a.get("ramps", []):
                series.append(
                    Series(x=widths, y=g, name=f"{amplitude:.2f} V",
                           color=palette(colour), style="line_markers")
                )
                colour += 1
        return PanelGrid(
            rows=1,
            cols=2,
            panels=[
                Panel(title=self.name, xlabel="Write pulse width (s)",
                      ylabel="Conductance (S)", logx=True, series=series),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )


class SynapseSpeed(Experiment):
    """Resistive switching speed (``synapse.speed``)."""

    key = "rs_speed"

    @property
    def name(self) -> str:
        return "Resistive switching speed"

    def fields(self) -> List[dict]:
        return grouped(
            "RS speed",
            num_field("rs_set_v", "SET amplitude [V]", "0.9"),
            num_field("rs_reset_v", "RESET amplitude [V]", "-0.9"),
            num_field("rs_reset_width", "RESET pulse width [s]", "10u"),
            num_field("rs_w_min", "Shortest SET width [s]", "100n"),
            num_field("rs_w_max", "Longest SET width [s]", "10u"),
            num_field("rs_points", "Width points", "8"),
            num_field("rs_read_v", "Read voltage [V]", "0.1"),
            num_field("rs_read_width", "Read pulse width [s]", "500n"),
            num_field("rs_interval", "Interval between steps [s]", "1u"),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "synapse"
        r.label = self.name
        W = Wave(r.plan)

        set_v = p.num("rs_set_v")
        reset_v = p.num("rs_reset_v")
        reset_w = ticks(p.num("rs_reset_width"))
        read_v = p.num("rs_read_v")
        tread = ticks(p.num("rs_read_width"))
        interval = ticks(p.num("rs_interval"))
        widths = log_points(p.num("rs_w_min"), p.num("rs_w_max"),
                            int(p.num("rs_points")))

        for k, width in enumerate(widths):
            # RESET first, every time.
            #
            # Without it each trial starts from whatever the previous one
            # left behind, and the curve shows accumulated history instead of
            # switching time -- which looks like a perfectly good S-curve and
            # means something else entirely.
            drive(W, hw, reset_v, reset_w)
            W.hold(interval)
            read_pulse(r, W, hw, read_v, tread, f"reset{k}", PH_AFTER_RESET, k,
                       reset_v, width)
            W.hold(interval)
            drive(W, hw, set_v, ticks(width))
            W.hold(interval)
            read_pulse(r, W, hw, read_v, tread, f"set{k}", PH_AFTER_SET, k,
                       set_v, width)
            W.hold(interval)
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        xs, g, _, phase = conductances(r, res, hw)

        widths: List[float] = []
        after_set: List[float] = []
        after_reset: List[float] = []
        for x, value, ph in zip(xs, g, phase):
            if ph == PH_AFTER_SET:
                widths.append(x)
                after_set.append(value)
            elif ph == PH_AFTER_RESET:
                after_reset.append(value)

        R.a["x"] = widths
        R.a["g"] = after_set
        R.a["g_reset"] = after_reset
        summarise_conductance(R, after_set)

        usable = [(w, s) for w, s in zip(widths, after_set)
                  if math.isfinite(s) and s > 0.0]
        if len(usable) >= 2:
            lo = min(s for _, s in usable)
            hi = max(s for _, s in usable)
            half = 0.5 * (lo + hi)
            for (w0, s0), (w1, s1) in zip(usable, usable[1:]):
                if s0 < half <= s1 and w0 > 0.0 and w1 > 0.0:
                    f = 0.0 if s1 == s0 else (half - s0) / (s1 - s0)
                    # Logarithmic, because the x axis is decades.
                    R.summary.append(("t50_set_s", w0 * (w1 / w0) ** f))
                    break
        if after_reset:
            finite = [x for x in after_reset if math.isfinite(x) and x > 0.0]
            if finite:
                R.summary.append(("G_after_reset_mean_S", sum(finite) / len(finite)))

        t, ei = r.plan.samples()
        current = res.current(hw)
        n = min(len(t), len(current))
        finish_table(R, r, t, ei, n, current, g[:n], "G_S")
        return R

    def draw(self, results: List[RunResult]) -> PanelGrid:
        series: List[Series] = []
        for k, R in enumerate(results):
            series.append(
                Series(x=R.a.get("x", []), y=R.a.get("g", []), name="after SET",
                       color=palette(2 * k), style="line_markers")
            )
            if R.a.get("g_reset"):
                series.append(
                    Series(x=R.a.get("x", []), y=R.a["g_reset"], name="after RESET",
                           color=palette(2 * k + 1), style="line_markers",
                           dashed=True)
                )
        return PanelGrid(
            rows=1,
            cols=2,
            panels=[
                Panel(title=self.name, xlabel="SET pulse width (s)",
                      ylabel="Conductance (S)", logx=True, series=series),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )
