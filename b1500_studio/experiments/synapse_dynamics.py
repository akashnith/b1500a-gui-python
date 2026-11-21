"""Synapse protocols whose variable is time, rate or programming state.

Ports of ``synapse.srdp``, ``synapse.stp``, ``synapse.mlc``,
``synapse.retention`` and ``synapse.endurance`` from the Qt tree.

Unlike the sweeps in :mod:`synapse_sweeps`, the interesting quantity here is
not "conductance after one pulse" but how the conductance moves while
something else is held: a pulse rate, a waiting time, a programming loop, or
a cycle count.
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
    log_points,
    summarise_conductance,
)

__all__ = [
    "SynapseSrdp",
    "SynapseStp",
    "SynapseMlc",
    "SynapseRetention",
    "SynapseEndurance",
]

PH_TRAIN, PH_RECOVERY, PH_VERIFY, PH_SET_STATE, PH_RESET_STATE = 1, 2, 3, 4, 5


def _read_fields() -> List[dict]:
    return [
        num_field("read_v", "Read voltage [V]", "0.1"),
        num_field("read_width", "Read pulse width [s]", "500n"),
    ]


class SynapseSrdp(Experiment):
    """Spike-rate-dependent plasticity (``synapse.srdp``)."""

    key = "srdp"

    @property
    def name(self) -> str:
        return "SRDP / frequency dependence"

    def fields(self) -> List[dict]:
        return grouped(
            "SRDP",
            num_field("write_v", "Pulse amplitude [V]", "0.6"),
            num_field("write_width", "Pulse width [s]", "500n"),
            num_field("pulses_per_train", "Pulses per train", "10"),
            num_field("f_start", "Lowest rate [Hz]", "1k"),
            num_field("f_stop", "Highest rate [Hz]", "1M"),
            num_field("f_points", "Rate points", "6"),
            *_read_fields(),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "synapse"
        r.label = self.name
        W = Wave(r.plan)

        voltage = p.num("write_v")
        width = ticks(p.num("write_width"))
        train = int(p.num("pulses_per_train"))
        read_v = p.num("read_v")
        tread = ticks(p.num("read_width"))
        edge = ticks(hw.edge)
        freqs = log_points(p.num("f_start"), p.num("f_stop"), int(p.num("f_points")))

        for k, frequency in enumerate(freqs):
            period = ticks(1.0 / max(frequency, 1e-9))
            for pulse in range(train):
                drive(W, hw, voltage, width)
                if pulse + 1 < train:
                    # The gap is the period minus the pulse that has just
                    # been emitted, so the RATE is what varies and the pulse
                    # itself stays identical across the sweep.
                    W.hold(max(0, period - width - 2 * edge))
            read_pulse(r, W, hw, read_v, tread, f"f{k}", PH_READ, k, voltage,
                       frequency)
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        xs, g, _, _ = conductances(r, res, hw)
        R.a["x"] = xs
        R.a["g"] = g
        summarise_conductance(R, g)

        usable = [(x, y) for x, y in zip(xs, g)
                  if math.isfinite(y) and y > 0.0 and x > 0.0]
        if len(usable) >= 3:
            # Rate dependence is read on a log-frequency axis: a synapse that
            # responds to rate shows a slope there, one that only responds to
            # pulse count does not.
            fit = linfit([math.log10(x) for x, _ in usable], [y for _, y in usable])
            R.summary.append(("G_per_decade_S", fit.slope))
            R.summary.append(("rate_linearity_R2", fit.r2))
            first, last = usable[0][1], usable[-1][1]
            if first > 0.0:
                R.summary.append(("rate_gain_percent", 100.0 * (last - first) / first))

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
                Panel(title=self.name, xlabel="Pulse rate (Hz)",
                      ylabel="Conductance (S)", logx=True, series=series),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )


class _TimeCourse(Experiment):
    """Read a decaying state at logarithmically spaced times."""

    x_label = "Time"

    def _timed_reads(
        self, r: RunSpec, W: Wave, hw: Hardware, times: Sequence[float],
        read_v: float, tread: int, phase: int, start_index: int,
        amplitude: float,
    ) -> None:
        """Reads placed so each lands at its own elapsed time."""
        edge = ticks(hw.edge)
        previous = 0.0
        for k, when in enumerate(times):
            # The delay is measured from the PREVIOUS read, and the read
            # pulse itself costs time, so both have to come off the gap or
            # the later points drift steadily late.
            gap = when - previous - secs(tread + 2 * edge)
            if gap > 0:
                W.hold(ticks(gap))
            read_pulse(r, W, hw, read_v, tread, f"t{start_index + k}", phase,
                       start_index + k, amplitude, when)
            previous = when

    def _decay_metrics(self, R: RunResult, xs: Sequence[float],
                       g: Sequence[float]) -> None:
        usable = [(x, y) for x, y in zip(xs, g)
                  if math.isfinite(y) and y > 0.0 and x > 0.0]
        if len(usable) < 3:
            return
        first, last = usable[0][1], usable[-1][1]
        R.summary.append(("G_first_S", first))
        R.summary.append(("G_last_S", last))
        if first > 0.0:
            R.summary.append(("retention_percent", 100.0 * last / first))
            R.summary.append(("decay_percent", 100.0 * (first - last) / first))
        # A straight line on a log-time axis is the signature of a broad
        # distribution of relaxation times, which is what these devices
        # normally show instead of a single exponential.
        fit = linfit([math.log10(x) for x, _ in usable], [y for _, y in usable])
        R.summary.append(("G_per_decade_S", fit.slope))
        R.summary.append(("log_time_linearity_R2", fit.r2))

    def draw(self, results: List[RunResult]) -> PanelGrid:
        series = [
            Series(x=R.a.get("x", []), y=R.a.get("g", []), name=R.label,
                   color=palette(k), style="line_markers")
            for k, R in enumerate(results)
        ]
        return PanelGrid(
            rows=1, cols=2,
            panels=[
                Panel(title=self.name, xlabel="Time (s)",
                      ylabel="Conductance (S)", logx=True, series=series),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )


class SynapseRetention(_TimeCourse):
    """Pulsed retention (``synapse.retention``)."""

    key = "retention"

    @property
    def name(self) -> str:
        return "Pulsed retention"

    def fields(self) -> List[dict]:
        return grouped(
            "Retention",
            num_field("write_v", "Conditioning amplitude [V]", "0.9"),
            num_field("write_width", "Conditioning pulse width [s]", "2u"),
            num_field("conditioning_pulses", "Conditioning pulses", "50"),
            num_field("ret_first", "First read at [s]", "1m"),
            num_field("ret_last", "Last read at [s]", "1"),
            num_field("ret_points", "Read points", "8"),
            *_read_fields(),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "synapse"
        r.label = self.name
        W = Wave(r.plan)

        voltage = p.num("write_v")
        width = ticks(p.num("write_width"))
        for _ in range(int(p.num("conditioning_pulses"))):
            drive(W, hw, voltage, width)

        times = log_points(p.num("ret_first"), p.num("ret_last"),
                           int(p.num("ret_points")))
        self._timed_reads(r, W, hw, times, p.num("read_v"),
                          ticks(p.num("read_width")), PH_READ, 0, voltage)
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        xs, g, _, _ = conductances(r, res, hw)
        R.a["x"], R.a["g"] = xs, g
        summarise_conductance(R, g)
        self._decay_metrics(R, xs, g)
        t, ei = r.plan.samples()
        current = res.current(hw)
        n = min(len(t), len(current))
        finish_table(R, r, t, ei, n, current, g[:n], "G_S")
        return R


class SynapseStp(_TimeCourse):
    """Pulse-train short-term plasticity and its decay (``synapse.stp``)."""

    key = "stp"

    @property
    def name(self) -> str:
        return "Pulse-train STP / decay"

    def fields(self) -> List[dict]:
        return grouped(
            "STP",
            num_field("write_v", "Pulse amplitude [V]", "0.7"),
            num_field("write_width", "Pulse width [s]", "500n"),
            num_field("pulses_per_train", "Pulses in the train", "10"),
            num_field("train_f", "Train frequency [Hz]", "100k"),
            num_field("rec_first", "First recovery read at [s]", "10u"),
            num_field("rec_last", "Last recovery read at [s]", "10m"),
            num_field("rec_points", "Recovery points", "6"),
            *_read_fields(),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "synapse"
        r.label = self.name
        W = Wave(r.plan)

        voltage = p.num("write_v")
        width = ticks(p.num("write_width"))
        train = int(p.num("pulses_per_train"))
        read_v = p.num("read_v")
        tread = ticks(p.num("read_width"))
        edge = ticks(hw.edge)
        period = ticks(1.0 / max(p.num("train_f"), 1e-9))

        # Every pulse in the train is read, so the facilitation during the
        # train is visible, not just its end state.
        for pulse in range(train):
            drive(W, hw, voltage, width)
            read_pulse(r, W, hw, read_v, tread, f"train{pulse}", PH_TRAIN, pulse,
                       voltage, float(pulse + 1))
            if pulse + 1 < train:
                W.hold(max(0, period - width - tread - 4 * edge))

        times = log_points(p.num("rec_first"), p.num("rec_last"),
                           int(p.num("rec_points")))
        self._timed_reads(r, W, hw, times, read_v, tread, PH_RECOVERY, train,
                          voltage)
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        xs, g, _, phase = conductances(r, res, hw)

        train_n: List[float] = []
        train_g: List[float] = []
        rec_t: List[float] = []
        rec_g: List[float] = []
        for x, value, ph in zip(xs, g, phase):
            if ph == PH_TRAIN:
                train_n.append(x)
                train_g.append(value)
            elif ph == PH_RECOVERY:
                rec_t.append(x)
                rec_g.append(value)

        R.a["train_n"], R.a["train_g"] = train_n, train_g
        R.a["x"], R.a["g"] = rec_t, rec_g
        summarise_conductance(R, train_g + rec_g)

        usable = [y for y in train_g if math.isfinite(y) and y > 0.0]
        if len(usable) >= 2 and usable[0] > 0.0:
            # Facilitation during the train: how far the conductance climbed
            # while the pulses were arriving.
            R.summary.append(
                ("train_facilitation_percent", 100.0 * (usable[-1] - usable[0]) / usable[0])
            )
        self._decay_metrics(R, rec_t, rec_g)

        t, ei = r.plan.samples()
        current = res.current(hw)
        n = min(len(t), len(current))
        finish_table(R, r, t, ei, n, current, g[:n], "G_S")
        return R

    def draw(self, results: List[RunResult]) -> PanelGrid:
        train: List[Series] = []
        recovery: List[Series] = []
        for k, R in enumerate(results):
            train.append(Series(x=R.a.get("train_n", []), y=R.a.get("train_g", []),
                                name="during train", color=palette(2 * k),
                                style="line_markers"))
            recovery.append(Series(x=R.a.get("x", []), y=R.a.get("g", []),
                                   name="recovery", color=palette(2 * k + 1),
                                   style="line_markers"))
        return PanelGrid(
            rows=1, cols=3,
            panels=[
                Panel(title="Facilitation during the train",
                      xlabel="Pulse number", ylabel="Conductance (S)",
                      series=train),
                Panel(title="Decay after the train", xlabel="Time (s)",
                      ylabel="Conductance (S)", logx=True, series=recovery),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )


class SynapseMlc(Experiment):
    """Multilevel write-verify (``synapse.mlc``)."""

    key = "mlc"

    @property
    def name(self) -> str:
        return "Multilevel write-verify"

    def fields(self) -> List[dict]:
        return grouped(
            "Multilevel",
            num_field("set_v", "Programming amplitude [V]", "0.7"),
            num_field("set_width", "Programming pulse width [s]", "200n"),
            num_field("target_levels", "Target levels", "4"),
            num_field("max_verify_pulses", "Pulses per level", "6"),
            num_field("verify_v", "Verify read voltage [V]", "0.1"),
            num_field("verify_width", "Verify read width [s]", "500n"),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "synapse"
        r.label = self.name
        W = Wave(r.plan)

        set_v = p.num("set_v")
        set_w = ticks(p.num("set_width"))
        levels = max(1, int(p.num("target_levels")))
        attempts = max(1, int(p.num("max_verify_pulses")))
        verify_v = p.num("verify_v")
        tverify = ticks(p.num("verify_width"))

        index = 0
        for level in range(levels):
            for attempt in range(attempts):
                drive(W, hw, set_v, set_w)
                # Verify after every programming pulse, not just at the end:
                # the point of write-verify is to see the approach, and the
                # pulse that overshoots is the one worth finding.
                read_pulse(r, W, hw, verify_v, tverify, f"L{level}a{attempt}",
                           PH_VERIFY, index, set_v, float(level + 1))
                index += 1
        self._levels = levels
        self._attempts = attempts
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        xs, g, _, _ = conductances(r, res, hw)
        R.a["x"] = list(range(len(g)))
        R.a["g"] = g
        R.a["level"] = xs
        summarise_conductance(R, g)

        levels = getattr(self, "_levels", 0)
        attempts = getattr(self, "_attempts", 0)
        finals: List[float] = []
        for level in range(levels):
            chunk = g[level * attempts:(level + 1) * attempts]
            chunk = [x for x in chunk if math.isfinite(x) and x > 0.0]
            if chunk:
                finals.append(chunk[-1])
        if len(finals) >= 2:
            R.summary.append(("levels_reached", float(len(finals))))
            # Level separation decides how many states are actually
            # distinguishable: levels closer together than their own spread
            # are not separate levels, whatever the programming intended.
            gaps = [abs(b - a) for a, b in zip(finals, finals[1:])]
            R.summary.append(("level_separation_min_S", min(gaps)))
            R.summary.append(("level_separation_mean_S", sum(gaps) / len(gaps)))
            span = max(finals) - min(finals)
            if span > 0.0:
                R.summary.append(
                    ("level_uniformity_percent", 100.0 * min(gaps) / (span / max(len(gaps), 1)))
                )

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
                Panel(title=self.name, xlabel="Verify number",
                      ylabel="Conductance (S)", series=series),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )


class SynapseEndurance(Experiment):
    """Pulsed SET/RESET endurance (``synapse.endurance``)."""

    key = "rs_endurance"

    @property
    def name(self) -> str:
        return "Pulsed endurance"

    def fields(self) -> List[dict]:
        return grouped(
            "Endurance",
            num_field("set_v", "SET amplitude [V]", "0.9"),
            num_field("set_width", "SET pulse width [s]", "50u"),
            num_field("reset_v", "RESET amplitude [V]", "-0.9"),
            num_field("reset_width", "RESET pulse width [s]", "50u"),
            num_field("cycles", "SET/RESET cycles", "50"),
            num_field("read_every", "Read every N cycles", "5"),
            *_read_fields(),
        )

    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        r = RunSpec()
        r.plan.hint = "synapse"
        r.label = self.name
        W = Wave(r.plan)

        set_v, reset_v = p.num("set_v"), p.num("reset_v")
        set_w = ticks(p.num("set_width"))
        reset_w = ticks(p.num("reset_width"))
        cycles = max(1, int(p.num("cycles")))
        every = max(1, int(p.num("read_every")))
        read_v = p.num("read_v")
        tread = ticks(p.num("read_width"))

        index = 0
        for cycle in range(cycles):
            drive(W, hw, set_v, set_w)
            # Read the SET state BEFORE the reset, and the RESET state after.
            # Reading only once per cycle gives one state and calls it
            # endurance, which cannot show a window closing from either side.
            if (cycle + 1) % every == 0 or cycle + 1 == cycles:
                read_pulse(r, W, hw, read_v, tread, f"set{cycle}", PH_SET_STATE,
                           index, set_v, float(cycle + 1))
                index += 1
            drive(W, hw, reset_v, reset_w)
            if (cycle + 1) % every == 0 or cycle + 1 == cycles:
                read_pulse(r, W, hw, read_v, tread, f"rst{cycle}",
                           PH_RESET_STATE, index, reset_v, float(cycle + 1))
                index += 1
        return [r]

    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        R = RunResult(label=r.label)
        xs, g, _, phase = conductances(r, res, hw)

        set_x: List[float] = []
        set_g: List[float] = []
        reset_x: List[float] = []
        reset_g: List[float] = []
        for x, value, ph in zip(xs, g, phase):
            if ph == PH_SET_STATE:
                set_x.append(x)
                set_g.append(value)
            elif ph == PH_RESET_STATE:
                reset_x.append(x)
                reset_g.append(value)

        R.a["x"], R.a["g"] = set_x, set_g
        R.a["x_reset"], R.a["g_reset"] = reset_x, reset_g
        summarise_conductance(R, set_g + reset_g)

        windows = [
            s / rr
            for s, rr in zip(set_g, reset_g)
            if math.isfinite(s) and math.isfinite(rr) and rr > 0.0 and s > 0.0
        ]
        if windows:
            R.summary.append(("window_first", windows[0]))
            R.summary.append(("window_last", windows[-1]))
            R.summary.append(("window_min", min(windows)))
            if windows[0] > 0.0:
                R.summary.append(
                    ("window_loss_percent",
                     100.0 * (windows[0] - windows[-1]) / windows[0])
                )
            if min(windows) < 1.5:
                R.notes.append(
                    f"The ON/OFF window falls to {min(windows):.2f}x; below "
                    "about 1.5x the two states are no longer reliably "
                    "distinguishable and the device has failed, whatever the "
                    "cycle count reached."
                )

        t, ei = r.plan.samples()
        current = res.current(hw)
        n = min(len(t), len(current))
        finish_table(R, r, t, ei, n, current, g[:n], "G_S")
        return R

    def draw(self, results: List[RunResult]) -> PanelGrid:
        series: List[Series] = []
        for k, R in enumerate(results):
            series.append(Series(x=R.a.get("x", []), y=R.a.get("g", []),
                                 name="SET state", color=palette(2 * k),
                                 style="line_markers"))
            series.append(Series(x=R.a.get("x_reset", []), y=R.a.get("g_reset", []),
                                 name="RESET state", color=palette(2 * k + 1),
                                 style="line_markers", dashed=True))
        return PanelGrid(
            rows=1, cols=2,
            panels=[
                Panel(title=self.name, xlabel="Cycle", ylabel="Conductance (S)",
                      logx=True, series=series),
                Panel(title="Summary", text=self.summary_lines(results)),
            ],
        )
