"""The eight SMU protocols, as objects the user interface can show.

Each class is a thin shell: the measurement itself lives in
:mod:`b1500_studio.protocols.smu` as a plain function, so the CLI, the tests
and the GUI run exactly the same code.  What the class adds is the part a
user interface needs -- the fields to ask for, the figure to draw, and the
largest voltage the run would apply, so it can be refused before anything is
connected rather than after.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence

from ..core.fields import Params, combo_field, grouped, num_field
from ..core.plotting import Panel, PanelGrid, Series, palette
from .base import (
    SmuProtocol,
    SmuSettings,
    current_range_code,
    current_range_labels,
)
from .smu import (
    SmuRun,
    dc_endurance,
    forming,
    ndr,
    probe_resistance,
    read_retention,
    slow_pulse_ltp_ltd,
    threshold_statistics,
)
from .sweep import POLARITIES, SWEEP_MODES, staircase, staircase_voltages

__all__ = [
    "SmuIv",
    "SmuForming",
    "SmuRead",
    "SmuNdr",
    "SmuThreshold",
    "SmuPulse",
    "SmuEndurance",
    "SmuResistance",
    "all_smu_protocols",
    "get_smu_protocol",
    "SMU_PROTOCOL_KEYS",
]


# --------------------------------------------------------------- shared bits


def measurement_fields(icomp: str = "1m", plc: str = "1",
                       rng: int = 0) -> List[dict]:
    """Compliance, integration time and current range.

    Every protocol asks for these, because every protocol needs different
    ones: a forming ramp wants microamps of compliance and a read wants
    milliamps, and a single shared value would make one of the two wrong.
    """
    return grouped(
        "Measurement",
        num_field("icomp", "Compliance [A]", icomp,
                  "the real protection -- the voltage limit is only a "
                  "software check on the request"),
        num_field("adc_plc", "ADC integration [PLC]", plc,
                  "longer integration averages out mains pickup; it costs "
                  "one line period per PLC, per point"),
        combo_field("i_range", "I range", current_range_labels(), rng,
                    "limited autoranging stops the instrument dropping to a "
                    "range so sensitive that a transition is missed while it "
                    "settles"),
    )


def _finite(values: Sequence[float]) -> List[float]:
    return [v for v in values if v == v and math.isfinite(v)]


def _summary_panel(runs: Sequence[SmuRun]) -> Panel:
    """The monospace block every SMU figure carries."""
    text: List[str] = []
    for R in runs:
        if not R.metrics and not R.notes:
            continue
        text.append(R.label)
        for name, value in R.metrics:
            text.append(f"  {name} = {value:.4g}")
        hits = R.compliance_hits()
        if hits:
            text.append(f"  compliance_hits = {hits}")
        for note in R.notes:
            text.append(f"  ! {note}")
        text.append("")
    return Panel(title="Summary", text=text or ["(no metrics)"])


def _xy_figure(runs: Sequence[SmuRun], logx: bool = False,
               logy: bool = False, markers: bool = True) -> PanelGrid:
    """One x-y panel plus the summary: the shape most of these share."""
    first = runs[0] if runs else SmuRun()
    series = [
        Series(x=R.x, y=R.y, name=R.label, color=palette(k),
               style="line_markers" if markers else "line")
        for k, R in enumerate(runs) if R.x
    ]
    return PanelGrid(
        rows=1, cols=2,
        panels=[
            Panel(title=first.label, xlabel=first.x_name, ylabel=first.y_name,
                  logx=logx, logy=logy, series=series),
            _summary_panel(runs),
        ],
    )


class _FunctionProtocol(SmuProtocol):
    """A protocol that is one call into :mod:`protocols.smu`."""

    logx = False
    logy = False

    def _call(self, smu, p: Params, s: SmuSettings) -> SmuRun:
        raise NotImplementedError

    def run(self, smu, p: Params, s: SmuSettings,
            log: Optional[Callable[[str], None]] = None,
            progress: Optional[Callable[[int, int, str], None]] = None,
            ) -> List[SmuRun]:
        if log:
            log(f"{self.name}: setting up channel {s.force_ch}")
        smu.setup(s.force_ch, s.common_ch or None, p.num("icomp"),
                  p.num("adc_plc"))
        if progress:
            progress(0, 1, self.name)
        run = self._call(smu, p, s)
        if progress:
            progress(1, 1, self.name)
        if log:
            log(f"{self.name}: {len(run.x)} point(s)")
        return [run]

    def draw(self, runs: Sequence[SmuRun]) -> PanelGrid:
        return _xy_figure(runs, logx=self.logx, logy=self.logy)


# ------------------------------------------------------------------ smu.iv


class SmuIv(SmuProtocol):
    """DC I-V, from a single ramp to a full bipolar resistive-switching loop."""

    key = "smu.iv"

    @property
    def name(self) -> str:
        return "DC I-V / bipolar RS sweep"

    def fields(self) -> List[dict]:
        return (
            grouped(
                "Sweep",
                # Magnitudes, never signed values. The two limbs are
                # independent -- a device can SET at +2 V and RESET at -4 V --
                # so they are two parameters, and each is the SIZE of its
                # excursion. A negative here would mean "V- max = -2" sweeping
                # to +2, the sort of double negative that puts the wrong
                # polarity on a device.
                num_field("sweep_v_pos_v", "V+ max [V]", "2",
                          "magnitude of the positive limb"),
                num_field("sweep_v_neg_v", "V- max (magnitude) [V]", "2",
                          "magnitude of the negative limb; enter it positive"),
                # Step size, not point count: the step is what an operator
                # reasons about, and it holds the resolution fixed when the
                # two limbs differ in length.
                num_field("sweep_step_v", "Step [V]", "0.1",
                          "sets the point count per limb, so both limbs keep "
                          "the same resolution"),
                num_field("step_dwell_s", "Delay per step [s]", "0.1"),
                num_field("cycles", "Cycles", "1"),
                combo_field("sweep_mode", "Sweep protocol",
                            list(SWEEP_MODES), 2,
                            "V1 is the first polarity and V2 the opposite "
                            "one; bipolar RS returns through zero between "
                            "them, which is how a switching loop is measured"),
                combo_field("first_polarity", "First polarity",
                            list(POLARITIES), 0),
            )
            + measurement_fields(icomp="1m", plc="1", rng=6)
            + grouped(
                "Analysis",
                # Not an extra applied bias: the sweep already passes through
                # this voltage on both limbs, and the analysis takes the
                # current it measured there.
                num_field("read_voltage_v", "ON/OFF read voltage [V]", "0.1",
                          "read off the finished loop; no extra bias is "
                          "applied to the device"),
                combo_field("log_i_axis", "Current axis",
                            ["Linear I", "Log |I|"], 1),
            )
        )

    # -- plan ----------------------------------------------------------

    def legs(self, p: Params):
        return staircase(
            p.num("sweep_v_pos_v"), p.num("sweep_v_neg_v"),
            p.num("sweep_step_v"), p.integer("sweep_mode"),
            p.integer("first_polarity") == 0, p.integer("cycles"),
        )

    def peak_voltage(self, p: Params) -> float:
        return max(abs(p.num("sweep_v_pos_v")), abs(p.num("sweep_v_neg_v")))

    def preview(self, p: Params, s: SmuSettings) -> PanelGrid:
        legs = self.legs(p)
        volts = staircase_voltages(legs)
        dwell = p.num("step_dwell_s")
        times = [i * dwell for i in range(len(volts))]
        info = [
            f"{len(legs)} leg(s), {len(volts)} measured point(s)",
            f"step        {p.num('sweep_step_v'):g} V",
            f"dwell       {dwell:g} s per step",
            f"duration    {len(volts) * dwell:.4g} s",
            f"peak |V|    {self.peak_voltage(p):g} V",
            "",
        ]
        for leg in legs:
            info.append(f"{leg.phase}: {leg.start:+.3g} -> {leg.stop:+.3g} V, "
                        f"{leg.steps} steps")
        return PanelGrid(
            rows=1, cols=2,
            panels=[
                Panel(title="Applied bias", xlabel="Time (s)",
                      ylabel="Voltage (V)",
                      series=[Series(x=times, y=volts, name="staircase",
                                     color=palette(0), style="line_markers")]),
                Panel(title="Sweep plan", text=info),
            ],
        )

    # -- run -----------------------------------------------------------

    def run(self, smu, p: Params, s: SmuSettings,
            log: Optional[Callable[[str], None]] = None,
            progress: Optional[Callable[[int, int, str], None]] = None,
            ) -> List[SmuRun]:
        icomp = p.num("icomp")
        plc = p.num("adc_plc")
        dwell = p.num("step_dwell_s")
        rng = current_range_code(p.integer("i_range"))
        legs = self.legs(p)

        run = SmuRun(label=self.name, x_name="Voltage (V)",
                     y_name="Current (A)", y2_name="Resistance (ohm)")

        smu.setup(s.force_ch, s.common_ch or None, icomp, plc)
        if s.common_ch:
            smu.write(f"DV {s.common_ch},0,{s.common_bias:.6g},{icomp:.6g}")
        smu.sweep_setup(s.force_ch, rng, 0.0, dwell, 0.0)
        try:
            for k, leg in enumerate(legs):
                if progress:
                    progress(k, len(legs), leg.phase)
                if log:
                    log(f"leg {k + 1}/{len(legs)}: {leg.phase}, "
                        f"{leg.start:+.3g} -> {leg.stop:+.3g} V")
                result = smu.sweep(
                    s.force_ch, leg.start, leg.stop, leg.points, icomp,
                    double=False, hold=0.0, delay=dwell, sdelay=0.0, plc=plc,
                    label=leg.phase,
                )
                # The instrument always measures the point it starts on. Where
                # the previous leg already measured it, drop it here rather
                # than let the same bias appear twice in the loop.
                skip = 0 if leg.include_start else 1
                for v, i, st in zip(result.v[skip:], result.i[skip:],
                                    list(result.status)[skip:]):
                    r = abs(v / i) if abs(i) > 1e-15 else float("nan")
                    run.add(v, i, st, leg.phase, r)
        finally:
            smu.sweep_end()
        if progress:
            progress(len(legs), len(legs), "done")
        self._analyse(run, p)
        return [run]

    # -- analysis ------------------------------------------------------

    @staticmethod
    def _analyse(run: SmuRun, p: Params) -> None:
        """Figures of merit read off the finished loop."""
        currents = [abs(i) for i in run.y if math.isfinite(i)]
        if currents:
            run.metric("i_max_A", max(currents))
        resistances = [r for r in _finite(run.y2) if r > 0]
        if resistances:
            run.metric("r_min_ohm", min(resistances))
            run.metric("r_max_ohm", max(resistances))

        # ON/OFF ratio, read off the loop at one voltage.
        #
        # A bipolar sweep passes through the read voltage more than once -- on
        # the way out and on the way back -- and in a switching device those
        # passes are in different states. That is the whole hysteresis: same
        # bias, different current. Points are matched within half a step,
        # because the staircase only lands exactly on the read voltage when it
        # happens to be a multiple of the step.
        v_read = abs(p.num("read_voltage_v"))
        tol = max(p.num("sweep_step_v") * 0.5, 1.0e-9)
        if v_read <= 0.0:
            return
        matched = [abs(i) for v, i in zip(run.x, run.y)
                   if math.isfinite(i) and abs(abs(v) - v_read) <= tol]
        matched = [i for i in matched if i > 0.0]
        # Two passes minimum: with only one, the "ratio" is a number divided
        # by itself, which is 1 regardless of the device and reads as a
        # definite result rather than as missing data.
        if len(matched) < 2:
            run.notes.append(
                f"the sweep passed {v_read:g} V fewer than twice, so no "
                "ON/OFF ratio could be read from it"
            )
            return
        i_on, i_off = max(matched), min(matched)
        run.metric("read_voltage_V", v_read)
        run.metric("i_on_A", i_on)
        run.metric("i_off_A", i_off)
        run.metric("on_off_ratio", i_on / i_off)
        run.metric("r_on_ohm", v_read / i_on)
        run.metric("r_off_ohm", v_read / i_off)

    # -- figure --------------------------------------------------------

    def draw(self, runs: Sequence[SmuRun]) -> PanelGrid:
        log_i = False
        series: List[Series] = []
        r_series: List[Series] = []
        for R in runs:
            # One curve per leg, so a loop reads as a loop instead of as one
            # line that doubles back over itself.
            order: List[str] = []
            for ph in R.phase:
                if ph not in order:
                    order.append(ph)
            for k, ph in enumerate(order or [R.label]):
                xs = [v for v, q in zip(R.x, R.phase) if q == ph] or R.x
                ys = [i for i, q in zip(R.y, R.phase) if q == ph] or R.y
                series.append(Series(x=xs, y=ys, name=ph or R.label,
                                     color=palette(k), style="line_markers"))
                if R.y2:
                    rs = [r for r, q in zip(R.y2, R.phase) if q == ph]
                    r_series.append(Series(x=xs, y=rs, name=ph or R.label,
                                           color=palette(k), style="line"))

        panels = [
            Panel(title="I-V", xlabel="Voltage (V)", ylabel="Current (A)",
                  logy=log_i, series=series),
            Panel(title="Resistance", xlabel="Voltage (V)",
                  ylabel="|V / I| (ohm)", logy=True, legend=False,
                  series=r_series),
            _summary_panel(runs),
        ]
        return PanelGrid(rows=1, cols=3, panels=panels)

    def draw_with(self, runs: Sequence[SmuRun], p: Params) -> PanelGrid:
        """:meth:`draw`, honouring the protocol's own log-axis choice."""
        grid = self.draw(runs)
        if p.integer("log_i_axis") == 1:
            # |I| on a log axis: a switching loop spans decades, and on a
            # linear axis everything below the ON state is a flat line at
            # zero. The sign is in the voltage, which is still linear.
            iv = grid.panels[0]
            iv.logy = True
            iv.ylabel = "|Current| (A)"
            for s in iv.series:
                s.y = [abs(v) for v in s.y]
        return grid


# --------------------------------------------------------- the seven others


class SmuForming(_FunctionProtocol):
    key = "smu.forming"
    logy = True

    @property
    def name(self) -> str:
        return "Forming / stress"

    def fields(self) -> List[dict]:
        return grouped(
            "Ramp",
            num_field("v_start", "Start voltage [V]", "0.5"),
            num_field("v_stop", "Stop voltage [V]", "5"),
            num_field("step", "Step [V]", "50m"),
            num_field("settle", "Delay per step [s]", "10m"),
        ) + measurement_fields(icomp="100u", plc="1", rng=6)

    def peak_voltage(self, p: Params) -> float:
        return max(abs(p.num("v_start")), abs(p.num("v_stop")))

    def _call(self, smu, p: Params, s: SmuSettings) -> SmuRun:
        return forming(smu, s.force_ch, p.num("v_start"), p.num("v_stop"),
                       p.num("step"), p.num("icomp"), p.num("settle"))


class SmuRead(_FunctionProtocol):
    key = "smu.read"
    logx = True

    @property
    def name(self) -> str:
        return "Read / retention / disturb"

    def fields(self) -> List[dict]:
        return grouped(
            "Read",
            num_field("v_read", "Read voltage [V]", "0.1"),
            num_field("duration", "Total duration [s]", "10"),
            num_field("points", "Read count", "20"),
        ) + measurement_fields(icomp="1m", plc="1", rng=0)

    def peak_voltage(self, p: Params) -> float:
        return abs(p.num("v_read"))

    def _call(self, smu, p: Params, s: SmuSettings) -> SmuRun:
        return read_retention(smu, s.force_ch, p.num("v_read"), p.num("icomp"),
                              p.num("duration"), p.integer("points"))


class SmuNdr(_FunctionProtocol):
    key = "smu.ndr"

    @property
    def name(self) -> str:
        return "NDR / peak-to-valley"

    def fields(self) -> List[dict]:
        return grouped(
            "Sweep",
            num_field("v_max", "Max voltage [V]", "1.5"),
            num_field("step", "Step [V]", "20m"),
            num_field("settle", "Delay per step [s]", "5m"),
        ) + measurement_fields(icomp="1m", plc="1", rng=6)

    def peak_voltage(self, p: Params) -> float:
        return abs(p.num("v_max"))

    def _call(self, smu, p: Params, s: SmuSettings) -> SmuRun:
        return ndr(smu, s.force_ch, p.num("v_max"), p.num("step"),
                   p.num("icomp"), p.num("settle"))


class SmuThreshold(_FunctionProtocol):
    key = "smu.threshold"

    @property
    def name(self) -> str:
        return "Threshold statistics"

    def fields(self) -> List[dict]:
        return grouped(
            "Trials",
            num_field("v_max", "Max voltage [V]", "1.5"),
            num_field("step", "Step [V]", "20m"),
            num_field("trials", "Trials", "10"),
            num_field("settle", "Delay per step [s]", "2m"),
        ) + grouped(
            "Reset between trials",
            num_field("reset_v", "Reset voltage [V]", "-1"),
            num_field("reset_s", "Reset duration [s]", "10m"),
        ) + measurement_fields(icomp="100u", plc="1", rng=6)

    def peak_voltage(self, p: Params) -> float:
        return max(abs(p.num("v_max")), abs(p.num("reset_v")))

    def _call(self, smu, p: Params, s: SmuSettings) -> SmuRun:
        return threshold_statistics(
            smu, s.force_ch, p.num("v_max"), p.num("step"), p.num("icomp"),
            p.integer("trials"), p.num("reset_v"), p.num("reset_s"),
            p.num("settle"),
        )


class SmuPulse(_FunctionProtocol):
    key = "smu.pulse"

    @property
    def name(self) -> str:
        return "Slow-pulse LTP / LTD"

    def fields(self) -> List[dict]:
        return grouped(
            "Pulses",
            num_field("v_pot", "Potentiation amplitude [V]", "0.7"),
            num_field("v_dep", "Depression amplitude [V]", "-0.7"),
            num_field("width", "Pulse width [s]", "100u",
                      "below about 500 us use the WGFMU instead: the SMU "
                      "cannot place an edge that fast"),
            num_field("pulses", "Pulses per half-cycle", "20"),
            num_field("cycles", "Cycles", "1"),
        ) + grouped(
            "Read",
            num_field("v_read", "Read voltage [V]", "0.1"),
        ) + measurement_fields(icomp="1m", plc="1", rng=0)

    def peak_voltage(self, p: Params) -> float:
        return max(abs(p.num("v_pot")), abs(p.num("v_dep")),
                   abs(p.num("v_read")))

    def _call(self, smu, p: Params, s: SmuSettings) -> SmuRun:
        return slow_pulse_ltp_ltd(
            smu, s.force_ch, p.num("v_pot"), p.num("v_dep"), p.num("width"),
            p.integer("pulses"), p.integer("cycles"), p.num("v_read"),
            p.num("icomp"),
        )


class SmuEndurance(_FunctionProtocol):
    key = "smu.endurance"
    logy = True

    @property
    def name(self) -> str:
        return "DC endurance"

    def fields(self) -> List[dict]:
        return grouped(
            "Cycling",
            num_field("v_set", "SET voltage [V]", "1"),
            num_field("v_reset", "RESET voltage [V]", "-1"),
            num_field("width", "Pulse width [s]", "1m"),
            num_field("cycles", "Cycles", "50"),
            num_field("read_every", "Read every N cycles", "5"),
        ) + grouped(
            "Read",
            num_field("v_read", "Read voltage [V]", "0.1"),
        ) + measurement_fields(icomp="1m", plc="1", rng=0)

    def peak_voltage(self, p: Params) -> float:
        return max(abs(p.num("v_set")), abs(p.num("v_reset")),
                   abs(p.num("v_read")))

    def _call(self, smu, p: Params, s: SmuSettings) -> SmuRun:
        return dc_endurance(
            smu, s.force_ch, p.num("v_set"), p.num("v_reset"), p.num("width"),
            p.integer("cycles"), p.integer("read_every"), p.num("v_read"),
            p.num("icomp"),
        )


class SmuResistance(_FunctionProtocol):
    key = "smu.resistance"

    @property
    def name(self) -> str:
        return "2-/4-probe resistance"

    def fields(self) -> List[dict]:
        return grouped(
            "Measurement",
            num_field("v_read", "Read voltage [V]", "0.1"),
            num_field("repeats", "Repeats", "10"),
        ) + grouped(
            "Four-probe",
            combo_field("four_probe", "Probe configuration",
                        ["2-probe (force channel only)",
                         "4-probe (separate sense channel)"], 0),
            num_field("sense_ch", "Sense channel", "0",
                      "only used in 4-probe; 0 means none"),
        ) + measurement_fields(icomp="1m", plc="1", rng=0)

    def peak_voltage(self, p: Params) -> float:
        return abs(p.num("v_read"))

    def _call(self, smu, p: Params, s: SmuSettings) -> SmuRun:
        four = p.integer("four_probe") == 1
        sense = p.integer("sense_ch") or None
        return probe_resistance(smu, s.force_ch, p.num("v_read"),
                                p.num("icomp"), p.integer("repeats"),
                                four, sense)


# ---------------------------------------------------------------- registry


def all_smu_protocols() -> List[SmuProtocol]:
    """One fresh instance of every SMU protocol, in display order."""
    return [
        SmuIv(),
        SmuForming(),
        SmuRead(),
        SmuNdr(),
        SmuThreshold(),
        SmuPulse(),
        SmuEndurance(),
        SmuResistance(),
    ]


#: Stable keys, in the same order as :func:`all_smu_protocols`.
SMU_PROTOCOL_KEYS = [
    "smu.iv",
    "smu.forming",
    "smu.read",
    "smu.ndr",
    "smu.threshold",
    "smu.pulse",
    "smu.endurance",
    "smu.resistance",
]


def get_smu_protocol(key: str) -> SmuProtocol:
    """Look up one protocol by key; ``iv`` and ``smu.iv`` both work."""
    k = key.strip().lower().replace("-", "_")
    if not k.startswith("smu."):
        k = "smu." + k
    for protocol in all_smu_protocols():
        if protocol.key == k:
            return protocol
    raise KeyError(
        f"unknown SMU protocol {key!r}; choose from "
        f"{', '.join(SMU_PROTOCOL_KEYS)}"
    )
