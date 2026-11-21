"""SMU protocols: the ``smu.*`` family from the Qt tree.

These are DC and slow-pulse measurements, so they run on the SMU through
:class:`b1500_studio.smu.Smu` rather than through the WGFMU experiment
registry. That split is physical, not organisational: below about 500 us a
pulse has to come from the WGFMU, and at or above it the SMU is the better
instrument because it has real current compliance.

Each protocol is a function taking a connected SMU and returning a
:class:`SmuRun`, so the CLI, the tests and anything else share one
implementation.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "SmuRun",
    "forming",
    "read_retention",
    "ndr",
    "threshold_statistics",
    "slow_pulse_ltp_ltd",
    "dc_endurance",
    "probe_resistance",
    "export_smu_csv",
]


@dataclass
class SmuRun:
    """One protocol's measured points plus its figures of merit."""

    label: str = ""
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)
    x_name: str = "x"
    y_name: str = "y"
    status: List[str] = field(default_factory=list)
    metrics: List[Tuple[str, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    #: Optional per-point leg name, so a figure can draw each limb of a
    #: bipolar sweep as its own curve instead of one line that doubles back.
    phase: List[str] = field(default_factory=list)
    #: Optional second measured column, e.g. the current behind a resistance.
    y2: List[float] = field(default_factory=list)
    y2_name: str = ""

    def add(self, x: float, y: float, status: str = "", phase: str = "",
            y2: Optional[float] = None) -> None:
        self.x.append(x)
        self.y.append(y)
        self.status.append(status)
        self.phase.append(phase)
        if y2 is not None:
            self.y2.append(y2)

    def metric(self, name: str, value: float) -> None:
        self.metrics.append((name, value))

    def compliance_hits(self) -> int:
        return sum(1 for s in self.status if "C" in (s or "").upper())


def _lin(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [a]
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def _log(a: float, b: float, n: int) -> List[float]:
    if n <= 1 or a <= 0 or b <= 0:
        return [a]
    return [a * (b / a) ** (i / (n - 1)) for i in range(n)]


def _resistance(v: float, i: float) -> float:
    return abs(v / i) if abs(i) > 1e-15 else float("nan")


# ------------------------------------------------------------------ forming


def forming(
    smu,
    ch: int,
    v_start: float = 0.5,
    v_stop: float = 5.0,
    step: float = 0.05,
    icomp: float = 1e-4,
    settle: float = 0.01,
    stop_on_compliance: bool = True,
) -> SmuRun:
    """Ramp until the device forms, and stop the moment it does.

    Forming is a one-way event: once the filament is there, continuing to
    ramp only drives more current through a device that has already changed.
    So the ramp halts on the first compliance hit rather than completing the
    sweep, and the voltage at that point is the forming voltage.
    """
    run = SmuRun(label="Forming / stress", x_name="Voltage (V)",
                 y_name="Current (A)")
    steps = max(2, int(round(abs(v_stop - v_start) / max(step, 1e-9))) + 1)
    formed_at = float("nan")
    for v in _lin(v_start, v_stop, steps):
        current, status = smu.read_at(ch, v, icomp, settle=settle)
        run.add(v, current, status)
        if abs(current) >= abs(icomp) * 0.95:
            formed_at = v
            run.notes.append(
                f"Compliance reached at {v:.3f} V: the device has formed. "
                "The ramp stopped here rather than driving a formed device "
                "further, which is not reversible."
            )
            if stop_on_compliance:
                break
    run.metric("v_forming_V", formed_at)
    if run.y:
        run.metric("i_max_A", max(abs(i) for i in run.y))
    if not math.isfinite(formed_at):
        run.notes.append(
            "No compliance hit: the device did not form within the ramp. "
            "Raise the stop voltage, or the compliance if the stack can take it."
        )
    return run


# ---------------------------------------------------------- read / retention


def read_retention(
    smu,
    ch: int,
    v_read: float = 0.1,
    icomp: float = 1e-3,
    duration_s: float = 10.0,
    points: int = 20,
    logarithmic: bool = True,
) -> SmuRun:
    """Repeated reads at one bias: retention, and read disturb.

    Logarithmic spacing by default, because a state that drifts does most of
    its drifting early and a linear schedule spends almost all its points
    watching the part that has already settled.
    """
    run = SmuRun(label="Read / retention / disturb", x_name="Time (s)",
                 y_name="Resistance (ohm)")
    times = (_log(max(duration_s / 1000.0, 1e-3), duration_s, points)
             if logarithmic else _lin(0.0, duration_s, points))
    started = time.monotonic()
    for target in times:
        wait = target - (time.monotonic() - started)
        if wait > 0:
            time.sleep(wait)
        current, status = smu.read_at(ch, v_read, icomp)
        run.add(target, _resistance(v_read, current), status)

    usable = [r for r in run.y if math.isfinite(r) and r > 0.0]
    if usable:
        run.metric("R_first_ohm", usable[0])
        run.metric("R_last_ohm", usable[-1])
        run.metric("drift_percent", 100.0 * (usable[-1] - usable[0]) / usable[0])
        run.metric("R_spread_percent",
                   100.0 * (max(usable) - min(usable)) / usable[0])
        if abs(usable[-1] - usable[0]) / usable[0] > 0.2:
            run.notes.append(
                "More than 20% drift over the read window. At this read "
                "voltage that is read disturb, not retention -- lower "
                "v_read and repeat before trusting it as a retention figure."
            )
    return run


# ---------------------------------------------------------------------- NDR


def ndr(
    smu,
    ch: int,
    v_max: float = 1.5,
    step: float = 0.02,
    icomp: float = 1e-3,
    settle: float = 0.005,
) -> SmuRun:
    """Sweep up looking for a region where current falls as voltage rises."""
    run = SmuRun(label="NDR / PVR", x_name="Voltage (V)", y_name="Current (A)")
    steps = max(3, int(round(v_max / max(step, 1e-9))) + 1)
    for v in _lin(0.0, v_max, steps):
        current, status = smu.read_at(ch, v, icomp, settle=settle)
        run.add(v, current, status)

    # The peak is the last rise before the fall; the valley is the minimum
    # after it. Their ratio is what a selector or an oscillator needs, and a
    # device without a valley has no NDR at all.
    peak_i = peak_v = float("nan")
    valley_i = valley_v = float("nan")
    for k in range(1, len(run.y)):
        if run.y[k] < run.y[k - 1] and not math.isfinite(peak_i):
            peak_i, peak_v = run.y[k - 1], run.x[k - 1]
        if math.isfinite(peak_i):
            if not math.isfinite(valley_i) or run.y[k] < valley_i:
                valley_i, valley_v = run.y[k], run.x[k]
    # A ratio of 1.0003 is noise, not a device feature. Requiring a real
    # drop stops a monotonic sweep with a little scatter on it from being
    # reported as negative differential resistance.
    genuine = (
        math.isfinite(peak_i) and math.isfinite(valley_i)
        and valley_i > 0.0 and peak_i > valley_i * 1.05
    )
    if genuine:
        run.metric("v_peak_V", peak_v)
        run.metric("i_peak_A", peak_i)
        run.metric("v_valley_V", valley_v)
        run.metric("i_valley_A", valley_i)
        run.metric("pvr", peak_i / valley_i)
    else:
        run.notes.append(
            "No negative differential resistance in this range: the current "
            "rose monotonically. NDR needs the sweep to pass the threshold, "
            "and a compliance low enough not to hide the fall."
        )
    if run.compliance_hits():
        run.notes.append(
            f"{run.compliance_hits()} points hit compliance; those are "
            "clipped, not measured, and a PVR computed across them is wrong."
        )
    return run


# ---------------------------------------------------------------- threshold


def threshold_statistics(
    smu,
    ch: int,
    v_max: float = 1.5,
    step: float = 0.02,
    icomp: float = 1e-4,
    trials: int = 10,
    reset_v: float = -1.0,
    reset_s: float = 0.01,
    settle: float = 0.002,
) -> SmuRun:
    """Repeat a threshold sweep and report the spread of the switching point.

    The reset between trials is what makes them independent. Without it the
    second trial starts from whatever the first left, and the "distribution"
    is a drift.
    """
    run = SmuRun(label="Threshold statistics", x_name="Trial",
                 y_name="Threshold voltage (V)")
    steps = max(3, int(round(v_max / max(step, 1e-9))) + 1)
    for trial in range(max(1, trials)):
        smu.read_at(ch, reset_v, icomp, settle=reset_s)
        threshold = float("nan")
        for v in _lin(0.0, v_max, steps):
            # Each step has to dwell: a device switches because it was held
            # above threshold for a time, not because a voltage was named.
            current, _ = smu.read_at(ch, v, icomp, settle=settle)
            if abs(current) >= abs(icomp) * 0.5:
                threshold = v
                break
        run.add(float(trial + 1), threshold)
        smu.read_at(ch, 0.0, icomp)

    found = [v for v in run.y if math.isfinite(v)]
    if len(found) >= 2:
        mean = sum(found) / len(found)
        variance = sum((v - mean) ** 2 for v in found) / (len(found) - 1)
        sd = math.sqrt(variance)
        run.metric("v_threshold_mean_V", mean)
        run.metric("v_threshold_sd_V", sd)
        run.metric("v_threshold_min_V", min(found))
        run.metric("v_threshold_max_V", max(found))
        if mean > 0.0:
            run.metric("v_threshold_cv_percent", 100.0 * sd / mean)
    else:
        run.notes.append(
            "Fewer than two trials reached the threshold; raise v_max or "
            "lower the compliance that defines a switch."
        )
    return run


# -------------------------------------------------------- slow-pulse LTP/LTD


def slow_pulse_ltp_ltd(
    smu,
    ch: int,
    v_pot: float = 1.0,
    v_dep: float = -1.0,
    width_s: float = 1e-3,
    pulses: int = 20,
    cycles: int = 1,
    v_read: float = 0.1,
    icomp: float = 1e-3,
) -> SmuRun:
    """Potentiation then depression, read after every pulse.

    The SMU's millisecond pulses are far slower than the WGFMU's, which is
    the point: this is the version of the measurement that has real current
    compliance behind it.
    """
    run = SmuRun(label="Slow-pulse LTP / LTD", x_name="Pulse number",
                 y_name="Conductance (S)")
    index = 0
    for _ in range(max(1, cycles)):
        for amplitude in (v_pot, v_dep):
            for _ in range(max(1, pulses)):
                smu.read_at(ch, amplitude, icomp, settle=width_s)
                current, status = smu.read_at(ch, v_read, icomp)
                index += 1
                g = current / v_read if v_read else float("nan")
                run.add(float(index), g, status)

    usable = [g for g in run.y if math.isfinite(g) and g > 0.0]
    if usable:
        run.metric("G_min_S", min(usable))
        run.metric("G_max_S", max(usable))
        run.metric("dynamic_range", max(usable) / min(usable))
    half = max(1, len(run.y) // 2)
    pot, dep = run.y[:half], run.y[half:]
    if len(pot) >= 2 and math.isfinite(pot[0]) and pot[0] > 0:
        run.metric("ltp_change_percent", 100.0 * (pot[-1] - pot[0]) / pot[0])
    if len(dep) >= 2 and math.isfinite(dep[0]) and dep[0] > 0:
        run.metric("ltd_change_percent", 100.0 * (dep[-1] - dep[0]) / dep[0])
    return run


# ------------------------------------------------------------- DC endurance


def dc_endurance(
    smu,
    ch: int,
    v_set: float = 1.0,
    v_reset: float = -1.0,
    width_s: float = 1e-3,
    cycles: int = 50,
    read_every: int = 5,
    v_read: float = 0.1,
    icomp: float = 1e-3,
) -> SmuRun:
    """SET/RESET cycling, reading both states periodically."""
    run = SmuRun(label="DC endurance", x_name="Cycle",
                 y_name="Resistance (ohm)")
    run_reset = SmuRun(label="RESET state")
    for cycle in range(max(1, cycles)):
        smu.read_at(ch, v_set, icomp, settle=width_s)
        due = (cycle + 1) % max(1, read_every) == 0 or cycle + 1 == cycles
        if due:
            current, status = smu.read_at(ch, v_read, icomp)
            run.add(float(cycle + 1), _resistance(v_read, current), status)
        smu.read_at(ch, v_reset, icomp, settle=width_s)
        if due:
            current, status = smu.read_at(ch, v_read, icomp)
            run_reset.add(float(cycle + 1), _resistance(v_read, current), status)

    windows = [
        off / on
        for on, off in zip(run.y, run_reset.y)
        if math.isfinite(on) and math.isfinite(off) and on > 0.0
    ]
    if windows:
        run.metric("window_first", windows[0])
        run.metric("window_last", windows[-1])
        run.metric("window_min", min(windows))
        if windows[0] > 0.0:
            run.metric("window_loss_percent",
                       100.0 * (windows[0] - windows[-1]) / windows[0])
        if min(windows) < 1.5:
            run.notes.append(
                f"The window closes to {min(windows):.2f}x. Below about 1.5x "
                "the two states are not reliably distinguishable, so the "
                "cycle count reached is not an endurance figure."
            )
    run.notes.append(f"RESET-state resistances: {len(run_reset.y)} points")
    return run


# ------------------------------------------------------------- resistance


def probe_resistance(
    smu,
    ch: int,
    v_read: float = 0.1,
    icomp: float = 1e-3,
    repeats: int = 10,
    four_probe: bool = False,
    sense_ch: Optional[int] = None,
) -> SmuRun:
    """Repeated resistance reads, 2-probe or 4-probe.

    A 2-probe reading includes the cables and the contacts. On a low-value
    device that is most of what it measures, which is why the 4-probe option
    exists and why the difference between them is worth reporting rather
    than assuming.
    """
    run = SmuRun(label="2-/4-probe resistance", x_name="Repeat",
                 y_name="Resistance (ohm)")
    for k in range(max(1, repeats)):
        current, status = smu.read_at(ch, v_read, icomp)
        measured = v_read
        if four_probe and sense_ch:
            # The sense channel reports the voltage actually across the
            # device, which is what makes this a four-probe measurement.
            try:
                measured, _ = smu.spot_v(sense_ch)
            except Exception:
                run.notes.append(
                    "The sense channel did not return a voltage; this fell "
                    "back to a 2-probe reading, which includes the leads."
                )
        run.add(float(k + 1), _resistance(measured, current), status)

    usable = [r for r in run.y if math.isfinite(r) and r > 0.0]
    if usable:
        mean = sum(usable) / len(usable)
        run.metric("R_mean_ohm", mean)
        if len(usable) >= 2:
            variance = sum((r - mean) ** 2 for r in usable) / (len(usable) - 1)
            sd = math.sqrt(variance)
            run.metric("R_sd_ohm", sd)
            if mean > 0.0:
                run.metric("R_cv_percent", 100.0 * sd / mean)
    run.metric("four_probe", 1.0 if (four_probe and sense_ch) else 0.0)
    return run


# ------------------------------------------------------------------ export


def export_smu_csv(runs: Sequence[SmuRun], folder: str,
                   stem: str = "smu") -> List[str]:
    """Write one CSV per run and return the paths.

    The metrics go in a ``#`` header rather than a second file: a number
    computed from a sweep belongs with that sweep, and a figure of merit
    separated from the data it came from is a figure of merit nobody can
    check.
    """
    import csv
    import os

    os.makedirs(folder, exist_ok=True)
    paths: List[str] = []
    for n, run in enumerate(runs):
        name = f"{stem}.csv" if len(runs) == 1 else f"{stem}_{n:02d}.csv"
        path = os.path.join(folder, name)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            fh.write(f"# {run.label}\n")
            for key, value in run.metrics:
                fh.write(f"# {key} = {value:.9g}\n")
            for note in run.notes:
                fh.write(f"# note: {note}\n")
            w = csv.writer(fh)
            header = ["phase", run.x_name, run.y_name]
            if run.y2:
                header.append(run.y2_name or "y2")
            header.append("status")
            w.writerow(header)
            for i in range(len(run.x)):
                row = [run.phase[i] if i < len(run.phase) else "",
                       f"{run.x[i]:.9g}", f"{run.y[i]:.9g}"]
                if run.y2:
                    row.append(f"{run.y2[i]:.9g}" if i < len(run.y2) else "")
                row.append(run.status[i] if i < len(run.status) else "")
                w.writerow(row)
        paths.append(path)
    return paths
