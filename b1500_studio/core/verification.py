"""Dry-run verification: hardware sanity, plan validity and physics checks.

Port of ``verifyExperiment()`` in ``pkg/src/experiments.cpp``.  Runs entirely
offline -- it builds the plans and inspects them, but never touches the
instrument -- so it is safe to call before every measurement.
"""

from __future__ import annotations

import math
from typing import List, Optional

from .driver import Hardware
from ..experiments.base import (
    Experiment,
    Params,
    Severity,
    VerificationIssue,
    VerificationResult,
    eng,
)
from .plan import GRID, parse_list

__all__ = ["verify_experiment"]

_READ_KEYS = ("syn_v_read", "epsc_v_read", "stdp_v_read", "rel_v_read", "ppf_v_read")
_REST_KEYS = (
    "syn_t_rest", "ppf_t_rest", "ref_t_rest", "epsc_t_rest",
    "stdp_t_rest", "stdp_reset_gap", "stoch_t_rest",
)
_EXCITATION_KEYS = (
    "syn_pot_v0", "syn_dep_v0", "epsc_v_p", "stdp_v_p", "stdp_v_post",
    "rel_v_sw", "if_v_p", "stoch_v_list", "ref_v_p", "ppf_v_p",
)
_WIDTH_KEYS = (
    "syn_pot_w0", "syn_dep_w0", "epsc_w_p", "stdp_w_p", "rel_w_sw",
    "if_w_p", "stoch_w_p", "ref_w_p", "ppf_w_p",
)
_PERIOD_KEYS = ("if_periods",)


def _err(res: VerificationResult, title: str, detail: str) -> None:
    res.ok = False
    res.issues.append(VerificationIssue(Severity.ERROR, title, detail))


def _warn(res: VerificationResult, title: str, detail: str) -> None:
    res.has_warnings = True
    res.issues.append(VerificationIssue(Severity.WARNING, title, detail))


def _values_of(p: Params, key: str) -> List[float]:
    """Numeric value(s) of a field, parsing list expressions when present."""
    if key in p.d and math.isfinite(p.d[key]):
        return [p.d[key]]
    raw = p.s.get(key)
    if not raw:
        return []
    try:
        return [v for v in parse_list(raw) if math.isfinite(v)]
    except Exception:
        return []


def verify_experiment(
    exp: Optional[Experiment], p: Params, hw: Hardware
) -> VerificationResult:
    """Validate parameters, plans and physics without touching hardware."""
    res = VerificationResult()

    if exp is None:
        _err(res, "No Experiment", "No measurement experiment selected.")
        res.summary_text = "No experiment selected."
        return res

    for name, value in p.d.items():
        if not math.isfinite(value):
            _err(
                res,
                "Non-finite Parameter",
                f"Parameter '{name}' must be a finite numeric value.",
            )
    if not res.ok:
        res.summary_text = (
            "Validation failed: enter finite values for all numeric parameters."
        )
        return res

    # ------------------------------------------------------ 1. hardware
    if hw.force_ch == hw.sink_ch:
        _err(res, "Channel Conflict",
             "Force channel and Sink channel cannot be identical.")
    if hw.force_meas != "I" and hw.sink_meas != "I":
        _err(res, "Measurement Channel",
             "At least one channel must measure current ('I').")
    if not math.isfinite(hw.vmax) or hw.vmax <= 0.0:
        _err(res, "Voltage Safety Limit",
             "Maximum device voltage must be finite and greater than zero.")
    if not math.isfinite(hw.sink_bias):
        _err(res, "Sink Bias", "Sink bias must be finite.")
    if not math.isfinite(hw.edge) or hw.edge < GRID:
        _err(res, "Timing Grid",
             f"Edge time ({eng(hw.edge, 's')}) must be >= 10 ns "
             "(WGFMU timebase resolution).")
    if not math.isfinite(hw.settle) or hw.settle < 0.0 or hw.settle > 0.95:
        _err(res, "Settle Fraction",
             "Settle fraction must be between 0.0 and 0.95.")
    elif hw.settle < 0.10:
        _warn(res, "Low Settle Fraction",
              "Settle fraction < 0.10. Cable/probe capacitive displacement "
              "currents may corrupt the initial sample.")
    elif hw.settle > 0.85:
        _warn(res, "High Settle Fraction",
              "Settle fraction > 0.85 leaves a very narrow window for "
              "measurement averaging.")
    if hw.max_points < 1:
        _err(res, "Point Limit", "Maximum data points must be at least one.")
    if hw.irange < 0 or hw.irange > 4:
        _err(res, "Current Range",
             "Select a supported WGFMU current range.")

    # -------------------------------------------------- 2. build plans
    try:
        runs = exp.build(p, hw)
        if not runs:
            _err(res, "Empty Run Set", "Experiment generated 0 runs.")
    except Exception as exc:
        _err(res, "Plan Construction Failed", str(exc))
        res.summary_text = f"Construction error: {exc}"
        return res

    res.run_count = len(runs)
    for r in runs:
        try:
            r.plan.validate(hw.vmax, hw.max_points)
        except Exception as exc:
            _err(res, f"Plan Validation Failed ({r.label})", str(exc))

        for e in range(len(r.plan.ev)):
            # An event that measures along a ramp has no single device
            # voltage, and should not have one invented for it: conductance
            # from a swept bias would be wrong at every point but one. The
            # amplitude and voltage-limit checks below still cover it.
            if e < len(r.ev_swept) and r.ev_swept[e]:
                continue
            # Finite, but not necessarily non-zero.
            #
            # Zero is a legitimate recorded bias: it is the origin of a pulsed
            # I-V curve, where the current is the measurement and conductance
            # is simply undefined. Rejecting it made a sweep through 0 V
            # unrunnable. What this check is really for is a bias that was
            # never recorded at all, which is NaN -- and nothing downstream
            # divides by this value, so a zero cannot propagate.
            if e >= len(r.ev_measure_v) or not math.isfinite(r.ev_measure_v[e]):
                _err(
                    res,
                    f"Invalid Measurement Bias ({r.label})",
                    f"Measurement event '{r.plan.ev[e].name}' records no device "
                    "voltage, so nothing can say what bias it was read at.",
                )

        max_device_v = abs(hw.sink_bias)
        for vec in r.plan.vec:
            max_device_v = max(max_device_v, abs(vec.v - hw.sink_bias))
        res.max_abs_v = max(res.max_abs_v, max_device_v)
        if math.isfinite(hw.vmax) and max_device_v > hw.vmax + 1e-12:
            _err(
                res,
                f"Device Voltage Limit ({r.label})",
                f"Maximum |V(force) - V(sink)| = {eng(max_device_v, 'V')} "
                f"exceeds the configured device-voltage limit {eng(hw.vmax, 'V')}.",
            )
        res.total_duration += r.plan.duration()
        res.total_points = max(res.total_points, r.plan.n_points())

    # ------------------------------------------- 3. physics / tutorial
    for rk in _READ_KEYS:
        if rk in p.d:
            vr = abs(p.d[rk])
            if vr <= 0.0:
                _err(res, "Invalid Read Voltage", "Read voltage must be > 0 V.")
            elif vr > 0.3:
                _warn(res, "Read Disturbance Risk",
                      f"Read voltage ({eng(vr, 'V')}) exceeds 0.3 V. Tutorial "
                      "recommends 0.05 - 0.2 V to avoid accidental state switching.")
            elif vr < 0.02:
                _warn(res, "Low Read Voltage",
                      f"Read voltage ({eng(vr, 'V')}) is < 20 mV. Sub-nA current "
                      "pickup or noise may reduce measurement accuracy.")

    if "syn_pot_v0" in p.d and "syn_dep_v0" in p.d:
        v_pot, v_dep = p.d["syn_pot_v0"], p.d["syn_dep_v0"]
        if v_pot * v_dep > 0:
            _warn(res, "Unipolar Stimulus in Bipolar Synapse",
                  f"LTP ({eng(v_pot, 'V')}) and LTD ({eng(v_dep, 'V')}) have the "
                  "same polarity. Bipolar devices require opposite polarities.")

    for rk in _REST_KEYS:
        if rk in p.d and p.d[rk] < 100e-9:
            _warn(res, "Short Recovery Time",
                  "Rest time between pulses is < 100 ns. May cause thermal or "
                  "charge accumulation artifacts.")

    for ek in _EXCITATION_KEYS:
        for amplitude in (abs(v) for v in _values_of(p, ek)):
            if amplitude > 1.5:
                _warn(res, "High Excitation Voltage",
                      f"Excitation voltage ({eng(amplitude, 'V')}) exceeds typical "
                      "safe limits (+/-1.5 V) for memristive devices. Risk of "
                      "electrochemical damage or irreversible changes.")
            elif amplitude > 1.0:
                _warn(res, "Elevated Excitation Voltage",
                      f"Excitation voltage ({eng(amplitude, 'V')}) is elevated. "
                      "Consider reducing to minimize device degradation.")

    if "syn_v_read" in p.d and ("syn_pot_v0" in p.d or "syn_dep_v0" in p.d):
        read_v = abs(p.d["syn_v_read"])
        write_v = max(
            abs(p.d.get("syn_pot_v0", 0.0)), abs(p.d.get("syn_dep_v0", 0.0))
        )
        if write_v > 0.0:
            ratio = read_v / write_v
            if ratio > 0.5:
                _warn(res, "Read/Write Ratio Too High",
                      f"Read voltage ({eng(read_v, 'V')}) is more than 50% of write "
                      f"voltage ({eng(write_v, 'V')}). Risk of disturbing device "
                      "state during measurement. Tutorial recommends read voltage "
                      "< 20-30% of write voltage.")
            elif ratio > 0.3:
                _warn(res, "Elevated Read/Write Ratio",
                      f"Read voltage ({eng(read_v, 'V')}) is {ratio * 100:.1f}% of "
                      f"write voltage ({eng(write_v, 'V')}). Consider reducing read "
                      "voltage for better non-disturbance.")

    # Duty cycle. The Qt version looked for "if_periods" in the numeric map,
    # where a list expression never lands, so this check never fired; parse
    # the list here so it actually runs.
    max_width = 0.0
    for wk in _WIDTH_KEYS:
        for w in (abs(v) for v in _values_of(p, wk)):
            max_width = max(max_width, w)
    min_period = math.inf
    for pk in _PERIOD_KEYS:
        for period in (abs(v) for v in _values_of(p, pk)):
            min_period = min(min_period, period)

    if max_width > 0.0 and math.isfinite(min_period) and min_period > 0:
        duty = max_width / min_period
        if duty > 0.5:
            _warn(res, "High Duty Cycle",
                  f"Estimated duty cycle is {duty * 100:.1f}%. High duty cycle may "
                  "cause device heating. Consider reducing pulse width or "
                  "increasing the interval between pulses.")
        elif duty > 0.3:
            _warn(res, "Moderate Duty Cycle",
                  f"Estimated duty cycle is {duty * 100:.1f}%. Monitor for thermal "
                  "effects during extended measurements.")

    if "syn_pot_v0" in p.d or "syn_dep_v0" in p.d:
        min_excitation = max(
            abs(p.d.get("syn_pot_v0", 0.0)), abs(p.d.get("syn_dep_v0", 0.0))
        )
        if min_excitation < 0.1:
            _warn(res, "Low Excitation Amplitude",
                  f"Excitation amplitude ({eng(min_excitation, 'V')}) may be too "
                  "small to produce reliable measurements. Consider increasing it "
                  "to improve signal-to-noise ratio.")

    # ------------------------------------------------------- 4. summary
    if not res.ok:
        first = res.issues[0].detail if res.issues else "unknown error"
        res.summary_text = f"Validation Failed: {first}"
    elif res.has_warnings:
        res.summary_text = (
            f"Validated with Warnings: {res.run_count} run(s), "
            f"{eng(res.total_duration, 's')} pulse duration, "
            f"max |V| = {eng(res.max_abs_v, 'V')}"
        )
    else:
        res.summary_text = (
            f"Validated (Safe): {res.run_count} run(s), "
            f"{eng(res.total_duration, 's')} pulse duration, "
            f"max |V| = {eng(res.max_abs_v, 'V')}"
        )
    return res
