"""Ferroelectric analysis maths.

Faithful port of ``src/core/ferroanalysis.{h,cpp}`` from the Qt tree. Pure
numerics -- no instrument I/O, no plotting, no numpy -- so every formula here
is unit-testable without hardware, exactly as ``plan.py`` is.

Sign and unit conventions used throughout:

* charge ``Q`` in coulomb, current in ampere, time in seconds
* electrode area in cm^2
* polarization ``P`` in uC/cm^2   (``P = Q / A * 1e6``)
* field in MV/cm                  (``E = 10 * V / d_nm``)
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "EPS0",
    "LoopMetrics",
    "PundMetrics",
    "cumulative_charge",
    "polarization",
    "rising_mask",
    "cross_values",
    "branch_difference",
    "analyze_loop",
    "analyze_pund",
    "leakage_from_two_frequencies",
    "permittivity",
    "slope",
]

#: Vacuum permittivity [F/m].
EPS0 = 8.8541878128e-12

NaN = float("nan")


# --------------------------------------------------------------- helpers


def _median(values: Sequence[float]) -> float:
    if not values:
        return NaN
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def _pick_nearest(values: Sequence[float], reference: float) -> float:
    """Element of ``values`` nearest ``reference``; median when NaN.

    A closed loop crosses P = 0 more than once per branch when the signal is
    noisy, and the median of those crossings is not necessarily the coercive
    voltage. Choosing the crossing closest to the switching-current peak picks
    the real one; falling back to the median keeps a sane answer when no peak
    could be located.
    """
    if not values:
        return NaN
    if not math.isfinite(reference):
        return _median(values)
    best = values[0]
    best_distance = abs(best - reference)
    for value in values:
        distance = abs(value - reference)
        if distance < best_distance:
            best_distance = distance
            best = value
    return best


def _interpolate_sorted(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    """Linear interpolation of ``ys`` at ``x``; clamped outside ``xs``."""
    if not xs:
        return NaN
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    hi = bisect.bisect_left(xs, x)
    lo = max(0, hi - 1)
    span = xs[hi] - xs[lo]
    f = 0.0 if abs(span) < 1.0e-300 else (x - xs[lo]) / span
    return ys[lo] + f * (ys[hi] - ys[lo])


def slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Least-squares slope of ``y`` against ``x``; 0 when undetermined."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    sx = sy = sxx = sxy = 0.0
    for i in range(n):
        sx += x[i]
        sy += y[i]
        sxx += x[i] * x[i]
        sxy += x[i] * y[i]
    denominator = n * sxx - sx * sx
    if abs(denominator) < 1.0e-300:
        return 0.0
    return (n * sxy - sx * sy) / denominator


# ------------------------------------------------------------- dataclasses


@dataclass
class LoopMetrics:
    """Remanent/coercive figures extracted from one closed P-V loop."""

    pr_plus: float = 0.0       #: P at V=0 on the falling branch   [uC/cm^2]
    pr_minus: float = 0.0      #: P at V=0 on the rising branch    [uC/cm^2]
    two_pr: float = 0.0        #: pr_plus - pr_minus               [uC/cm^2]
    p_max: float = 0.0         #: largest P reached                [uC/cm^2]
    vc_plus: float = 0.0       #: positive coercive voltage        [V]
    vc_minus: float = 0.0      #: negative coercive voltage        [V]
    vc_mean: float = 0.0       #: (vc_plus - vc_minus)/2           [V]
    imprint: float = 0.0       #: (vc_plus + vc_minus)/2           [V]
    ec_plus: float = 0.0       #: positive coercive field          [MV/cm]
    ec_minus: float = 0.0      #: negative coercive field          [MV/cm]
    peak_current: float = 0.0  #: max |I| anywhere in the loop     [A]
    eps_r: float = 0.0         #: relative permittivity from the loop tips
    valid: bool = False


@dataclass
class PundMetrics:
    """Switched-charge figures from a P-U-N-D sequence."""

    q_p: float = 0.0
    q_u: float = 0.0
    q_n: float = 0.0
    q_d: float = 0.0
    p_switched_plus: float = 0.0   #: (qP - qU)/A = dP+  [uC/cm^2]
    p_switched_minus: float = 0.0  #: (qN - qD)/A = dP-  [uC/cm^2]
    p_star: float = 0.0            #: qP/A, switching + non-switching
    p_hat: float = 0.0             #: qU/A, non-switching only
    two_pr: float = 0.0            #: (|dP+| + |dP-|)/2  [uC/cm^2]
    pr: float = 0.0                #: two_pr / 2         [uC/cm^2]
    valid: bool = False


# ----------------------------------------------------------------- charge


def cumulative_charge(
    t: Sequence[float],
    current: Sequence[float],
    remove_drift: bool = True,
    dt: float = 0.0,
) -> List[float]:
    """Cumulative charge by integrating current.

    With a uniform sample interval pass ``dt > 0``: the exact running sum with
    a half-sample back-shift is used, which is what hardware box-averaged
    samples actually represent. Otherwise the trapezoid rule over ``t``.

    When ``remove_drift`` is true the net charge accumulated over the record is
    subtracted linearly in time. That removes the DC leakage ramp and closes
    the loop, which is mandatory before remanence can be read off at V = 0 --
    an unclosed loop puts the whole leakage integral into Pr.
    """
    n = min(len(t), len(current))
    charge = [0.0] * n
    if n == 0:
        return charge

    if dt > 0.0:
        running = 0.0
        for i in range(n):
            running += current[i] * dt
            charge[i] = running - 0.5 * current[i] * dt
    else:
        for i in range(1, n):
            charge[i] = charge[i - 1] + 0.5 * (current[i] + current[i - 1]) * (
                t[i] - t[i - 1]
            )

    if remove_drift and n > 1 and t[n - 1] > t[0]:
        total = charge[n - 1] + (0.5 * current[n - 1] * dt if dt > 0.0 else 0.0)
        span = t[n - 1] - t[0]
        for i in range(n):
            charge[i] -= total * (t[i] - t[0]) / span
    return charge


def polarization(
    charge: Sequence[float], area_cm2: float, centre: bool = True
) -> List[float]:
    """Charge -> polarization in uC/cm^2, centred on the loop midpoint."""
    p = [0.0] * len(charge)
    if not charge or not area_cm2 > 0.0:
        return p
    for i, q in enumerate(charge):
        p[i] = q / area_cm2 * 1.0e6  # C/cm^2 -> uC/cm^2
    if centre:
        offset = 0.5 * (min(p) + max(p))
        p = [value - offset for value in p]
    return p


# ---------------------------------------------------------------- branches


def rising_mask(v: Sequence[float]) -> List[bool]:
    """Per-sample mask: true where the drive voltage is rising."""
    n = len(v)
    mask = [True] * n
    if n < 2:
        return mask

    sign = [0.0] * (n - 1)
    for i in range(n - 1):
        d = v[i + 1] - v[i]
        sign[i] = 1.0 if d > 0.0 else (-1.0 if d < 0.0 else 0.0)

    # Fill leading zeros with the first real direction, then let each flat
    # stretch inherit the direction before it. A flat top otherwise reads as
    # neither branch and the loop loses its seam.
    first = 0.0
    for s in sign:
        if s != 0.0:
            first = s
            break
    if first == 0.0:
        first = 1.0
    for i in range(len(sign)):
        if sign[i] == 0.0:
            sign[i] = first if i == 0 else sign[i - 1]

    for i in range(n - 1):
        mask[i] = sign[i] > 0.0
    mask[n - 1] = mask[n - 2]
    return mask


def cross_values(
    x: Sequence[float],
    y: Sequence[float],
    level: float,
    mask: Sequence[bool],
    direction: int,
) -> List[float]:
    """Linear interpolation of ``y`` wherever ``x`` crosses ``level``.

    Only crossings whose direction matches ``direction`` (+1 rising,
    -1 falling) and whose sample pair is inside ``mask`` are returned.
    """
    out: List[float] = []
    n = min(len(x), len(y), len(mask))
    for i in range(n - 1):
        if not mask[i]:
            continue
        x0 = x[i]
        x1 = x[i + 1]
        if (x0 - level) * (x1 - level) > 0.0 or x1 == x0:
            continue
        if direction > 0 and not x1 > x0:
            continue
        if direction < 0 and not x1 < x0:
            continue
        f = (level - x0) / (x1 - x0)
        out.append(y[i] + f * (y[i + 1] - y[i]))
    return out


def branch_difference(
    v: Sequence[float], current: Sequence[float], vref: Sequence[float]
) -> Tuple[List[float], List[float]]:
    """``I_rise(V) - I_fall(V)``, on the rising-branch voltages, ascending.

    Ohmic leakage is identical on both branches and cancels, leaving the
    switching peaks. Returns ``([], [])`` when the record is too short or has
    no usable branches.
    """
    n = min(len(v), len(current), len(vref))
    if n < 6:
        return [], []

    rise = rising_mask(vref)
    v_rise: List[float] = []
    i_rise: List[float] = []
    v_fall: List[float] = []
    i_fall: List[float] = []
    for i in range(n):
        if rise[i]:
            v_rise.append(v[i])
            i_rise.append(current[i])
        else:
            v_fall.append(v[i])
            i_fall.append(current[i])
    if len(v_rise) < 3 or len(v_fall) < 3:
        return [], []

    fall_sorted = sorted(range(len(v_fall)), key=lambda k: v_fall[k])
    v_fall_sorted = [v_fall[k] for k in fall_sorted]
    i_fall_sorted = [i_fall[k] for k in fall_sorted]

    rise_order = sorted(range(len(v_rise)), key=lambda k: v_rise[k])
    voltage_out: List[float] = []
    difference_out: List[float] = []
    for index in rise_order:
        voltage = v_rise[index]
        interpolated = _interpolate_sorted(v_fall_sorted, i_fall_sorted, voltage)
        voltage_out.append(voltage)
        difference_out.append(i_rise[index] - interpolated)
    return voltage_out, difference_out


# ------------------------------------------------------------------ loops


def analyze_loop(
    v_in: Sequence[float],
    p_in: Sequence[float],
    current_in: Sequence[float],
    vref_in: Sequence[float],
    thickness_nm: float,
) -> LoopMetrics:
    """Full loop extraction.

    ``vref_in`` is the programmed drive voltage, used to decide branch
    direction; pass the measured ``v_in`` for it when no separate reference
    exists. ``thickness_nm <= 0`` skips the field and permittivity figures.
    """
    m = LoopMetrics()
    n = min(len(v_in), len(p_in), len(current_in), len(vref_in))
    if n < 4:
        return m

    # Close the seam so the last-to-first transition is a real segment.
    v = list(v_in[:n]) + [v_in[0]]
    p = list(p_in[:n]) + [p_in[0]]
    current = list(current_in[:n]) + [current_in[0]]
    vref = list(vref_in[:n]) + [vref_in[0]]

    rise = rising_mask(vref)
    fall = [not r for r in rise]

    # Voltage at the switching-current extremes, used to disambiguate Vc.
    v_switch_plus = NaN
    v_switch_minus = NaN
    best_plus = -math.inf
    best_minus = math.inf
    for i in range(len(current)):
        if rise[i] and current[i] > best_plus:
            best_plus = current[i]
            v_switch_plus = v[i]
        if fall[i] and current[i] < best_minus:
            best_minus = current[i]
            v_switch_minus = v[i]

    m.pr_plus = _median(cross_values(v, p, 0.0, fall, -1))
    m.pr_minus = _median(cross_values(v, p, 0.0, rise, +1))
    if math.isfinite(m.pr_plus) and math.isfinite(m.pr_minus):
        m.two_pr = m.pr_plus - m.pr_minus

    m.vc_plus = _pick_nearest(cross_values(p, v, 0.0, rise, +1), v_switch_plus)
    m.vc_minus = _pick_nearest(cross_values(p, v, 0.0, fall, -1), v_switch_minus)
    if math.isfinite(m.vc_plus) and math.isfinite(m.vc_minus):
        m.vc_mean = 0.5 * (m.vc_plus - m.vc_minus)
        m.imprint = 0.5 * (m.vc_plus + m.vc_minus)
        if thickness_nm > 0.0:
            m.ec_plus = 10.0 * m.vc_plus / thickness_nm  # V/nm -> MV/cm
            m.ec_minus = 10.0 * m.vc_minus / thickness_nm

    m.p_max = max(p)
    m.peak_current = max((abs(value) for value in current), default=0.0)

    # Permittivity from the slope of the loop tips, where switching is done.
    if thickness_nm > 0.0 and len(v) > 8:
        max_abs_v = max(abs(value) for value in v)
        slopes: List[float] = []
        for sign in (1, -1):
            tip_v: List[float] = []
            tip_p: List[float] = []
            for i in range(len(v)):
                if sign * v[i] > 0.9 * max_abs_v:
                    tip_v.append(v[i])
                    tip_p.append(p[i])
            if len(tip_v) >= 3:
                slopes.append(slope(tip_v, tip_p))
        if slopes:
            mean = sum(slopes) / len(slopes)
            # mean is uC/cm^2 per V; -> C/m^2 per V is *1e-2
            m.eps_r = mean * 1.0e-2 * thickness_nm * 1.0e-9 / EPS0

    m.valid = math.isfinite(m.two_pr)
    return m


def analyze_pund(
    q_p: float, q_u: float, q_n: float, q_d: float, area_cm2: float
) -> PundMetrics:
    """PUND figures from the four integrated pulse charges."""
    m = PundMetrics(q_p=q_p, q_u=q_u, q_n=q_n, q_d=q_d)
    if not area_cm2 > 0.0:
        return m
    scale = 1.0e6 / area_cm2  # C -> uC/cm^2
    m.p_star = q_p * scale
    m.p_hat = q_u * scale
    m.p_switched_plus = (q_p - q_u) * scale
    m.p_switched_minus = (q_n - q_d) * scale
    m.two_pr = 0.5 * (abs(m.p_switched_plus) + abs(m.p_switched_minus))
    m.pr = 0.5 * m.two_pr
    m.valid = True
    return m


def leakage_from_two_frequencies(
    v1: Sequence[float],
    i1: Sequence[float],
    f1: float,
    v2: Sequence[float],
    i2: Sequence[float],
    f2: float,
) -> List[float]:
    """Two-frequency (DLCC) leakage separation.

    ``I_leak(V) = [f2*I1(V) - f1*I2(V)] / (f2 - f1)``, on the ``f1`` grid.
    Returns an empty list when the inputs are unusable.
    """
    n1 = min(len(v1), len(i1))
    n2 = min(len(v2), len(i2))
    if n1 < 2 or n2 < 2 or not f2 > f1 or not f1 > 0.0:
        return []

    order = sorted(range(n2), key=lambda k: v2[k])
    vs = [v2[k] for k in order]
    ins = [i2[k] for k in order]

    leak: List[float] = []
    for i in range(n1):
        other = _interpolate_sorted(vs, ins, v1[i])
        leak.append((f2 * i1[i] - f1 * other) / (f2 - f1))
    return leak


def permittivity(capacitance_f: float, area_cm2: float, thickness_nm: float) -> float:
    """Relative permittivity from a capacitance: ``eps_r = C*d / (eps0*A)``."""
    if not area_cm2 > 0.0 or not thickness_nm > 0.0 or not capacitance_f > 0.0:
        return 0.0
    area_m2 = area_cm2 * 1.0e-4
    thickness_m = thickness_nm * 1.0e-9
    return capacitance_f * thickness_m / (EPS0 * area_m2)
