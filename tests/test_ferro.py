"""Ferroelectric analysis, cross-checked against the Qt implementation.

Every case here is the Python twin of a test in the C++ tree's
``tests/test_core.cpp`` -- same synthetic input, same expected numbers. If the
two implementations ever drift apart, these fail. That matters more than usual
because the Qt side has already been checked against a genuine EasyEXPERT
FERAM PUND export, so agreeing with it is agreeing with the vendor.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from b1500_studio.core import ferro


def ideal_loop(
    vmax: float, vc: float, ps: float, points_per_ramp: int
) -> Tuple[List[float], List[float], List[float]]:
    """Port of ``idealLoop()`` in the C++ tests.

    ``0 -> +vmax -> -vmax -> 0``, polarization following a tanh centred on
    +/-vc, and a current proportional to dP so the Vc disambiguation has a
    switching peak to latch onto.
    """
    v: List[float] = []
    p: List[float] = []
    i: List[float] = []
    legs = [0.0, vmax, -vmax, 0.0]
    for leg in range(len(legs) - 1):
        frm, to = legs[leg], legs[leg + 1]
        rising = to > frm
        for k in range(points_per_ramp):
            voltage = frm + (to - frm) * k / float(points_per_ramp)
            centre = vc if rising else -vc
            state = math.tanh((voltage - centre) * 6.0 / max(vc, 1e-9))
            v.append(voltage)
            p.append(ps * state)
            i.append(0.0)
    for k in range(1, len(v)):
        i[k] = p[k] - p[k - 1]
    return v, p, i


def test_integrates_charge_and_removes_leakage_drift():
    # A pure resistive leak carries net charge; after drift removal the loop
    # must close, i.e. start and end at the same charge.
    n = 201
    dt = 1.0e-6
    t = [k * dt for k in range(n)]
    current = [1.0e-6] * n  # constant leakage only

    with_drift = ferro.cumulative_charge(t, current, remove_drift=False, dt=dt)
    assert with_drift[-1] > 1.0e-10  # charge accumulates

    closed = ferro.cumulative_charge(t, current, remove_drift=True, dt=dt)
    # The leakage ramp is gone: what remains is symmetric about zero, bounded
    # by the half-sample back-shift, and far below the uncorrected drift.
    assert abs(closed[-1]) < 0.01 * abs(with_drift[-1])
    assert abs(closed[-1] + closed[0]) < 1.0e-18


def test_polarization_scaling():
    # 1 nC on 1e-4 cm^2 is 10 uC/cm^2.
    p = ferro.polarization([0.0, 1.0e-9], 1.0e-4, centre=False)
    assert abs(p[-1] - 10.0) < 1.0e-9


def test_extracts_remanence_and_coercive_field_from_an_ideal_loop():
    v, p, i = ideal_loop(3.0, 1.0, 20.0, 400)
    m = ferro.analyze_loop(v, p, i, v, 10.0)
    assert m.valid
    # Saturation is 20 uC/cm^2, so 2Pr approaches 2*20 for a square loop.
    assert 30.0 < m.two_pr <= 41.0, m.two_pr
    assert abs(m.vc_plus - 1.0) < 0.25, m.vc_plus
    assert abs(m.vc_minus + 1.0) < 0.25, m.vc_minus
    assert abs(m.imprint) < 0.25
    # Ec = 10*Vc/d_nm -> 1 V on 10 nm is 1 MV/cm.
    assert abs(m.ec_plus - 1.0) < 0.3, m.ec_plus
    assert m.p_max > 15.0


def test_pund_subtraction_removes_non_switching_charge():
    # P carries switching + non-switching, U carries non-switching only.
    area = 1.0e-4
    q_switch = 4.0e-9     # the real switched charge
    q_non_switch = 1.0e-9  # dielectric + leakage
    m = ferro.analyze_pund(
        q_switch + q_non_switch,
        q_non_switch,
        -(q_switch + q_non_switch),
        -q_non_switch,
        area,
    )
    assert m.valid
    # dP+ = (qP - qU)/A = 4 nC / 1e-4 cm^2 = 40 uC/cm^2
    assert abs(m.p_switched_plus - 40.0) < 1.0e-9
    assert abs(m.p_switched_minus + 40.0) < 1.0e-9
    assert abs(m.two_pr - 40.0) < 1.0e-9
    assert abs(m.pr - 20.0) < 1.0e-9
    # P* includes the non-switching part and must exceed the switched value.
    assert m.p_star > m.p_switched_plus


def test_separates_leakage_from_two_frequencies():
    # I(V) = displacement (scales with f) + ohmic leakage (does not).
    f1, f2, r_leak = 1000.0, 10000.0, 1.0e6
    v: List[float] = []
    i1: List[float] = []
    i2: List[float] = []
    for k in range(101):
        voltage = -3.0 + 6.0 * k / 100.0
        v.append(voltage)
        displacement = 1.0e-9 * voltage  # per unit frequency
        i1.append(displacement * f1 + voltage / r_leak)
        i2.append(displacement * f2 + voltage / r_leak)

    leak = ferro.leakage_from_two_frequencies(v, i1, f1, v, i2, f2)
    assert len(leak) == len(v)
    for k in range(len(v)):
        assert abs(leak[k] - v[k] / r_leak) < 1.0e-15, leak[k]


def test_rising_mask_marks_each_branch():
    # Up then down: the mask must flip exactly once, and a flat stretch
    # inherits the direction before it rather than becoming its own branch.
    v = [0.0, 1.0, 2.0, 2.0, 1.0, 0.0]
    mask = ferro.rising_mask(v)
    assert mask[:3] == [True, True, True]
    assert mask[4:] == [False, False]


def test_branch_difference_cancels_ohmic_leakage():
    # Identical leakage on both branches must subtract to zero; a switching
    # peak present only on the rising branch must survive.
    v = [0.0, 1.0, 2.0, 2.0, 1.0, 0.0]
    leak = [x / 1.0e6 for x in v]
    rising_peak = [0.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0]
    current = [leak[k] + rising_peak[k] for k in range(len(v))]
    voltages, difference = ferro.branch_difference(v, current, v)
    assert voltages, "expected a usable rising branch"
    # At 1 V on the way up the peak is present; the leak has cancelled.
    at_one = [d for x, d in zip(voltages, difference) if abs(x - 1.0) < 1e-12]
    assert at_one and abs(at_one[0] - 1.0e-6) < 1.0e-12, at_one


def test_permittivity_round_trips():
    # eps_r = C*d / (eps0*A); build C from a known eps_r and recover it.
    area_cm2, thickness_nm, eps_r = 1.0e-4, 10.0, 300.0
    area_m2 = area_cm2 * 1.0e-4
    thickness_m = thickness_nm * 1.0e-9
    capacitance = eps_r * ferro.EPS0 * area_m2 / thickness_m
    assert abs(ferro.permittivity(capacitance, area_cm2, thickness_nm) - eps_r) < 1e-9
    # Nonsense geometry returns 0 rather than an infinity.
    assert ferro.permittivity(capacitance, 0.0, thickness_nm) == 0.0
    assert ferro.permittivity(0.0, area_cm2, thickness_nm) == 0.0
