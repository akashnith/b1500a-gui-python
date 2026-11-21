"""The PUND experiment, end to end against the simulated ferroelectric.

These are invariants, not golden numbers: the demo device is a toy and its
exact charge depends on how much of the switch lands inside the measured ramp.
What must hold is the *structure* of a PUND -- that P switches and U does not,
that the two polarities are symmetric, and that the flat top is excluded.
"""

from __future__ import annotations

import math

from b1500_studio.instruments.demo import simulate
from b1500_studio.core.driver import Hardware
from b1500_studio.experiments.ferroelectric import (
    PH_D,
    PH_N,
    PH_P,
    PH_U,
    RAMP_FALL,
    RAMP_RISE,
    FePund,
)
from b1500_studio.experiments.registry import get_experiment, params_from_fields


def run_pund(**overrides):
    exp = FePund()
    p = params_from_fields(exp.fields(), overrides)
    hw = Hardware()
    hw.vmax = 5.0
    (spec,) = exp.build(p, hw)
    res = simulate(spec.plan, hw)
    return exp, spec, exp.analyze(spec, res, hw)


def metrics(result):
    return dict(result.summary)


def test_pund_is_registered():
    assert get_experiment("pund").key == "pund"


def test_plan_has_five_pulses_and_measures_only_four():
    _, spec, _ = run_pund()
    # preset, P, U, N, D -- and the preset is deliberately not measured, so
    # eight measured ramps remain (rise and fall of each of the four).
    assert len(spec.plan.ev) == 8
    phases = {(spec.ev_phase[i], spec.ev_idx[i]) for i in range(len(spec.plan.ev))}
    for phase in (PH_P, PH_U, PH_N, PH_D):
        assert (phase, RAMP_RISE) in phases
        assert (phase, RAMP_FALL) in phases
    # Every measured event is marked swept: a ramp has no single read bias.
    assert all(spec.ev_swept)


def test_flat_top_is_not_measured():
    """No sample may fall inside a flat top.

    This is the whole measurement. Integrate the top as well and what gets
    added is leakage times the top width -- which looks like signal and scales
    with a parameter that should not matter.
    """
    _, spec, _ = run_pund(pund_w="4u", pund_edge="1u", pund_samples="20")
    edge_ticks = 100  # 1 us on the 10 ns grid
    for e in spec.plan.ev:
        span = (e.points - 1) * e.interval + e.avg
        assert span <= edge_ticks, f"event {e.name} spans {span} ticks > one ramp"


def test_p_switches_and_u_does_not():
    _, _, R = run_pund()
    m = metrics(R)
    # P carries switching + non-switching charge, U only non-switching.
    #
    # The bound is 2x rather than something tighter because the separation is
    # genuinely partial: at the default 2 V with a 1 us ramp the device has a
    # switching time of about 500 ns, so a few percent is still switching when
    # U arrives. That is a property of the device and the ramp, not a defect
    # -- raising the amplitude or slowing the ramp widens the gap.
    assert m["charge_p_C"] > 2 * m["charge_u_C"] > 0
    assert m["p_star_uC_cm2"] > m["p_hat_uC_cm2"] > 0
    # The negative half mirrors it.
    assert m["charge_n_C"] < 2 * m["charge_d_C"] < 0


def test_both_polarities_switch_by_the_same_amount():
    _, _, R = run_pund()
    m = metrics(R)
    plus = m["dp_plus_uC_cm2"]
    minus = m["dp_minus_uC_cm2"]
    assert plus > 1.0, plus
    assert minus < -1.0, minus
    # A symmetric simulated device: the two must agree to a few percent.
    assert abs(plus + minus) < 0.05 * abs(plus), (plus, minus)
    assert math.isclose(m["two_pr_uC_cm2"], 2 * m["pr_uC_cm2"], rel_tol=1e-9)


def test_falling_ramps_carry_no_switching():
    """The 'a' charges are the trailing ramp, reported separately.

    Coming back down from +Vmax to 0 the device stays switched, so P and U
    must agree there. If they did not, the preset/P/U sequence would not be
    doing what it claims.
    """
    _, _, R = run_pund()
    m = metrics(R)
    assert abs(m["charge_pa_C"] - m["charge_ua_C"]) < 0.1 * abs(m["charge_p_C"])


def test_psw_tracks_amplitude():
    """More drive switches more of the film, up to saturation."""
    _, _, low = run_pund(pund_v="1.2")
    _, _, high = run_pund(pund_v="3.0")
    assert metrics(high)["psw_uC_cm2"] > metrics(low)["psw_uC_cm2"]


def test_table_and_loop_arrays_are_populated():
    _, spec, R = run_pund()
    assert "charge_C" in R.cols
    assert len(R.data) == len(R.cols)
    n = spec.plan.n_points()
    assert len(R.data[0]) == n
    # The subtracted P-E loop covers all four measured ramps.
    assert len(R.a["loop_drive"]) == len(R.a["loop_p"]) > 0
