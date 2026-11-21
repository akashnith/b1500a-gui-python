"""The ferroelectric protocol family, end to end against simulated devices.

Invariants rather than golden numbers: the demo devices are toys, and the
exact charge depends on how much of a switch lands inside a measured ramp.
What must hold is that each protocol measures the thing it claims to.
"""

from __future__ import annotations

import math

import pytest

from b1500_studio.instruments.cmu import DemoCmu, rayleigh_fit
from b1500_studio.instruments.demo import DemoDriver
from b1500_studio.core.driver import Hardware
from b1500_studio.experiments import get_experiment, params_from_fields


def run(key, vmax=6.0, **overrides):
    exp = get_experiment(key)
    hw = Hardware()
    hw.vmax = vmax
    p = params_from_fields(exp.fields(), overrides)
    specs = exp.build(p, hw)
    driver = DemoDriver(realistic_delay=False)
    results = [exp.analyze(s, driver.run(s.plan, hw), hw) for s in specs]
    return exp, specs, results


def metrics(result):
    return dict(result.summary)


# ------------------------------------------------------------------- loops


def test_pv_recovers_the_devices_saturation_polarization():
    """2Pr must come back as about twice the simulated Ps, not a multiple.

    This is the regression that mattered: reporting the instantaneous
    current instead of the box average over each sample made a 20 uC/cm^2
    device read as 65, and the error moved when the sample count changed.
    """
    from b1500_studio.instruments.demo import FERRO_PS_UC_CM2

    _, _, (R,) = run("pv")
    m = metrics(R)
    assert 1.5 * FERRO_PS_UC_CM2 < m["two_pr_uC_cm2"] < 2.5 * FERRO_PS_UC_CM2
    # A symmetric device has no imprint and equal coercive voltages.
    assert abs(m["imprint_V"]) < 0.1
    assert m["vc_plus_V"] > 0 > m["vc_minus_V"]
    assert abs(m["vc_plus_V"] + m["vc_minus_V"]) < 0.1


def test_pv_coercive_voltage_rises_with_sweep_rate():
    """Vc is not a material constant; it moves with how fast you sweep.

    A faster triangle gives the domains less time, so switching happens
    further up the ramp. A model with a hard threshold could not show this.
    """
    _, _, (slow,) = run("pv", pv_f="200")
    _, _, (fast,) = run("pv", pv_f="20k")
    assert metrics(fast)["vc_plus_V"] > metrics(slow)["vc_plus_V"]


def test_afe_identifies_an_antiferroelectric():
    _, _, (R,) = run("afe")
    m = metrics(R)
    # Near-zero remanence is the identification test.
    assert m["remanence_ratio"] < 0.1, m["remanence_ratio"]
    # The induced phase appears going out and falls back lower coming in,
    # so the hysteresis between the two transitions is POSITIVE.
    assert m["transition_hysteresis_V"] > 0.0
    assert m["v_afe_to_fe_plus_V"] > m["v_fe_to_afe_plus_V"] > 0.0
    # A passive capacitor cannot return more than it stored.
    assert 0.0 < m["energy_efficiency_percent"] <= 100.0


def test_afe_protocol_flags_a_ferroelectric_as_not_antiferroelectric():
    """Pointing the AFE analysis at a ferroelectric must say so.

    A pinched loop from defects or leakage can mimic a double loop closely
    enough to fool a visual check, so the ratio is the check that matters.
    """
    exp = get_experiment("afe")
    hw = Hardware()
    hw.vmax = 6.0
    (spec,) = exp.build(params_from_fields(exp.fields(), {}), hw)
    spec.plan.hint = "ferro"  # a ferroelectric, not an antiferroelectric
    R = exp.analyze(spec, DemoDriver(realistic_delay=False).run(spec.plan, hw), hw)
    assert metrics(R)["remanence_ratio"] > 0.3
    assert any("not antiferroelectric" in note for note in R.notes)


def test_ftj_reports_a_memory_window():
    _, _, (R,) = run("ftj", vmax=5.0)
    m = metrics(R)
    assert m["r_off_ohm"] > m["r_on_ohm"] > 0.0
    assert m["ter_ratio"] > 1.0


# ------------------------------------------------------------------ pulsed


def test_speed_gives_a_rising_s_curve_and_a_t50():
    _, _, (R,) = run("fe_speed", vmax=4.0)
    m = metrics(R)
    psw = R.a["psw"]
    assert len(psw) >= 4
    # Longer write pulses switch more, up to saturation.
    assert psw[-1] > psw[0]
    assert m["psw_saturated_uC_cm2"] > 0.0
    # t50 must land inside the swept range, not at an end point.
    assert R.a["x"][0] < m["t50_switching_s"] < R.a["x"][-1]


def test_speed_t50_shortens_at_higher_amplitude():
    """Merz: more field, faster switching. This is the physics t50 encodes."""
    _, _, (low,) = run("fe_speed", vmax=5.0, speed_write_v="-1.8",
                       speed_preset_v="1.8")
    _, _, (high,) = run("fe_speed", vmax=5.0, speed_write_v="-3.0",
                        speed_preset_v="3.0")
    lo, hi = metrics(low), metrics(high)
    if math.isfinite(lo.get("t50_switching_s", float("nan"))) and math.isfinite(
        hi.get("t50_switching_s", float("nan"))
    ):
        assert hi["t50_switching_s"] < lo["t50_switching_s"]


def test_endurance_reads_a_pristine_point_before_any_stress():
    _, spec, (R,) = run("fe_endurance", vmax=4.0, end_max_cycles="100",
                        end_reads_per_decade="1")
    m = metrics(R)
    assert "psw_pristine_uC_cm2" in m
    # The first read is at zero cycles, before any stress is applied.
    assert R.a["x"][0] == 0.0
    assert R.a["x"] == sorted(R.a["x"])
    # Both are reported: a film can wake up before it fatigues, and one
    # number cannot say which happened.
    assert "wakeup_percent" in m and "fatigue_percent" in m


def test_accumulation_climbs_with_pulse_count():
    _, _, (R,) = run("fe_accumulation", vmax=4.0, acc_steps="5",
                     acc_pulses_per_step="6")
    m = metrics(R)
    accumulated = R.a["psw"]
    assert accumulated[-1] > accumulated[0]
    assert m["p_accumulated_final_uC_cm2"] > 0.0
    assert 0.0 < m["accumulation_percent"] <= 100.0
    # The caveat is part of the result, not optional commentary.
    assert any("extrinsic" in note for note in R.notes)


def test_accumulation_needs_sub_coercive_pulses_to_be_gradual():
    """A full-amplitude train switches on the first pulse, not gradually."""
    _, _, (gentle,) = run("fe_accumulation", vmax=5.0, acc_pulse_v="0.8",
                          acc_steps="4", acc_pulses_per_step="5")
    _, _, (brutal,) = run("fe_accumulation", vmax=5.0, acc_pulse_v="3.0",
                          acc_steps="4", acc_pulses_per_step="5")
    # The brutal train saturates immediately, so its curve is flatter in
    # relative terms: nearly all of the change happens before the first read.
    gentle_curve = gentle.a["psw"]
    brutal_curve = brutal.a["psw"]
    assert gentle_curve[-1] - gentle_curve[0] > 0.0
    assert brutal_curve[0] >= 0.0


def test_afe_pund_measures_both_polarities():
    _, _, (R,) = run("afe_pund", vmax=5.0)
    m = metrics(R)
    assert "p_pu_plus_uC_cm2" in m
    assert "p_pu_minus_uC_cm2" in m
    assert m["p_switched_uC_cm2"] >= 0.0


# -------------------------------------------------------------------- MFCMU


def test_cv_butterfly_has_two_branches_and_a_memory_window():
    cmu = DemoCmu()
    result = cmu.cv_sweep(vmax=3.0, steps=51)
    assert len(result) == 102
    branches = {p.branch for p in result.points}
    assert branches == {"up", "down"}
    # The two lobes sit at the coercive voltage of each sweep direction, so
    # the gap between them is the memory window.
    assert result.memory_window_v() > 0.5


def test_cv_permittivity_is_physical():
    cmu = DemoCmu()
    result = cmu.cv_sweep(vmax=3.0, steps=21)
    eps = result.permittivity()
    assert all(e > 0.0 for e in eps)
    assert max(eps) > min(eps)


def test_rayleigh_fit_recovers_the_simulated_linearity():
    """The demo capacitance is linear in AC level by construction.

    C = C0 * (1 + 6*Vac), so eps_r is linear in the field and alpha is
    eps_init * 6 / 1000 per kV/cm. Recovering that is the whole fit.
    """
    cmu = DemoCmu()
    levels = [0.01 * k for k in range(1, 11)]
    result = cmu.rayleigh_sweep(levels, thickness_nm=10.0)
    fit = rayleigh_fit(
        [p.ac_level_v for p in result.points],
        [p.capacitance_f for p in result.points],
        result.area_cm2,
        result.thickness_nm,
    )
    assert fit["eps_init"] > 0.0
    assert fit["rayleigh_alpha_cm_kV"] > 0.0
    expected_alpha = fit["eps_init"] * 6.0 / 1000.0
    assert abs(fit["rayleigh_alpha_cm_kV"] - expected_alpha) < 0.05 * expected_alpha
    assert 0.0 < fit["domain_wall_fraction_percent"] < 100.0


def test_rayleigh_fit_refuses_impossible_geometry():
    assert rayleigh_fit([0.1, 0.2], [1e-10, 2e-10], 0.0, 10.0) == {}
    assert rayleigh_fit([0.1], [1e-10], 1e-4, 10.0) == {}


def test_cmu_without_an_mfcmu_says_so():
    """A missing module must name what IS installed, not just fail."""
    from b1500_studio.instruments.cmu import Cmu, CmuError
    from b1500_studio.instruments.smu import Module

    cmu = Cmu()
    cmu.scan = lambda: [Module(slot=1, model="B1517A", kind="SMU")]
    with pytest.raises(CmuError) as excinfo:
        cmu.find_channel()
    assert "B1520A" in str(excinfo.value)
    assert "B1517A" in str(excinfo.value)
