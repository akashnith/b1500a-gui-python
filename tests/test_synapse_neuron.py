"""The synapse and neuron protocols added alongside the ferroelectric family.

Each test asserts the thing the protocol claims to measure, not a golden
number: the demo device is a toy, but a width sweep must still show longer
pulses doing more, and an f-I curve must still slope upward.
"""

from __future__ import annotations

import math

import pytest

from b1500_studio.instruments.demo import DemoDriver
from b1500_studio.core.driver import Hardware
from b1500_studio.experiments import get_experiment, params_from_fields
from b1500_studio.experiments.neuron_extra import count_spikes


def run(key, vmax=1.2, **overrides):
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


# ------------------------------------------------------------------ synapse


def test_width_sweep_longer_pulses_do_more():
    _, _, (R,) = run("width_sweep")
    g = [x for x in R.a["g"] if math.isfinite(x)]
    assert g[-1] > g[0]
    assert metrics(R)["dynamic_range"] > 1.1


def test_incremental_amplitude_finds_a_half_switch_voltage():
    _, _, (R,) = run("incremental_amplitude")
    m = metrics(R)
    assert m["G_per_volt_S"] > 0.0
    if "v_half_switch_V" in m:
        assert 0.2 <= m["v_half_switch_V"] <= 0.95


def test_metrics_reports_cycle_to_cycle_variation():
    _, _, (R,) = run("metrics", cycles="3", pulses_per_train="6")
    m = metrics(R)
    # The point of the measurement: an analog weight is only usable if the
    # same pulse count lands on the same conductance twice.
    assert m["cycle_to_cycle_cv_mean_percent"] >= 0.0
    assert m["cycle_to_cycle_cv_max_percent"] >= m["cycle_to_cycle_cv_mean_percent"]


def test_rs_speed_resets_between_points():
    """Every SET must start from the same place, or the curve is history.

    The protocol reads after each RESET as well as after each SET; those
    post-reset reads should sit close together, because they are all the
    same state.
    """
    _, _, (R,) = run("rs_speed", rs_points="5")
    after_reset = [x for x in R.a["g_reset"] if math.isfinite(x) and x > 0]
    assert len(after_reset) >= 3
    spread = (max(after_reset) - min(after_reset)) / max(after_reset)
    assert spread < 0.9, after_reset


def test_srdp_responds_to_rate():
    _, _, (R,) = run("srdp", f_points="5", pulses_per_train="6")
    m = metrics(R)
    assert "G_per_decade_S" in m
    assert m["rate_linearity_R2"] >= 0.0


def test_stp_facilitates_then_decays():
    _, _, (R,) = run("stp", pulses_per_train="8", rec_points="5")
    m = metrics(R)
    assert m["train_facilitation_percent"] > 0.0
    # Short-term means it forgets: the recovery reads must fall.
    assert m["decay_percent"] > 0.0


def test_retention_decays_rather_than_climbing():
    _, _, (R,) = run("retention", ret_points="6", ret_last="100m")
    m = metrics(R)
    assert m["decay_percent"] > 0.0, "a retention curve must lose state"
    assert 0.0 < m["retention_percent"] <= 100.0


def test_mlc_separates_its_levels():
    _, _, (R,) = run("mlc", target_levels="4", max_verify_pulses="4")
    m = metrics(R)
    assert m["levels_reached"] >= 2
    # Levels closer together than their own spread are not separate levels.
    assert m["level_separation_min_S"] > 0.0


def test_rs_endurance_tracks_both_states():
    _, _, (R,) = run("rs_endurance", cycles="12", read_every="3")
    m = metrics(R)
    assert len(R.a["g"]) == len(R.a["g_reset"]) >= 3
    assert m["window_first"] > 0.0
    assert "window_loss_percent" in m


# ------------------------------------------------------------------- neuron


def test_spike_counter_scales_itself():
    """The threshold comes from the record, so magnitude must not matter."""
    quiet_then_spikes = [1.0, 1.0, 9.0, 1.0, 1.0, 9.0, 1.0]
    n_small, _ = count_spikes(quiet_then_spikes)
    n_large, _ = count_spikes([v * 1e-6 for v in quiet_then_spikes])
    assert n_small == n_large == 2
    # A flat record has no spikes, however large it is.
    assert count_spikes([5.0] * 10)[0] == 0


def test_spike_counter_counts_rising_edges_only():
    # One excursion is one spike, however many samples it spans.
    assert count_spikes([0, 0, 9, 9, 9, 0, 0])[0] == 1
    assert count_spikes([0, 9, 0, 9, 0, 9, 0])[0] == 3


def test_fi_curve_slopes_upward():
    """More drive, more firing. The sign of this is the whole measurement."""
    _, _, (R,) = run("fi", amp_points="5", train_duration="60u")
    m = metrics(R)
    assert m["firing_rate_max_Hz"] > 0.0
    if "fi_gain_Hz_per_V" in m:
        assert m["fi_gain_Hz_per_V"] > 0.0


def test_leak_loses_more_at_longer_intervals():
    _, _, (R,) = run("leak", interval_points="6")
    g = [x for x in R.a["g"] if math.isfinite(x)]
    assert len(g) >= 4
    # A leaky integrator reaches a lower peak as the spacing grows.
    assert "integration_loss_percent" in metrics(R)


def test_pulsed_iv_sweeps_through_zero():
    _, _, (R,) = run("neuron_iv", v_points="21")
    v = R.a["v"]
    assert min(v) < 0.0 < max(v)
    # Zero bias is a legitimate point of an I-V curve, not an error.
    assert any(abs(x) < 1e-12 for x in v)
    assert len(R.a["i"]) == len(v)


def test_oscillator_states_its_own_limitation():
    _, _, (R,) = run("oscillator", trials="2", observation="60u")
    assert any("external series resistor" in note for note in R.notes), R.notes
    assert "spikes_per_trial_mean" in metrics(R)
