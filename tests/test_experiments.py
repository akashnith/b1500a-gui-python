"""Every experiment: builds, validates, analyses, draws."""

from __future__ import annotations

import math

import pytest

from b1500_studio.instruments.demo import DemoDriver
from b1500_studio.core.driver import Hardware
from b1500_studio.experiments import (
    EXPERIMENT_KEYS,
    all_experiments,
    get_experiment,
    params_from_fields,
)
from b1500_studio.experiments.base import eng, linfit
from b1500_studio.experiments.preview import draw_preview

# Shrunken parameters so the suite runs in seconds but exercises every path.
SMALL = {
    "synapse": {"syn_n_pot": "5", "syn_n_dep": "5", "syn_ret_n": "4",
                "syn_ret_t0": "100u", "syn_ret_t1": "1m"},
    "epsc": {"epsc_n_reads": "6", "epsc_t_last": "1m"},
    "stdp": {"stdp_dt_list": "lin(-20u,20u,5)", "stdp_t_rest": "10u"},
    "relaxation": {"rel_n_reads": "8", "rel_t_last": "100u", "rel_n_pre": "2"},
    "integrate_fire": {"if_periods": "log(4u,16u,3)", "if_n": "10"},
    "stochastic": {"stoch_v_list": "lin(0.45,0.9,3)", "stoch_trials": "10",
                   "stoch_t_rest": "10u"},
    "ppf": {"ppf_dt_list": "log(10u,1m,4)", "ppf_t_rest": "2m"},
    "refractory": {"ref_dt_list": "log(2u,200u,4)", "ref_t_rest": "1m"},
    "pund": {"pund_preset_delay": "1m", "pund_delay": "100u",
             "pund_samples": "10"},
    "pv": {"pv_points": "40", "pv_preset_cycles": "1"},
    "fe_iv": {"iv_points": "40"},
    "afe": {"afe_points": "40", "afe_preset_cycles": "1"},
    "ftj": {"ftj_steps": "9", "ftj_relax": "10u"},
    "fe_speed": {"speed_points": "4", "speed_samples": "12",
                 "speed_settle": "10u"},
    "fe_endurance": {"end_max_cycles": "100", "end_samples": "12",
                     "end_reads_per_decade": "1"},
    "fe_accumulation": {"acc_steps": "4", "acc_pulses_per_step": "4",
                        "acc_samples": "12"},
    "afe_pund": {"afep_samples": "12", "afep_relax": "10u"},
}

# Safety limit each experiment's DEFAULTS need, in volts.
#
# One volt does not fit a ferroelectric: its coercive voltage alone is about
# that, so a sub-coercive "test" would exercise the plumbing while measuring
# a device that never switches. These are the limits an operator would
# actually set for the protocol, not a number chosen to make a test pass.
VMAX = {
    "pund": 3.0, "pv": 5.0, "fe_iv": 5.0, "afe": 6.0, "ftj": 5.0,
    "fe_speed": 4.0, "fe_endurance": 4.0, "fe_accumulation": 4.0,
    "afe_pund": 5.0,
}


ALL = [e.key for e in all_experiments()]


@pytest.fixture
def hw():
    return Hardware()


def hw_for(key, hw):
    """``hw`` with the safety limit this experiment's defaults need."""
    hw.vmax = VMAX.get(key, hw.vmax)
    return hw


@pytest.fixture
def driver():
    return DemoDriver(realistic_delay=False)


def build(key, hw):
    exp = get_experiment(key)
    hw_for(key, hw)
    p = params_from_fields(exp.fields(), SMALL.get(key, {}))
    return exp, exp.build(p, hw)


class TestRegistry:
    def test_keys_match(self):
        assert sorted(ALL) == sorted(EXPERIMENT_KEYS)

    def test_unknown_key(self):
        with pytest.raises(KeyError):
            get_experiment("nope")

    @pytest.mark.parametrize("key", ALL)
    def test_fields_well_formed(self, key):
        exp = get_experiment(key)
        names = set()
        for f in exp.fields():
            assert {"name", "label", "default", "kind"} <= set(f)
            assert f["name"] not in names, "duplicate field"
            names.add(f["name"])
            if f["kind"] == "combo":
                assert 0 <= int(f["default"]) < len(f["choices"])

    @pytest.mark.parametrize("key", ALL)
    def test_defaults_parse(self, key):
        exp = get_experiment(key)
        p = params_from_fields(exp.fields(), {})
        for f in exp.fields():
            assert f["name"] in p.s


@pytest.mark.parametrize("key", ALL)
class TestEveryExperiment:
    def test_builds_and_validates(self, key, hw):
        exp, specs = build(key, hw)
        assert specs, "no runs produced"
        for s in specs:
            s.plan.validate(hw.vmax, hw.max_points)
            assert s.plan.total() > 0
            assert len(s.plan.ev) == len(s.ev_phase) == len(s.ev_measure_v)

    def test_measure_events_have_bias(self, key, hw):
        """Every read event records the bias it was read at.

        Events that measure along a ramp are exempt and say so: a swept bias
        has no single value, and inventing one would make the conductance
        computed from it wrong at every point but one.
        """
        _, specs = build(key, hw)
        for s in specs:
            for e, v in enumerate(s.ev_measure_v):
                if e < len(s.ev_swept) and s.ev_swept[e]:
                    continue
                # Finite, but zero is allowed: it is the origin of a pulsed
                # I-V curve, where the current is the measurement and
                # conductance is simply undefined.
                assert math.isfinite(v)

    def test_analyses(self, key, hw, driver):
        exp, specs = build(key, hw)
        for s in specs:
            R = exp.analyze(s, driver.run(s.plan, hw), hw)
            assert R.label
            assert R.summary, "no metrics produced"
            assert R.cols and R.data
            assert len(R.data) == len(R.cols)
            rows = len(R.data[0])
            assert all(len(c) == rows for c in R.data)

    def test_draws(self, key, hw, driver):
        exp, specs = build(key, hw)
        results = [exp.analyze(s, driver.run(s.plan, hw), hw) for s in specs]
        grid = exp.draw(results)
        assert grid.panels and grid.rows >= 1 and grid.cols >= 1
        assert any(p.series or p.text for p in grid.panels)

    def test_preview(self, key, hw):
        _, specs = build(key, hw)
        grid = draw_preview(specs, key)
        assert grid.panels[0].series


class TestPhysics:
    """The simulated device should behave the way the metrics assume."""

    def test_synapse_potentiates(self, hw, driver):
        exp, specs = build("synapse", hw)
        R = exp.analyze(specs[0], driver.run(specs[0].plan, hw), hw)
        metrics = dict(R.summary)
        assert metrics["LTP slope (S/pulse)"] > 0, "LTP should raise G"
        assert metrics["Gmax/Gmin"] > 1.0
        assert metrics["LTP linearity R2"] > 0.9

    def test_ppf_facilitates(self, hw, driver):
        exp, specs = build("ppf", hw)
        R = exp.analyze(specs[0], driver.run(specs[0].plan, hw), hw)
        assert dict(R.summary)["PPF max (%)"] > 100.0

    def test_stochastic_probability_bounded(self, hw, driver):
        exp, specs = build("stochastic", hw)
        for s in specs:
            R = exp.analyze(s, driver.run(s.plan, hw), hw)
            m = dict(R.summary)
            assert 0.0 <= m["firing probability"] <= 1.0
            assert m["Wilson 95% lower"] <= m["firing probability"] + 1e-9
            assert m["firing probability"] <= m["Wilson 95% upper"] + 1e-9

    def test_integrate_fire_spike_prob_bounded(self, hw, driver):
        exp, specs = build("integrate_fire", hw)
        for s in specs:
            R = exp.analyze(s, driver.run(s.plan, hw), hw)
            assert 0.0 <= dict(R.summary)["spike prob."] <= 1.0


class TestHelpers:
    @pytest.mark.parametrize(
        "value,unit,expected",
        [(1e-6, "s", "1 us"), (0, "V", "0 V"), (1500, "", "1.5 k"),
         (2.5e-3, "A", "2.5 mA")],
    )
    def test_eng(self, value, unit, expected):
        assert eng(value, unit) == expected

    def test_eng_non_finite(self):
        assert eng(float("nan")) == "n/a"

    def test_linfit_exact(self):
        L = linfit([0, 1, 2, 3], [1, 3, 5, 7])
        assert L.slope == pytest.approx(2.0)
        assert L.icpt == pytest.approx(1.0)
        assert L.r2 == pytest.approx(1.0)

    def test_linfit_too_short(self):
        assert math.isnan(linfit([1], [1]).slope)
