"""Dry-run verification: safety refusals and physics warnings."""

from __future__ import annotations

import pytest

from b1500_studio.core.driver import Hardware
from b1500_studio.experiments import get_experiment, params_from_fields
from b1500_studio.experiments.base import Severity
from b1500_studio.core.verification import verify_experiment

SMALL = {"syn_n_pot": "4", "syn_n_dep": "4", "syn_ret_n": "3",
         "syn_ret_t0": "100u", "syn_ret_t1": "1m"}


def params(**over):
    exp = get_experiment("synapse")
    merged = dict(SMALL)
    merged.update(over)
    return exp, params_from_fields(exp.fields(), merged)


def titles(res):
    return {i.title for i in res.issues}


class TestHardwareChecks:
    def test_defaults_pass(self):
        exp, p = params()
        res = verify_experiment(exp, p, Hardware())
        assert res.ok
        assert res.run_count == 1
        assert res.max_abs_v > 0

    def test_same_channel_rejected(self):
        exp, p = params()
        res = verify_experiment(exp, p, Hardware(force_ch=101, sink_ch=101))
        assert not res.ok and "Channel Conflict" in titles(res)

    def test_no_current_channel_rejected(self):
        exp, p = params()
        res = verify_experiment(exp, p, Hardware(force_meas="V", sink_meas="V"))
        assert not res.ok and "Measurement Channel" in titles(res)

    def test_edge_below_grid_rejected(self):
        exp, p = params()
        res = verify_experiment(exp, p, Hardware(edge=1e-12))
        assert not res.ok and "Timing Grid" in titles(res)

    def test_bad_settle_rejected(self):
        exp, p = params()
        assert not verify_experiment(exp, p, Hardware(settle=1.5)).ok

    def test_low_settle_warns(self):
        exp, p = params()
        res = verify_experiment(exp, p, Hardware(settle=0.01))
        assert res.ok and res.has_warnings
        assert "Low Settle Fraction" in titles(res)

    def test_bad_irange_rejected(self):
        exp, p = params()
        assert not verify_experiment(exp, p, Hardware(irange=9)).ok

    def test_no_experiment(self):
        _, p = params()
        res = verify_experiment(None, p, Hardware())
        assert not res.ok


class TestSafetyLimits:
    def test_over_voltage_refused(self):
        exp, p = params(syn_pot_v0="2.5")
        res = verify_experiment(exp, p, Hardware(vmax=1.0))
        assert not res.ok
        assert "Device Voltage Limit" in " ".join(titles(res))

    def test_raising_limit_allows_it(self):
        exp, p = params(syn_pot_v0="1.2", syn_dep_v0="-1.2")
        assert verify_experiment(exp, p, Hardware(vmax=2.0)).ok

    def test_build_failure_is_reported(self):
        exp, p = params(syn_n_pot="0", syn_n_dep="0")
        res = verify_experiment(exp, p, Hardware())
        assert not res.ok
        assert "Plan Construction Failed" in titles(res)


class TestPhysicsWarnings:
    def test_high_read_voltage_warns(self):
        exp, p = params(syn_v_read="0.5")
        res = verify_experiment(exp, p, Hardware())
        assert res.has_warnings and "Read Disturbance Risk" in titles(res)

    def test_low_read_voltage_warns(self):
        exp, p = params(syn_v_read="0.005")
        res = verify_experiment(exp, p, Hardware())
        assert res.has_warnings and "Low Read Voltage" in titles(res)

    def test_zero_read_voltage_rejected(self):
        exp, p = params(syn_v_read="0")
        assert not verify_experiment(exp, p, Hardware()).ok

    def test_unipolar_synapse_warns(self):
        exp, p = params(syn_pot_v0="0.7", syn_dep_v0="0.7")
        res = verify_experiment(exp, p, Hardware())
        assert "Unipolar Stimulus in Bipolar Synapse" in titles(res)

    def test_read_write_ratio_warns(self):
        exp, p = params(syn_v_read="0.28", syn_pot_v0="0.5", syn_dep_v0="-0.5")
        res = verify_experiment(exp, p, Hardware())
        assert any("Read/Write Ratio" in t for t in titles(res))

    def test_duty_cycle_check_runs(self):
        """if_periods is a list expression; the check must still fire."""
        exp = get_experiment("integrate_fire")
        p = params_from_fields(
            exp.fields(), {"if_w_p": "3u", "if_periods": "log(4u,16u,3)",
                           "if_n": "5"}
        )
        res = verify_experiment(exp, p, Hardware())
        assert any("Duty Cycle" in t for t in titles(res))


class TestSummary:
    def test_summary_mentions_state(self):
        exp, p = params()
        assert "Validated" in verify_experiment(exp, p, Hardware()).summary_text
        exp, p = params(syn_pot_v0="9")
        assert "Failed" in verify_experiment(exp, p, Hardware()).summary_text

    def test_severity_values(self):
        exp, p = params(syn_pot_v0="9")
        res = verify_experiment(exp, p, Hardware())
        assert all(
            i.severity in (Severity.INFO, Severity.WARNING, Severity.ERROR)
            for i in res.issues
        )
