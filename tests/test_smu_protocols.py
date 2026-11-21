"""The ``smu.*`` protocol family against the simulated memristor.

The demo SMU is a bipolar device with r_on = 2 kohm, r_off = 200 kohm and a
set/reset voltage of 0.6 V, so the numbers these assert against are
properties of that model, not arbitrary thresholds.
"""

from __future__ import annotations

import math

import pytest

from b1500_studio.instruments.demo import DemoSmu
from b1500_studio.protocols.smu import (
    dc_endurance,
    forming,
    ndr,
    probe_resistance,
    read_retention,
    slow_pulse_ltp_ltd,
    threshold_statistics,
)


@pytest.fixture
def smu():
    device = DemoSmu()
    device.open()
    return device


def metrics(run):
    return dict(run.metrics)


def test_forming_finds_a_forming_voltage_and_stops_there(smu):
    run = forming(smu, 1, v_start=0.4, v_stop=5.0, step=0.05, icomp=1e-3)
    m = metrics(run)
    assert math.isfinite(m["v_forming_V"])
    # It must stop ON the forming event, not sweep past it: forming is not
    # reversible, and continuing only drives a device that has changed.
    assert run.x[-1] == pytest.approx(m["v_forming_V"])
    assert any("formed" in note for note in run.notes)


def test_forming_says_so_when_nothing_forms(smu):
    # A compliance far above anything this device will draw in range.
    run = forming(smu, 1, v_start=0.0, v_stop=0.3, step=0.05, icomp=1.0)
    assert not math.isfinite(metrics(run)["v_forming_V"])
    assert any("did not form" in note for note in run.notes)


def test_settle_time_actually_biases_the_device(smu):
    """A held bias must change the state, or forming cannot be simulated.

    read_at used to pass dt = 0 to the device model, so the settle argument
    was accepted and ignored: a device could sit at 5 V and not move.
    """
    before, _ = smu.read_at(1, 0.1, 1e-3)
    smu.read_at(1, 1.5, 1e-3, settle=0.01)
    after, _ = smu.read_at(1, 0.1, 1e-3)
    assert abs(after) > 2 * abs(before)


def test_threshold_statistics_reports_a_spread(smu):
    run = threshold_statistics(smu, 1, v_max=2.0, step=0.02, icomp=1e-4,
                               trials=6)
    m = metrics(run)
    assert m["v_threshold_mean_V"] > 0.0
    assert m["v_threshold_sd_V"] >= 0.0
    assert m["v_threshold_min_V"] <= m["v_threshold_mean_V"] <= m["v_threshold_max_V"]


def test_ndr_refuses_to_call_noise_a_peak_to_valley_ratio(smu):
    """The demo memristor has no NDR, so none must be reported.

    A monotonic sweep with scatter on it will always contain a point lower
    than the one before, and reporting that as a PVR of 1.0003 is worse than
    reporting nothing.
    """
    run = ndr(smu, 1, v_max=0.5, step=0.02, icomp=1e-3)
    assert "pvr" not in metrics(run)
    assert any("No negative differential resistance" in n for n in run.notes)


def test_resistance_recovers_the_devices_off_state(smu):
    run = probe_resistance(smu, 1, v_read=0.1, repeats=8)
    m = metrics(run)
    # r_off of the simulated device is 200 kohm.
    assert 1.5e5 < m["R_mean_ohm"] < 2.5e5
    assert m["R_cv_percent"] < 10.0
    assert m["four_probe"] == 0.0


def test_read_retention_is_logarithmically_spaced(smu):
    run = read_retention(smu, 1, v_read=0.1, duration_s=0.05, points=6)
    assert len(run.x) == 6
    # Logarithmic: each gap is larger than the one before it.
    gaps = [b - a for a, b in zip(run.x, run.x[1:])]
    assert all(b > a for a, b in zip(gaps, gaps[1:]))
    assert "drift_percent" in metrics(run)


def test_slow_pulse_potentiates_then_depresses(smu):
    run = slow_pulse_ltp_ltd(smu, 1, v_pot=0.7, v_dep=-0.7, width_s=1e-4,
                             pulses=10, cycles=1, v_read=0.1)
    m = metrics(run)
    # The signs are the measurement: potentiation raises conductance,
    # depression lowers it.
    assert m["ltp_change_percent"] > 0.0
    assert m["ltd_change_percent"] < 0.0
    assert m["dynamic_range"] > 1.0


def test_dc_endurance_reports_a_window(smu):
    run = dc_endurance(smu, 1, v_set=1.0, v_reset=-1.0, width_s=1e-3,
                       cycles=10, read_every=2, v_read=0.1)
    m = metrics(run)
    assert m["window_first"] > 1.0
    assert m["window_min"] > 1.0
    # r_off / r_on is 100 for this device, and a 1 ms pulse is long enough to
    # switch it fully, so the window should land near that ratio.
    assert 10.0 < m["window_first"] < 1000.0


def test_endurance_window_grows_with_pulse_width(smu):
    """A partly-switched device has a smaller window, not a broken one.

    100 us only moves this device part of the way, so the two states sit
    closer together. Reporting that honestly is the point -- it is the
    difference between "the device is failing" and "the pulse is too short".
    """
    short = dc_endurance(smu, 1, v_set=1.0, v_reset=-1.0, width_s=1e-4,
                         cycles=6, read_every=2, v_read=0.1)
    smu._x = 0.0
    long = dc_endurance(smu, 1, v_set=1.0, v_reset=-1.0, width_s=1e-3,
                        cycles=6, read_every=2, v_read=0.1)
    assert metrics(long)["window_first"] > metrics(short)["window_first"]


def test_compliance_hits_are_counted_not_hidden(smu):
    run = forming(smu, 1, v_start=0.4, v_stop=5.0, step=0.05, icomp=1e-5)
    # The clipped points are flagged, because a clipped point is not a
    # measurement and anything computed across it is wrong.
    assert run.compliance_hits() >= 1
