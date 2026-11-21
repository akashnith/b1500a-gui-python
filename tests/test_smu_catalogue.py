"""The SMU protocols as the user interface sees them, and the DC staircase.

The staircase tests are the important ones here: an asymmetric bipolar sweep
is the measurement the whole ``smu.iv`` protocol exists for, and getting a
limb's sign or step count wrong puts the wrong polarity, or the wrong
resolution, on a real device.
"""

from __future__ import annotations

import math

import pytest

from b1500_studio.core.fields import params_from_fields
from b1500_studio.instruments.demo import DemoSmu
from b1500_studio.protocols import (
    SMU_PROTOCOL_KEYS,
    SmuSettings,
    all_smu_protocols,
    get_smu_protocol,
)
from b1500_studio.protocols.base import current_range_code, sweep_step_count
from b1500_studio.protocols.smu import export_smu_csv
from b1500_studio.protocols.sweep import staircase, staircase_voltages

ALL = list(SMU_PROTOCOL_KEYS)


def settings(**kw) -> SmuSettings:
    base = dict(force_ch=1, common_ch=2, vmax=6.0)
    base.update(kw)
    return SmuSettings(**base)


def params_for(key, **overrides):
    protocol = get_smu_protocol(key)
    return protocol, params_from_fields(protocol.fields(), overrides)


# ------------------------------------------------------------- the staircase


class TestStaircase:
    def test_single_mode_goes_out_and_stops(self):
        legs = staircase(2.0, 2.0, 0.5, mode=0, positive_first=True)
        assert len(legs) == 1
        assert (legs[0].start, legs[0].stop) == (0.0, 2.0)
        assert legs[0].steps == 4

    def test_bipolar_returns_through_zero_both_times(self):
        legs = staircase(1.0, 1.0, 0.5, mode=2, positive_first=True)
        assert [(l.start, l.stop) for l in legs] == [
            (0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)
        ]

    def test_dual_crosses_straight_over(self):
        legs = staircase(1.0, 1.0, 0.5, mode=1, positive_first=True)
        assert [(l.start, l.stop) for l in legs] == [
            (0.0, 1.0), (1.0, -1.0), (-1.0, 0.0)
        ]

    def test_limbs_are_magnitudes_and_the_leg_supplies_the_sign(self):
        """V- max is entered positive; the negative limb still goes negative.

        A signed field here would mean "V- max = -2" sweeping to +2, which is
        the double negative that puts the wrong polarity on a device.
        """
        legs = staircase(2.0, 4.0, 1.0, mode=2, positive_first=True)
        peaks = [l.stop for l in legs]
        assert max(peaks) == pytest.approx(2.0)
        assert min(peaks) == pytest.approx(-4.0)

    def test_negative_first_swaps_which_limb_leads(self):
        legs = staircase(2.0, 4.0, 1.0, mode=2, positive_first=False)
        assert legs[0].stop == pytest.approx(-4.0)
        assert any(l.stop == pytest.approx(2.0) for l in legs)

    def test_asymmetric_limbs_keep_one_step_size(self):
        """The whole point of a per-limb step count.

        A single points-per-leg setting would give the shorter limb a finer
        step, so the same transition would be resolved differently depending
        on which direction it was approached from.
        """
        legs = staircase(1.0, 4.0, 0.25, mode=2, positive_first=True)
        for leg in legs:
            assert leg.step_size() == pytest.approx(0.25)
        assert legs[0].steps == 4      # 1 V / 0.25
        assert legs[2].steps == 16     # 4 V / 0.25

    def test_step_count_does_not_duplicate_the_peak(self):
        """2.0 / 0.1 is 19.999999999999996 in binary floating point.

        Without the epsilon it rounds up to 21 points and every sweep gets a
        duplicate point at its peak.
        """
        assert sweep_step_count(2.0, 0.1) == 20
        assert sweep_step_count(1.5, 0.02) == 75
        assert sweep_step_count(0.0, 0.1) == 1
        assert sweep_step_count(1.0, 0.0) == 1

    def test_later_cycles_do_not_remeasure_the_zero_they_start_on(self):
        one = staircase_voltages(staircase(1.0, 1.0, 0.5, 2, True, cycles=1))
        two = staircase_voltages(staircase(1.0, 1.0, 0.5, 2, True, cycles=2))
        # The second cycle adds its own points but not a second copy of the
        # zero the first one ended at.
        assert len(two) == 2 * len(one) - 1

    def test_the_sweep_actually_reaches_both_peaks(self):
        volts = staircase_voltages(staircase(2.0, 3.0, 0.5, 2, True))
        assert max(volts) == pytest.approx(2.0)
        assert min(volts) == pytest.approx(-3.0)


# ----------------------------------------------------------------- ranging


class TestRanging:
    def test_a_bad_index_falls_back_to_auto_not_to_a_fixed_range(self):
        """A fixed range below the real current reads as saturated.

        Saturated looks like a measurement rather than like an error, so an
        out-of-range choice must never land on one.
        """
        assert current_range_code(-1) == 0
        assert current_range_code(999) == 0

    def test_known_codes_match_the_flex_table(self):
        assert current_range_code(0) == 0      # auto, full range
        assert current_range_code(6) == 16     # auto, min 100 uA
        assert current_range_code(16) == -18   # fixed 10 mA


# --------------------------------------------------------------- catalogue


class TestCatalogue:
    def test_keys_match_the_registry(self):
        assert [p.key for p in all_smu_protocols()] == ALL

    def test_lookup_accepts_the_short_form(self):
        assert get_smu_protocol("iv").key == "smu.iv"
        assert get_smu_protocol("smu.iv").key == "smu.iv"
        with pytest.raises(KeyError):
            get_smu_protocol("nope")

    @pytest.mark.parametrize("key", ALL)
    def test_fields_are_well_formed_and_parse(self, key):
        protocol, p = params_for(key)
        names = set()
        for f in protocol.fields():
            assert {"name", "label", "default", "kind"} <= set(f)
            assert f["name"] not in names, f"duplicate field {f['name']}"
            names.add(f["name"])
            if f["kind"] == "combo":
                assert 0 <= int(f["default"]) < len(f["choices"])
            assert f["name"] in p.s

    @pytest.mark.parametrize("key", ALL)
    def test_every_protocol_declares_compliance_and_integration(self, key):
        """Compliance is the real protection, so no protocol may omit it."""
        _protocol, p = params_for(key)
        assert p.num("icomp") > 0.0
        assert p.num("adc_plc") >= 1.0

    @pytest.mark.parametrize("key", ALL)
    def test_runs_against_the_simulator_and_draws(self, key):
        protocol, p = params_for(key)
        smu = DemoSmu()
        smu.open()
        try:
            runs = protocol.run(smu, p, settings())
        finally:
            smu.close()
        assert runs and runs[0].x, f"{key} measured nothing"
        assert len(runs[0].x) == len(runs[0].y)
        grid = protocol.draw(runs)
        assert grid.panels
        assert any(panel.series or panel.text for panel in grid.panels)

    @pytest.mark.parametrize("key", ALL)
    def test_previews_without_an_instrument(self, key):
        protocol, p = params_for(key)
        grid = protocol.preview(p, settings())
        assert grid.panels[0].series or grid.panels[0].text

    @pytest.mark.parametrize("key", ALL)
    def test_the_voltage_ceiling_is_enforced_before_anything_opens(self, key):
        """Verify must refuse on a limit, with no session and no device."""
        protocol, p = params_for(key)
        ok, message = protocol.verify(p, settings(vmax=0.01))
        assert not ok, key
        assert "safety limit" in message

    def test_a_shared_force_and_return_channel_is_refused(self):
        protocol, p = params_for("smu.read")
        ok, message = protocol.verify(p, settings(force_ch=3, common_ch=3))
        assert not ok
        assert "same channel" in message


# ------------------------------------------------------------------ smu.iv


class TestSmuIv:
    def test_the_sweep_measures_every_planned_point_once(self):
        protocol, p = params_for("smu.iv", sweep_v_pos_v="1",
                                 sweep_v_neg_v="1", sweep_step_v="0.25")
        planned = len(staircase_voltages(protocol.legs(p)))
        smu = DemoSmu()
        smu.open()
        try:
            runs = protocol.run(smu, p, settings())
        finally:
            smu.close()
        assert len(runs[0].x) == planned

    def test_a_switching_device_gives_an_on_off_ratio(self):
        """The loop passes the read voltage twice in different states.

        That is the whole hysteresis -- same bias, different current -- so
        the ratio is read off the sweep rather than measured separately.
        """
        protocol, p = params_for("smu.iv", sweep_v_pos_v="1.2",
                                 sweep_v_neg_v="1.2", sweep_step_v="0.05",
                                 read_voltage_v="0.2")
        smu = DemoSmu()
        smu.open()
        try:
            runs = protocol.run(smu, p, settings())
        finally:
            smu.close()
        m = dict(runs[0].metrics)
        assert m["on_off_ratio"] > 2.0
        assert m["i_on_A"] > m["i_off_A"] > 0.0
        assert m["r_off_ohm"] > m["r_on_ohm"] > 0.0

    def test_a_single_sweep_refuses_to_report_a_ratio(self):
        """One pass gives a number divided by itself, which is always 1.

        Reporting that reads as a definite result rather than as missing
        data, so the protocol says why it has none instead.
        """
        protocol, p = params_for("smu.iv", sweep_mode="0",
                                 sweep_v_pos_v="1", sweep_step_v="0.1",
                                 read_voltage_v="0.5")
        smu = DemoSmu()
        smu.open()
        try:
            runs = protocol.run(smu, p, settings())
        finally:
            smu.close()
        assert "on_off_ratio" not in dict(runs[0].metrics)
        assert any("fewer than twice" in n for n in runs[0].notes)

    def test_each_leg_is_its_own_curve(self):
        protocol, p = params_for("smu.iv", sweep_step_v="0.5")
        smu = DemoSmu()
        smu.open()
        try:
            runs = protocol.run(smu, p, settings())
        finally:
            smu.close()
        assert len(set(runs[0].phase)) == len(protocol.legs(p))
        grid = protocol.draw(runs)
        assert len(grid.panels[0].series) == len(protocol.legs(p))

    def test_the_log_axis_choice_reaches_the_figure(self):
        protocol, p = params_for("smu.iv", sweep_step_v="0.5",
                                 log_i_axis="Log |I|")
        smu = DemoSmu()
        smu.open()
        try:
            runs = protocol.run(smu, p, settings())
        finally:
            smu.close()
        grid = protocol.draw_with(runs, p)
        iv = grid.panels[0]
        assert iv.logy
        # A log axis cannot carry a negative number, so the sign moves to
        # the (still linear) voltage axis.
        assert all(y > 0 for s in iv.series for y in s.y if y == y)

    def test_preview_counts_the_points_the_run_will_measure(self):
        protocol, p = params_for("smu.iv", sweep_step_v="0.2")
        grid = protocol.preview(p, settings())
        drawn = grid.panels[0].series[0]
        assert len(drawn.y) == len(staircase_voltages(protocol.legs(p)))


# ------------------------------------------------------------------ export


def test_export_writes_the_metrics_beside_the_data(tmp_path):
    protocol, p = params_for("smu.iv", sweep_step_v="0.5")
    smu = DemoSmu()
    smu.open()
    try:
        runs = protocol.run(smu, p, settings())
    finally:
        smu.close()
    paths = export_smu_csv(runs, str(tmp_path), "smu_iv")
    assert len(paths) == 1
    text = open(paths[0], encoding="utf-8").read()
    # A figure of merit separated from the data it came from is one nobody
    # can check, so it travels in the same file.
    assert "# on_off_ratio = " in text
    assert "Voltage (V)" in text
    assert len(text.splitlines()) > len(runs[0].x)
