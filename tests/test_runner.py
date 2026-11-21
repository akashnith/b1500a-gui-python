"""Runner orchestration, CSV export, drivers and the hang-proof session."""

from __future__ import annotations

import csv
import os

import pytest

from b1500_studio import make_driver
from b1500_studio.instruments.demo import DemoDriver, simulate
from b1500_studio.core.driver import Hardware, Results, irange_amps
from b1500_studio.experiments import get_experiment, params_from_fields
from b1500_studio.core.runner import Runner, export_csv, export_results

SMALL = {"syn_n_pot": "4", "syn_n_dep": "4", "syn_ret_n": "3",
         "syn_ret_t0": "100u", "syn_ret_t1": "1m"}


@pytest.fixture
def synapse():
    exp = get_experiment("synapse")
    return exp, params_from_fields(exp.fields(), SMALL)


@pytest.fixture
def hw():
    return Hardware()


class TestDriverBasics:
    def test_irange_clamped(self):
        assert irange_amps(-5) == 1e-6
        assert irange_amps(99) == 1e-2
        assert irange_amps(2) == 1e-4

    def test_demo_channels(self):
        assert DemoDriver().channels() == [101, 102]

    def test_results_current_requires_an_i_channel(self, hw):
        with pytest.raises(Exception):
            Results().current(Hardware(force_meas="V", sink_meas="V"))

    def test_make_driver_factory(self):
        assert make_driver("demo").name == "DEMO"
        with pytest.raises(ValueError):
            make_driver("nonsense")

    def test_simulate_respects_current_range(self, synapse, hw):
        exp, p = synapse
        spec = exp.build(p, hw)[0]
        low = Hardware(irange=0)  # 1 uA full scale
        res = simulate(spec.plan, low)
        assert all(abs(v) <= irange_amps(0) + 1e-18 for v in res.current(low))


class TestRunner:
    def test_runs_all_specs(self, synapse, hw):
        exp, p = synapse
        report = Runner(DemoDriver(realistic_delay=False), hw).run(exp, p)
        assert report.ok
        assert len(report.results) == len(report.specs) == 1
        assert report.driver == "DEMO"
        assert report.experiment and report.started

    def test_log_is_captured(self, synapse, hw):
        exp, p = synapse
        lines = []
        report = Runner(DemoDriver(realistic_delay=False), hw,
                        log=lines.append).run(exp, p)
        assert lines and report.log
        assert any("run 1/1" in line for line in lines)

    def test_progress_callback(self, synapse, hw):
        exp, p = synapse
        seen = []
        Runner(DemoDriver(realistic_delay=False), hw,
               progress=lambda i, n, label: seen.append((i, n))).run(exp, p)
        assert seen and seen[-1][0] == seen[-1][1]

    def test_multi_run_experiment(self, hw):
        exp = get_experiment("integrate_fire")
        p = params_from_fields(
            exp.fields(), {"if_periods": "log(4u,16u,3)", "if_n": "6"}
        )
        report = Runner(DemoDriver(realistic_delay=False), hw).run(exp, p)
        assert len(report.results) == 3

    def test_build_error_is_reported(self, hw):
        exp = get_experiment("synapse")
        p = params_from_fields(exp.fields(), {"syn_n_pot": "0", "syn_n_dep": "0"})
        report = Runner(DemoDriver(realistic_delay=False), hw).run(exp, p)
        assert not report.ok and "plan construction failed" in report.error

    def test_voltage_limit_stops_the_run(self, hw):
        exp = get_experiment("synapse")
        p = params_from_fields(exp.fields(), dict(SMALL, syn_pot_v0="5"))
        report = Runner(DemoDriver(realistic_delay=False),
                        Hardware(vmax=1.0)).run(exp, p)
        assert not report.ok and "exceeds the safety limit" in report.error

    def test_abort_before_start(self, synapse, hw):
        exp, p = synapse
        runner = Runner(DemoDriver(realistic_delay=False), hw)
        runner.abort()
        report = runner.run(exp, p)
        assert report.aborted and not report.results


class TestExport:
    def test_csv_round_trip(self, synapse, hw, tmp_path):
        exp, p = synapse
        report = Runner(DemoDriver(realistic_delay=False), hw).run(exp, p)
        path = export_csv(report.results[0], str(tmp_path / "out.csv"))
        assert os.path.isfile(path)

        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        header = next(r for r in rows if r and r[0] == "event")
        assert header[1:] == report.results[0].cols
        body = rows[rows.index(header) + 1:]
        assert len(body) == len(report.results[0].data[0])
        assert all(len(r) == len(header) for r in body)

    def test_export_results_writes_one_per_run(self, hw, tmp_path):
        exp = get_experiment("integrate_fire")
        p = params_from_fields(
            exp.fields(), {"if_periods": "log(4u,16u,3)", "if_n": "6"}
        )
        report = Runner(DemoDriver(realistic_delay=False), hw).run(exp, p)
        paths = export_results(report, str(tmp_path))
        assert len(paths) == 3
        assert all(os.path.isfile(x) for x in paths)


class TestRemoteSession:
    """The subprocess wrapper must contain a hung or missing DLL."""

    def test_missing_dll_raises_not_hangs(self):
        from b1500_studio.core.driver import DriverError
        from b1500_studio.instruments.session import RemoteWgfmu

        drv = RemoteWgfmu("definitely_not_a_real_wgfmu.dll", open_timeout=30.0)
        with pytest.raises(DriverError):
            drv.open("GPIB0::16::INSTR")
        drv.close()

    def test_close_is_safe_when_never_opened(self):
        from b1500_studio.instruments.session import RemoteWgfmu

        RemoteWgfmu().close()  # must not raise


class TestWgfmuAddress:
    """wgfmu.dll rejects the full VISA resource; it wants the bare address."""

    @pytest.mark.parametrize(
        "resource,expected",
        [
            ("GPIB0::16::INSTR", "GPIB0::16"),
            ("GPIB0::16", "GPIB0::16"),
            ("  gpib0::17::instr  ", "gpib0::17"),
            ("TCPIP0::1.2.3.4::INSTR", "TCPIP0::1.2.3.4"),
            ("", ""),
        ],
    )
    def test_normalize_address(self, resource, expected):
        from b1500_studio.instruments.wgfmu_dll import normalize_address

        assert normalize_address(resource) == expected

    def test_default_channels_match_slot_two_module(self):
        """The B1530A in slot 2 reports channel IDs 201/202."""
        hw = Hardware()
        assert (hw.force_ch, hw.sink_ch) == (201, 202)
