"""SMU driver: parsing, sweeps, safety and the simulated device."""

from __future__ import annotations

import math

import pytest

from b1500_studio.instruments.demo import DemoSmu
from b1500_studio.instruments.smu import (
    I_RANGES,
    SMU_MODELS,
    Smu,
    SmuError,
    SweepResult,
    export_sweep_csv,
    iv_panels,
)


@pytest.fixture
def smu():
    s = DemoSmu()
    s.open()
    yield s
    s.close()


class TestReplyParsing:
    def test_parses_fmt1_datum(self):
        value, status = Smu.parse_value("NAI+1.23456E-06")
        assert value == pytest.approx(1.23456e-6)
        assert status == "N"

    def test_parses_compliance_status(self):
        _, status = Smu.parse_value("CAI+1.00000E-03")
        assert status == "C"

    def test_falls_back_to_bare_number(self):
        value, status = Smu.parse_value("  +4.2E-05  ")
        assert value == pytest.approx(4.2e-5)
        assert status == "?"

    def test_rejects_garbage(self):
        with pytest.raises(SmuError):
            Smu.parse_value("no numbers here")


class TestConstants:
    def test_ranges_have_auto_and_fixed(self):
        assert I_RANGES["Auto"] == 0
        assert I_RANGES["Fixed 1 mA"] < 0
        assert all(isinstance(v, int) for v in I_RANGES.values())

    def test_known_smu_models(self):
        assert "B1511B" in SMU_MODELS


class TestSweepResult:
    def test_resistance_and_compliance(self):
        r = SweepResult(
            t=[0, 1, 2], v=[1.0, 1.0, 1.0],
            i=[1e-3, 0.0, 1e-6], status=["N", "N", "C"],
        )
        res = r.resistance()
        assert res[0] == pytest.approx(1e3)
        assert math.isnan(res[1])  # zero current -> undefined
        assert r.compliance_hits == 1
        assert len(r) == 3


class TestDemoSmu:
    def test_identifies_and_scans(self, smu):
        assert "B1500A" in smu.idn()
        kinds = {m.kind for m in smu.scan()}
        assert {"SMU", "WGFMU"} <= kinds

    def test_connected_flag(self):
        s = DemoSmu()
        assert not s.connected
        s.open()
        assert s.connected
        s.close()
        assert not s.connected

    def test_context_manager(self):
        with DemoSmu() as s:
            assert s.connected

    def test_spot_read_is_ohmic_at_low_bias(self, smu):
        smu.setup(1, None, 1e-3)
        value, status = smu.read_at(1, 0.1, 1e-3)
        assert value > 0
        assert status in ("N", "C")

    def test_sweep_point_count(self, smu):
        single = smu.sweep(1, -1, 1, 51, 1e-3)
        assert len(single) == 51
        double = smu.sweep(1, -1, 1, 51, 1e-3, double=True)
        assert len(double) == 101  # up then back, shared end point

    def test_sweep_rejects_too_few_points(self, smu):
        with pytest.raises(ValueError):
            smu.sweep(1, -1, 1, 1, 1e-3)

    def test_compliance_is_enforced(self, smu):
        r = smu.sweep(1, -2, 2, 81, 1e-6, double=True)
        assert all(abs(x) <= 1e-6 + 1e-18 for x in r.i)
        assert r.compliance_hits > 0

    def test_hysteresis_loop(self, smu):
        """A double sweep must switch: on/off ratio well above 1."""
        r = smu.sweep(1, -1.5, 1.5, 101, 1e-3, double=True)
        res = [abs(x) for x in r.resistance() if math.isfinite(x) and x > 0]
        assert max(res) / min(res) > 10

    def test_pulse_hardware_and_software(self, smu):
        value, _ = smu.pulse(1, 1.0, 1e-3, 1e-3, hardware=True)
        assert math.isfinite(value)
        value, _ = smu.pulse(1, 1.0, 1e-4, 1e-3, hardware=False)
        assert math.isnan(value)  # software timing cannot measure


class TestOutputs:
    def test_iv_panels(self, smu):
        r = smu.sweep(1, -1, 1, 41, 1e-3, double=True)
        grid = iv_panels([r])
        assert len(grid.panels) == 4
        assert any(p.series for p in grid.panels)
        assert any(p.text for p in grid.panels)

    def test_csv_export(self, smu, tmp_path):
        import csv

        r = smu.sweep(1, -1, 1, 21, 1e-3)
        path = export_sweep_csv(r, str(tmp_path / "iv.csv"))
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        header = next(x for x in rows if x and x[0] == "t_s")
        assert header == ["t_s", "V_V", "I_A", "status", "R_ohm"]
        body = rows[rows.index(header) + 1:]
        assert len(body) == len(r)


class TestRealSmuGuards:
    def test_commands_refused_when_not_connected(self):
        s = Smu()
        with pytest.raises(SmuError, match="not connected"):
            s.write("CN 1")
        with pytest.raises(SmuError, match="not connected"):
            s.query("*IDN?")

    def test_close_is_safe_when_never_opened(self):
        Smu().close()  # must not raise

    def test_safe_zero_is_safe_when_never_opened(self):
        Smu().safe_zero([1, 2])  # must not raise
