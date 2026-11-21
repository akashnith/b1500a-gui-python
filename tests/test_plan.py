"""Waveform model and SI parsing."""

from __future__ import annotations

import math

import pytest

from b1500_studio.core.plan import (
    GRID,
    Event,
    Plan,
    PlanError,
    Wave,
    parse_list,
    parse_si,
    secs,
    ticks,
)


class TestParseSI:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("50n", 50e-9),
            ("1.5u", 1.5e-6),
            ("2m", 2e-3),
            ("1e-6", 1e-6),
            ("100 ns", 100e-9),
            ("1k", 1e3),
            ("1M", 1e6),
            ("3p", 3e-12),
            ("0.7", 0.7),
            ("-0.7", -0.7),
            ("200k", 200e3),
        ],
    )
    def test_values(self, text, expected):
        assert parse_si(text) == pytest.approx(expected)

    def test_micro_signs(self):
        assert parse_si("1µs") == pytest.approx(1e-6)
        assert parse_si("1μs") == pytest.approx(1e-6)

    @pytest.mark.parametrize("bad", ["", "   ", "abc", "1x", "1 furlong"])
    def test_rejects_garbage(self, bad):
        with pytest.raises(PlanError):
            parse_si(bad)


class TestParseList:
    def test_comma_list(self):
        assert parse_list("1u, 10u") == pytest.approx([1e-6, 10e-6])

    def test_lin(self):
        assert parse_list("lin(0.5,1,3)") == pytest.approx([0.5, 0.75, 1.0])

    def test_log(self):
        got = parse_list("log(1u,100u,3)")
        assert got == pytest.approx([1e-6, 10e-6, 100e-6])

    def test_single_point(self):
        assert parse_list("lin(2,5,1)") == pytest.approx([2.0])

    def test_log_rejects_non_positive(self):
        with pytest.raises(PlanError):
            parse_list("log(0,1,3)")

    def test_rejects_wrong_arity(self):
        with pytest.raises(PlanError):
            parse_list("lin(1,2)")


class TestTicks:
    def test_round_trip(self):
        assert secs(ticks(1e-6)) == pytest.approx(1e-6)

    def test_below_grid_rejected(self):
        with pytest.raises(PlanError):
            ticks(GRID / 3)

    def test_rounds_to_grid(self):
        assert ticks(1.04e-8) == 1


class TestPlan:
    def _plan(self):
        p = Plan()
        W = Wave(p)
        W.to(0.5, 10)
        W.hold(100)
        W.to(0.0, 10)
        p.ev.append(Event("e0", t=10, points=4, interval=10, avg=10))
        return p

    def test_totals(self):
        p = self._plan()
        assert p.total() == 120
        assert p.duration() == pytest.approx(120 * GRID)
        assert p.n_points() == 4
        assert p.max_abs_v() == pytest.approx(0.5)

    def test_samples_are_window_centres(self):
        p = self._plan()
        t, idx = p.samples()
        assert len(t) == 4 and set(idx) == {0}
        assert t[0] == pytest.approx((10 + 0 + 5) * GRID)
        assert t[1] == pytest.approx((10 + 10 + 5) * GRID)

    def test_validate_passes(self):
        self._plan().validate(1.0, 1000)

    def test_voltage_limit(self):
        p = self._plan()
        with pytest.raises(PlanError, match="safety limit"):
            p.validate(0.1, 1000)

    def test_point_limit(self):
        p = self._plan()
        with pytest.raises(PlanError, match="data points"):
            p.validate(1.0, 2)

    def test_event_past_end(self):
        p = self._plan()
        p.ev[0] = Event("e0", t=100, points=10, interval=10, avg=10)
        with pytest.raises(PlanError, match="past the end"):
            p.validate(1.0, 1000)

    def test_overlapping_events(self):
        p = self._plan()
        p.ev.append(Event("e1", t=0, points=1, interval=10, avg=10))
        with pytest.raises(PlanError, match="overlaps"):
            p.validate(1.0, 1000)

    def test_bad_averaging(self):
        p = self._plan()
        p.ev[0] = Event("e0", t=10, points=1, interval=5, avg=10)
        with pytest.raises(PlanError, match="averaging"):
            p.validate(1.0, 1000)

    def test_empty_waveform(self):
        with pytest.raises(PlanError, match="empty waveform"):
            Plan().validate(1.0, 1000)


class TestWave:
    def test_cursor_and_level(self):
        p = Plan()
        W = Wave(p)
        assert W.to(1.0, 5) == 5
        assert W.hold(7) == 12
        assert W.now() == 12
        assert W.v == pytest.approx(1.0)
        assert len(p.vec) == 2
