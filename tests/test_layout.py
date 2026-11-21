"""Figure layout, the protocol families, and the colour theme.

The overlap test here exists because twelve figures shipped with every panel
drawn on top of the first one.  The suite was checking that ``draw()``
returned panels, which it did -- it never checked that they landed in
different places, so a figure showing one panel and hiding two looked
exactly like a passing figure.
"""

from __future__ import annotations

import pytest

from b1500_studio.core import plotting
from b1500_studio.instruments.demo import DemoDriver
from b1500_studio.core.driver import Hardware
from b1500_studio.experiments import all_experiments, params_from_fields
from b1500_studio.experiments.registry import (
    EXPERIMENT_FAMILIES,
    EXPERIMENT_KEYS,
    experiments_by_family,
    family_of,
)
from b1500_studio.core.plotting import Panel, PanelGrid, _placements

from test_experiments import SMALL, VMAX

ALL = [e.key for e in all_experiments()]


def figure_for(key):
    exp = next(e for e in all_experiments() if e.key == key)
    hw = Hardware()
    hw.vmax = VMAX.get(key, hw.vmax)
    p = params_from_fields(exp.fields(), SMALL.get(key, {}))
    driver = DemoDriver(realistic_delay=False)
    specs = exp.build(p, hw)
    return exp.draw([exp.analyze(s, driver.run(s.plan, hw), hw) for s in specs])


# ------------------------------------------------------------- placement


class TestPlacement:
    @pytest.mark.parametrize("key", ALL)
    def test_no_two_panels_share_a_cell(self, key):
        grid = figure_for(key)
        cells = [(r, c) for _p, r, c in _placements(grid)]
        assert len(set(cells)) == len(cells), (
            f"{key}: panels overlap at {cells}; one is drawn over another"
        )

    @pytest.mark.parametrize("key", ALL)
    def test_grid_is_big_enough_for_its_panels(self, key):
        grid = figure_for(key)
        assert len(grid.panels) <= max(grid.rows, 1) * max(grid.cols, 1), (
            f"{key}: more panels than cells"
        )

    def test_unstated_cells_fill_in_reading_order(self):
        grid = PanelGrid(rows=2, cols=2, panels=[Panel(), Panel(), Panel()])
        assert [(r, c) for _p, r, c in _placements(grid)] == [
            (0, 0), (0, 1), (1, 0)
        ]

    def test_stated_cells_are_kept_and_reserved(self):
        """An automatic panel must route around one that named its cell."""
        grid = PanelGrid(
            rows=2, cols=2,
            panels=[Panel(), Panel(row=0, col=0), Panel()],
        )
        placed = [(r, c) for _p, r, c in _placements(grid)]
        assert placed[1] == (0, 0)          # the explicit one keeps its cell
        assert placed[0] != placed[1]       # and nothing lands on top of it
        assert len(set(placed)) == 3

    def test_a_span_reserves_every_cell_it_covers(self):
        grid = PanelGrid(
            rows=2, cols=2,
            panels=[Panel(row=0, col=0, col_span=2), Panel()],
        )
        placed = [(r, c) for _p, r, c in _placements(grid)]
        assert placed == [(0, 0), (1, 0)]


# -------------------------------------------------------------- families


class TestFamilies:
    def test_every_experiment_is_filed_exactly_once(self):
        filed = [k for _n, _d, keys in EXPERIMENT_FAMILIES for k in keys]
        assert sorted(filed) == sorted(EXPERIMENT_KEYS)
        assert len(filed) == len(set(filed)), "a key is in two families"

    def test_grouping_loses_nothing(self):
        grouped = [e.key for _n, _d, members in experiments_by_family()
                   for e in members]
        assert sorted(grouped) == sorted(EXPERIMENT_KEYS)

    def test_unknown_keys_land_in_other_rather_than_vanishing(self):
        assert family_of("not_a_protocol") == "Other"

    @pytest.mark.parametrize("key", ALL)
    def test_known_keys_are_not_other(self, key):
        assert family_of(key) != "Other", f"{key} is not filed under a family"


# ----------------------------------------------------------------- theme


class TestTheme:
    def test_light_is_the_default(self):
        assert plotting.theme_name() == "light"

    def test_switching_theme_changes_the_series_colours(self):
        try:
            light = plotting.palette(0)
            plotting.set_theme("dark")
            assert plotting.theme_name() == "dark"
            assert plotting.palette(0) != light
            assert plotting.BG != "#ffffff"
        finally:
            plotting.set_theme("light")
        assert plotting.palette(0) == light

    def test_unknown_theme_is_refused(self):
        with pytest.raises(ValueError):
            plotting.set_theme("solarized")
        assert plotting.theme_name() == "light"

    def test_palette_wraps_and_never_invents_a_hue(self):
        hues = {plotting.palette(i) for i in range(40)}
        assert len(hues) == 8, "a 9th series must reuse a slot, not a new hue"
