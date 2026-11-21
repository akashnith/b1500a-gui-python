"""Where a finished run goes, and what the figure keeps when it gets there.

The on-screen figure and the saved figure are deliberately different: the
window drops the summary panel because those numbers are in the metrics
table beside it, and the file keeps it because a file travels alone.
"""

from __future__ import annotations

import json
import os

import pytest

from b1500_studio.core import plotting
from b1500_studio.core.plotting import (
    Panel,
    PanelGrid,
    Series,
    has_text_panels,
    without_text_panels,
)
from b1500_studio.core.storage import (
    default_parent,
    run_directory,
    save_run,
    slugify,
)


def grid_with_summary() -> PanelGrid:
    return PanelGrid(
        rows=1, cols=3,
        panels=[
            Panel(title="I-V", xlabel="V", ylabel="A",
                  series=[Series(name="sweep", x=[0, 1, 2], y=[1e-6, 1e-5, 1e-4])]),
            Panel(title="R", xlabel="V", ylabel="ohm",
                  series=[Series(name="sweep", x=[0, 1, 2], y=[3, 2, 1])]),
            Panel(title="Summary", text=["on_off_ratio = 100"]),
        ],
    )


# -------------------------------------------------------- the summary panel


class TestSummaryPanel:
    def test_dropped_for_the_screen_and_kept_for_the_file(self):
        grid = grid_with_summary()
        assert has_text_panels(grid)
        screen = without_text_panels(grid)
        assert not has_text_panels(screen)
        assert len(screen.panels) == 2
        # The original is untouched: save() still has a summary to write.
        assert len(grid.panels) == 3
        assert has_text_panels(grid)

    def test_the_remaining_panels_reflow_into_the_gap(self):
        """Dropping a panel must not leave the hole it used to fill."""
        screen = without_text_panels(grid_with_summary())
        assert screen.rows * screen.cols == 2
        cells = [(r, c) for _p, r, c in plotting._placements(screen)]
        assert len(set(cells)) == 2

    def test_a_grid_with_no_text_panel_is_returned_unchanged(self):
        grid = PanelGrid(rows=1, cols=1, panels=[Panel(title="only")])
        assert without_text_panels(grid) is grid

    def test_a_panel_with_both_text_and_data_is_not_a_summary(self):
        """Text is only a summary when it is the whole panel."""
        grid = PanelGrid(
            rows=1, cols=1,
            panels=[Panel(title="annotated", text=["note"],
                          series=[Series(x=[0, 1], y=[0, 1])])],
        )
        assert not has_text_panels(grid)
        assert without_text_panels(grid) is grid

    def test_render_honours_the_flag(self):
        from matplotlib.figure import Figure

        fig = Figure()
        plotting.render(fig, grid_with_summary(), summary=True)
        assert len(fig.axes) == 3
        plotting.render(fig, grid_with_summary(), summary=False)
        assert len(fig.axes) == 2


# ---------------------------------------------------------------- the folder


class TestRunFolder:
    def test_named_by_date_time_and_protocol(self, tmp_path):
        folder = run_directory(str(tmp_path), "smu.iv")
        parts = os.path.normpath(folder).split(os.sep)
        assert len(parts[-1].split("_", 1)[0]) == 6          # HHMMSS
        assert parts[-1].endswith("smu.iv")
        assert len(parts[-2]) == 10 and parts[-2][4] == "-"  # YYYY-MM-DD

    def test_two_runs_in_the_same_second_do_not_overwrite(self, tmp_path):
        """Silently losing a measurement is the one thing this must not do."""
        from datetime import datetime

        when = datetime(2026, 11, 20, 14, 30, 52)
        a = run_directory(str(tmp_path), "smu.iv", when)
        b = run_directory(str(tmp_path), "smu.iv", when)
        assert a != b
        assert os.path.isdir(a) and os.path.isdir(b)

    def test_slugify_never_returns_an_empty_name(self):
        assert slugify("smu.iv") == "smu.iv"
        assert slugify("P-V loop / 2") == "P-V-loop-2"
        assert slugify("///") == "run"
        assert slugify("") == "run"

    def test_the_default_parent_is_outside_the_package(self):
        """A package folder gets reinstalled, moved and deleted by tools.

        None of those tools know somebody's data is in it.
        """
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assert not default_parent().startswith(here)


# ----------------------------------------------------------------- saving


class TestSaveRun:
    def test_writes_data_figure_and_manifest(self, tmp_path):
        saved = save_run(
            str(tmp_path),
            key="smu.iv",
            name="DC I-V / bipolar RS sweep",
            write_csvs=lambda folder: [_write(folder, "data.csv", "v,i\n0,1\n")],
            grid=grid_with_summary(),
            parameters={"sweep_step_v": "0.1"},
            setup={"force_ch": 1, "common_ch": 2},
            metrics=[("on_off_ratio", 100.0)],
            notes=["a note"],
        )
        assert saved.ok, saved.error
        names = sorted(os.path.basename(f) for f in saved.files)
        assert names == ["data.csv", "figure.png", "manifest.json"]

        m = json.load(open(os.path.join(saved.folder, "manifest.json"),
                           encoding="utf-8"))
        # The manifest is the part that matters six months later: a column
        # of currents with no record of what produced it is not a
        # measurement.
        assert m["protocol"] == "smu.iv"
        assert m["parameters"]["sweep_step_v"] == "0.1"
        assert m["setup"]["force_ch"] == 1
        assert m["metrics"]["on_off_ratio"] == 100.0
        assert m["notes"] == ["a note"]
        assert sorted(m["files"]) == ["data.csv", "figure.png"]

    def test_the_saved_figure_keeps_its_summary(self, tmp_path):
        saved = save_run(str(tmp_path), key="smu.iv", grid=grid_with_summary())
        png = os.path.join(saved.folder, "figure.png")
        assert os.path.getsize(png) > 5000

    def test_a_failure_to_save_is_reported_not_raised(self, tmp_path):
        """The measurement already happened.

        Losing the window to an exception over a full disk would throw away
        the data as well as the file.
        """
        def explode(_folder):
            raise OSError("disk full")

        saved = save_run(str(tmp_path), key="smu.iv", write_csvs=explode)
        assert not saved.ok
        assert "disk full" in saved.error
        # The folder and the manifest still exist, so the run is recorded
        # even though its data could not be written.
        assert os.path.isfile(os.path.join(saved.folder, "manifest.json"))

    def test_runs_land_under_one_parent_by_date(self, tmp_path):
        a = save_run(str(tmp_path), key="smu.iv")
        b = save_run(str(tmp_path), key="synapse")
        assert os.path.dirname(a.folder) == os.path.dirname(b.folder)
        assert os.path.dirname(os.path.dirname(a.folder)) == str(tmp_path)


def _write(folder: str, name: str, text: str) -> str:
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path
