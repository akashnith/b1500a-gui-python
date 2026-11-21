"""Run orchestration: execute an experiment's runs and collect the results.

Sits between the experiments (pure) and the drivers (hardware).  Used by both
the CLI and the GUI so that the run/abort/log/export behaviour is identical.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional, Sequence

from .driver import Driver, Hardware
from ..experiments.base import Experiment, Params, RunResult, RunSpec

__all__ = ["RunReport", "Runner", "export_csv", "export_results"]


@dataclass
class RunReport:
    """Everything one invocation produced."""

    experiment: str = ""
    key: str = ""
    driver: str = ""
    started: str = ""
    results: List[RunResult] = field(default_factory=list)
    specs: List[RunSpec] = field(default_factory=list)
    log: List[str] = field(default_factory=list)
    aborted: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and not self.aborted


class Runner:
    """Execute every :class:`RunSpec` an experiment builds, in order."""

    def __init__(
        self,
        driver: Driver,
        hw: Hardware,
        log: Optional[Callable[[str], None]] = None,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        self.driver = driver
        self.hw = hw
        self._log_cb = log
        self._progress_cb = progress
        self._abort = False

    def log(self, message: str) -> None:
        if self._log_cb:
            self._log_cb(message)

    def abort(self) -> None:
        """Ask the current run to stop as soon as it can."""
        self._abort = True
        try:
            self.driver.abort()
        except Exception:
            pass

    def run(self, exp: Experiment, p: Params) -> RunReport:
        """Build, execute and analyse every run of ``exp``.

        The abort flag is deliberately *not* cleared here: a Stop pressed
        between constructing the Runner and entering ``run()`` must still be
        honoured. Each invocation gets a fresh Runner.
        """
        report = RunReport(
            experiment=exp.name,
            key=exp.key,
            driver=self.driver.name,
            started=datetime.now().isoformat(timespec="seconds"),
        )

        def note(message: str) -> None:
            report.log.append(message)
            self.log(message)

        try:
            specs = exp.build(p, self.hw)
        except Exception as exc:
            report.error = f"plan construction failed: {exc}"
            note(report.error)
            return report

        report.specs = specs
        if not specs:
            report.error = "experiment generated 0 runs"
            note(report.error)
            return report

        pause = 0.0
        try:
            pause = exp.pause_between_runs(p)
        except Exception:
            pass

        total = len(specs)
        for i, spec in enumerate(specs):
            if self._abort:
                report.aborted = True
                note("aborted before run %d/%d" % (i + 1, total))
                break
            if self._progress_cb:
                self._progress_cb(i, total, spec.label)
            note(
                f"run {i + 1}/{total}: {spec.label} "
                f"({spec.plan.duration():.6g} s, {spec.plan.n_points()} points)"
            )
            try:
                spec.plan.validate(self.hw.vmax, self.hw.max_points)
                data = self.driver.run(spec.plan, self.hw, note)
                report.results.append(exp.analyze(spec, data, self.hw))
            except Exception as exc:
                report.error = f"run '{spec.label}' failed: {exc}"
                note(report.error)
                return report

            if pause > 0 and i + 1 < total and not self._abort:
                deadline = time.monotonic() + pause
                while time.monotonic() < deadline and not self._abort:
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

        if self._progress_cb:
            self._progress_cb(total, total, "done")
        return report


# ------------------------------------------------------------------ export


def export_csv(result: RunResult, path: str) -> str:
    """Write one :class:`RunResult` table to ``path`` and return the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    rows = len(result.data[0]) if result.data else 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["# experiment"] + [result.label])
        for key, value in result.summary:
            w.writerow(["#", key, f"{value:.10g}"])
        for n in result.notes:
            w.writerow(["# note", n])
        w.writerow(["event"] + list(result.cols))
        for i in range(rows):
            event = result.events[i] if i < len(result.events) else ""
            w.writerow([event] + [
                ("" if col[i] != col[i] else f"{col[i]:.10g}")
                if i < len(col)
                else ""
                for col in result.data
            ])
    return path


def export_results(
    report: RunReport, folder: str, stem: Optional[str] = None
) -> List[str]:
    """Write one CSV per run into ``folder``; returns the paths written."""
    os.makedirs(folder, exist_ok=True)
    base = stem or (
        report.key
        + "_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    paths: List[str] = []
    for i, result in enumerate(report.results):
        safe = "".join(
            ch if ch.isalnum() or ch in "-_." else "_" for ch in result.label
        ).strip("_")
        name = f"{base}_{i:02d}_{safe}.csv" if safe else f"{base}_{i:02d}.csv"
        paths.append(export_csv(result, os.path.join(folder, name)))
    return paths
