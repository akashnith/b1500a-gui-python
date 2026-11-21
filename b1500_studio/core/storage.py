"""Where a measurement goes when it is finished.

A run that exists only on screen is a run you cannot go back to. Every
completed measurement is written to its own folder under one parent, named
by date and time, holding the data, the figure, and enough of the setup to
say what produced it:

    <parent>/2026-11-20/143052_smu.iv/
        data.csv          the measured points, metrics in the header
        figure.png        the figure as the program drew it, summary included
        manifest.json     protocol, parameters, wiring, metrics, notes

The manifest is the part that matters six months later. A CSV of currents
with no record of the compliance, the step size or the channel it came from
is a column of numbers, not a measurement.

This module knows nothing about instruments or protocols: the caller hands
it a function that writes the CSVs, which is what keeps ``core`` free of
both.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SavedRun",
    "default_parent",
    "run_directory",
    "save_run",
    "slugify",
]


def default_parent() -> str:
    """Where runs go when nobody has said otherwise.

    The user's home, not the install directory: a package folder can be
    reinstalled, moved or deleted by a tool that has no idea somebody's data
    is in it.
    """
    override = os.environ.get("B1500_STUDIO_DATA")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(os.path.expanduser("~"), "B1500-Studio-Data")


def slugify(text: str) -> str:
    """A file-name-safe version of ``text``, never empty."""
    out = []
    for ch in text.strip():
        if ch.isalnum() or ch in "-._":
            out.append(ch)
        elif ch in " /\\":
            out.append("-")
    slug = "".join(out).strip("-._")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def run_directory(parent: str, key: str,
                  when: Optional[datetime] = None) -> str:
    """Create and return the folder this run owns.

    One folder per run, not one file: a run produces several files, and
    keeping them together is what stops a figure being separated from the
    data it was drawn from.
    """
    when = when or datetime.now()
    day = when.strftime("%Y-%m-%d")
    stamp = when.strftime("%H%M%S")
    folder = os.path.join(os.path.expanduser(parent), day,
                          f"{stamp}_{slugify(key)}")
    # A second run inside the same second would otherwise overwrite the
    # first, and silently losing a measurement is the one thing this module
    # must not do.
    unique = folder
    n = 2
    while os.path.exists(unique):
        unique = f"{folder}-{n}"
        n += 1
    os.makedirs(unique, exist_ok=True)
    return unique


@dataclass
class SavedRun:
    """Where a run landed, and what was written there."""

    folder: str = ""
    files: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def save_run(
    parent: str,
    *,
    key: str,
    name: str = "",
    write_csvs: Optional[Callable[[str], Sequence[str]]] = None,
    grid=None,
    parameters: Optional[Dict[str, str]] = None,
    setup: Optional[Dict[str, object]] = None,
    metrics: Optional[Sequence[Tuple[str, float]]] = None,
    notes: Optional[Sequence[str]] = None,
    when: Optional[datetime] = None,
) -> SavedRun:
    """Write one finished run into its own folder under ``parent``.

    Returns where everything went. A failure to save is reported rather than
    raised: the measurement already happened, and losing the window to an
    exception over a full disk would throw away the data as well as the
    file.
    """
    try:
        folder = run_directory(parent, key, when)
    except Exception as exc:
        return SavedRun(error=f"could not create the run folder: {exc}")

    saved = SavedRun(folder=folder)
    trouble: List[str] = []

    if write_csvs is not None:
        try:
            saved.files.extend(write_csvs(folder))
        except Exception as exc:
            trouble.append(f"data: {exc}")

    if grid is not None:
        try:
            from .plotting import save as save_figure

            saved.files.append(save_figure(grid, os.path.join(folder,
                                                              "figure.png")))
        except Exception as exc:
            trouble.append(f"figure: {exc}")

    manifest = {
        "protocol": key,
        "name": name or key,
        "saved_at": (when or datetime.now()).isoformat(timespec="seconds"),
        "parameters": dict(parameters or {}),
        "setup": dict(setup or {}),
        "metrics": {k: v for k, v in (metrics or [])},
        "notes": list(notes or []),
        "files": [os.path.basename(f) for f in saved.files],
    }
    try:
        path = os.path.join(folder, "manifest.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=False,
                      default=_jsonable)
            fh.write("\n")
        saved.files.append(path)
    except Exception as exc:
        trouble.append(f"manifest: {exc}")

    if trouble:
        saved.error = "; ".join(trouble)
    return saved


def _jsonable(value):
    """Last resort for anything json cannot serialise on its own."""
    try:
        return float(value)
    except Exception:
        return str(value)
