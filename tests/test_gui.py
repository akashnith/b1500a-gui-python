"""GUI construction.

These run in a *fresh subprocess* on purpose. Importing PySide6 inside the
test session first would mask the exact bug these guard against: matplotlib's
Qt backend binding to a different Qt wrapper (PyQt6/PyQt5) than the
application uses, which only shows up on a cold start.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code: str, timeout: int = 180):
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env.pop("QT_API", None)  # the app must pin this itself
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


SINGLE_BINDING = """
import sys
import b1500_studio.gui.app          # noqa: F401
loaded = [m for m in ("PySide6.QtWidgets", "PyQt6.QtWidgets", "PyQt5.QtWidgets")
          if m in sys.modules]
print(",".join(loaded))
"""

BUILD_WINDOW = """
from PySide6.QtWidgets import QApplication
from b1500_studio.gui.app import MainWindow, SETUP_KEY
from b1500_studio.experiments import EXPERIMENT_KEYS
from b1500_studio.protocols import SMU_PROTOCOL_KEYS
app = QApplication([])
w = MainWindow(demo=True)

ALL_KEYS = list(EXPERIMENT_KEYS) + list(SMU_PROTOCOL_KEYS)

# One page per protocol -- WGFMU and SMU alike -- plus the instrument page,
# and one navigator entry for each. Derived, not hardcoded: a literal here
# has to be edited every time a protocol is added, which makes adding one
# look like a regression.
expected = len(ALL_KEYS) + 1
assert w.pages.count() == expected, (w.pages.count(), expected)
assert len(w.item_of) == expected, (len(w.item_of), expected)

# Every protocol must be reachable from the navigator; one that exists but
# cannot be selected is one nobody can run.
for key in ALL_KEYS:
    assert w.select_experiment(key), key
    assert w.current_protocol().key == key

# The two kinds must not be confused for each other: current_experiment()
# returns None for an SMU protocol and vice versa, which is what lets the
# handlers dispatch on it.
for key in SMU_PROTOCOL_KEYS:
    w.select_experiment(key)
    assert w.current_experiment() is None, key
    assert w.current_smu_protocol().key == key
    assert w.needs_smu(), key
for key in EXPERIMENT_KEYS:
    w.select_experiment(key)
    assert w.current_smu_protocol() is None, key
    assert not w.needs_smu(), key

# Every SMU protocol verifies and previews with the shipped defaults.
for key in SMU_PROTOCOL_KEYS:
    w.select_experiment(key)
    assert w.on_verify(), (key, w.validation.text())
    w.on_preview()

assert w.select_experiment("synapse")
assert w.on_verify(), w.validation.text()
w.on_preview()
assert len(w.figure.axes) == 1

# The setup page is not an experiment, so the actions that need one are off.
w.select_experiment(SETUP_KEY)
assert w.current_experiment() is None
assert not w.btn_run.isEnabled()

# The filter narrows the navigator to what was typed, and clears back.
w.filter.setText("endur")
visible = {k for k, i in w.item_of.items() if not i.isHidden()}
assert visible == {"rs_endurance", "fe_endurance", "smu.endurance"}, visible
w.filter.setText("")
assert not w.item_of["synapse"].isHidden()
print("OK")
"""


class TestGuiColdStart:
    def test_only_one_qt_binding_is_loaded(self):
        """matplotlib and the app must agree on PySide6."""
        p = _run(SINGLE_BINDING)
        assert p.returncode == 0, p.stderr
        loaded = [x for x in p.stdout.strip().split(",") if x]
        assert loaded == ["PySide6.QtWidgets"], (
            f"Qt binding clash: {loaded}. matplotlib's Qt backend must bind to "
            "PySide6; see the import-order note in gui/app.py."
        )

    def test_window_builds_and_verifies(self):
        p = _run(BUILD_WINDOW)
        assert p.returncode == 0, p.stderr
        assert "OK" in p.stdout

    def test_cli_gui_entry_point_imports(self):
        p = _run("from b1500_studio.cli import cmd_gui; print('OK')")
        assert p.returncode == 0, p.stderr


ON_SCREEN = """
from PySide6.QtWidgets import QApplication
from b1500_studio.gui.app import MainWindow
app = QApplication([])
w = MainWindow(demo=True)
w.show()
g = w.geometry()
area = app.primaryScreen().availableGeometry()
assert area.contains(g), f"window {g} is not inside screen {area}"
assert w.figure.axes, "canvas is blank on startup"
assert w.figure.axes[0].texts, "welcome panel has no text"
print("OK")
"""


class TestWindowPlacement:
    def test_opens_fully_on_screen_and_not_blank(self):
        """Guards two real bugs: off-screen placement and a blank canvas."""
        p = _run(ON_SCREEN)
        assert p.returncode == 0, p.stderr
        assert "OK" in p.stdout
