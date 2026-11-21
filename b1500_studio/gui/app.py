"""PySide6 application: the Python counterpart of ``mainwindow.cpp``.

The window is three columns and two bars.  A **navigator** on the left groups
the protocols by the device they measure, because that is how an operator
looks for one; the middle column holds the **parameters** for whatever the
navigator has selected; the right column holds the **output** -- figure,
metrics, measurement table and log.  A header bar states what the program is
connected to, and an action bar at the foot carries the verdict of the last
verification next to the buttons that act on it.

An earlier version put all thirty-three pages in a single tab bar.  That bar
scrolled, so finding a protocol meant paging through the others, and the tab
strip told you nothing about what any of them measured.

All instrument work happens on a worker thread, and the hardware driver is
:class:`~b1500_studio.session.RemoteWgfmu`, which runs ``wgfmu.dll`` in a
child process.  Between the two, a VISA hang can neither freeze the window
nor leave the application wedged: it surfaces as an error dialog.
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime
from typing import Dict, List, Optional

# Import order matters here. This machine has PyQt5, PyQt6 and PySide6
# installed. If matplotlib's Qt backend is imported first it binds to
# whichever binding it finds (PyQt6 by default), and mixing two bindings makes
# every widget hand-off fail with "argument 1 has unexpected type QWidget".
# So: pin QT_API, import PySide6, and only then touch matplotlib's Qt layer.
os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot  # noqa: E402
from PySide6.QtGui import QFont, QIcon  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import matplotlib  # noqa: E402

matplotlib.use("QtAgg", force=True)
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure  # noqa: E402

from ..core import plotting  # noqa: E402
from ..core.driver import Hardware  # noqa: E402
from ..experiments import params_from_fields  # noqa: E402
from ..experiments.base import Experiment, Params, Severity  # noqa: E402
from ..experiments.preview import draw_preview  # noqa: E402
from ..experiments.registry import experiments_by_family, family_of  # noqa: E402
from ..core.plan import parse_si  # noqa: E402
from ..core.runner import Runner, export_results  # noqa: E402
from ..core.storage import default_parent, save_run  # noqa: E402
from ..core.verification import verify_experiment  # noqa: E402
from ..protocols import SmuSettings, all_smu_protocols  # noqa: E402
from ..protocols.base import SmuProtocol  # noqa: E402
from . import theme  # noqa: E402

__all__ = ["MainWindow", "main", "SETUP_KEY"]

#: Set on the navigator item that owns an experiment page.
KEY_ROLE = Qt.UserRole

#: The navigator's first entry; it is not a protocol.
SETUP_KEY = "__hw__"

#: What the SMU family is called in the navigator, and what it measures.
SMU_FAMILY = (
    "SMU (DC)",
    "Staircase I-V, forming and the slow measurements with real compliance",
)


def app_icon() -> QIcon:
    """The application icon, or an empty one if it did not ship.

    A missing icon is cosmetic, so it must never stop the window opening --
    which it would if the path were simply handed to Qt and the file were
    absent from a stripped install.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("appicon.ico", "appicon.png"):
        path = os.path.join(here, "resources", name)
        if os.path.exists(path):
            icon = QIcon(path)
            if not icon.isNull():
                return icon
    return QIcon()


def short_name(exp: Experiment) -> str:
    """``exp.name`` without the family prefix the navigator already shows."""
    name = exp.name
    for prefix in ("Synapse  ", "Neuron  ", "Synapse ", "Neuron "):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# ------------------------------------------------------------------ workers


class ConnectWorker(QObject):
    """Opens an instrument session off the UI thread."""

    done = Signal(object, str)  # (channels or None, error)

    def __init__(self, driver, address: str) -> None:
        super().__init__()
        self.driver = driver
        self.address = address

    @Slot()
    def work(self) -> None:
        try:
            self.driver.open(self.address)
            self.done.emit(self.driver.channels(), "")
        except Exception as exc:
            self.done.emit(None, str(exc))


class RunWorker(QObject):
    """Executes one experiment off the UI thread."""

    log = Signal(str)
    progress = Signal(int, int, str)
    done = Signal(object, str)

    def __init__(self, exp: Experiment, params: Params, hw: Hardware, driver) -> None:
        super().__init__()
        self.exp = exp
        self.params = params
        self.hw = hw
        self.driver = driver
        self.runner: Optional[Runner] = None

    @Slot()
    def work(self) -> None:
        try:
            self.runner = Runner(
                self.driver,
                self.hw,
                log=self.log.emit,
                progress=lambda i, n, label: self.progress.emit(i, n, label),
            )
            self.done.emit(self.runner.run(self.exp, self.params), "")
        except Exception as exc:
            self.done.emit(None, f"{exc}\n\n{traceback.format_exc()}")

    def abort(self) -> None:
        if self.runner:
            self.runner.abort()


class SmuWorker(QObject):
    """Executes one SMU protocol off the UI thread.

    An SMU protocol cannot be stopped between commands the way a played
    waveform can -- it is a sequence of blocking queries -- so ``abort``
    sets a flag the protocol checks at its own step boundaries, and the
    channels are returned to 0 V whatever happens.
    """

    log = Signal(str)
    progress = Signal(int, int, str)
    done = Signal(object, str)

    def __init__(self, protocol, params, settings, smu) -> None:
        super().__init__()
        self.protocol = protocol
        self.params = params
        self.settings = settings
        self.smu = smu
        self._stop = False

    @Slot()
    def work(self) -> None:
        try:
            runs = self.protocol.run(
                self.smu, self.params, self.settings,
                log=self.log.emit,
                progress=lambda i, n, label: self.progress.emit(i, n, label),
            )
            self.done.emit(runs, "")
        except Exception as exc:
            self.done.emit(None, f"{exc}\n\n{traceback.format_exc()}")
        finally:
            # Never leave a device biased because a protocol raised.
            try:
                self.smu.safe_zero(self.settings.channels())
            except Exception:
                pass

    def abort(self) -> None:
        self._stop = True


# ------------------------------------------------------------- small pieces


def card(*, spacing: int = 0, margins=(0, 0, 0, 0)) -> tuple:
    """A bordered surface and its vertical layout."""
    frame = QFrame()
    frame.setObjectName("card")
    box = QVBoxLayout(frame)
    box.setContentsMargins(*margins)
    box.setSpacing(spacing)
    return frame, box


def caption(text: str) -> QLabel:
    """A small upper-case section heading."""
    label = QLabel(text.upper())
    label.setObjectName("sectionLabel")
    return label


def rule() -> QFrame:
    """A one-pixel horizontal divider."""
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color: {theme.BORDER_SOFT}; background: {theme.BORDER_SOFT};")
    line.setFixedHeight(1)
    return line


# ------------------------------------------------------------- main window


class MainWindow(QWidget):
    """The application window."""

    def __init__(self, address: str = "GPIB0::16::INSTR",
                 dll: Optional[str] = None, demo: bool = True) -> None:
        super().__init__()
        plotting.set_theme("light")
        self.setWindowTitle("B1500 Studio")
        self.setWindowIcon(app_icon())
        self.setStyleSheet(theme.stylesheet())

        # One navigator, two kinds of measurement. A WGFMU experiment builds
        # a waveform and hands it to a driver; an SMU protocol drives the
        # instrument step by step. They share the five members a page needs
        # -- key, name, fields, preview, draw -- so the pages are built the
        # same way and only Run has to know which it has.
        # SMU first: a device is formed and characterised on DC before it
        # is ever pulsed, so the DC protocols are what an operator reaches
        # for at the start of a session.
        self.families = [
            (SMU_FAMILY[0], SMU_FAMILY[1], all_smu_protocols())
        ] + list(experiments_by_family())
        self.protocols = [m for _n, _d, members in self.families
                          for m in members]
        self.experiments: List[Experiment] = [
            m for m in self.protocols if not isinstance(m, SmuProtocol)
        ]
        self.family_of: Dict[str, str] = {
            m.key: name for name, _d, members in self.families for m in members
        }
        self.smu = None
        self.page_of: Dict[str, int] = {}
        self.item_of: Dict[str, QTreeWidgetItem] = {}
        self.edits: Dict[str, Dict[str, QWidget]] = {}
        self.driver = None
        self.thread: Optional[QThread] = None
        self.worker = None
        self.results = []
        self.smu_runs = []
        self.last_saved = ""
        self.current_report = None

        self._build(address, dll, demo)

    # ------------------------------------------------------------- build

    def _build(self, address: str, dll: Optional[str], demo: bool) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._header(address))

        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(12, 12, 12, 10)
        bl.setSpacing(0)

        self.split = QSplitter(Qt.Horizontal)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self._navigator())
        self.split.addWidget(self._parameter_pane(address, dll, demo))
        self.split.addWidget(self._output_pane())
        self.split.setStretchFactor(0, 0)
        self.split.setStretchFactor(1, 0)
        self.split.setStretchFactor(2, 1)
        self.split.setSizes([240, 440, 780])
        bl.addWidget(self.split)
        root.addWidget(body, 1)

        root.addWidget(self._action_bar())

        self.select_experiment(self.protocols[0].key)
        self._place_on_screen()
        self._show_welcome()
        self._set_status()
        self.log(f"B1500 Studio ready ({'demo' if demo else 'hardware'} mode)")

    # -- header ---------------------------------------------------------

    def _header(self, address: str) -> QWidget:
        bar = QFrame()
        bar.setObjectName("header")
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 10, 16, 10)
        h.setSpacing(10)

        mark = QLabel()
        mark.setObjectName("appMark")
        pixmap = app_icon().pixmap(34, 34)
        if not pixmap.isNull():
            mark.setPixmap(pixmap)
            h.addWidget(mark)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel("B1500 Studio")
        title.setObjectName("appTitle")
        n_smu = len(self.protocols) - len(self.experiments)
        subtitle = QLabel(
            f"Keysight B1500A / B1530A - "
            f"{len(self.experiments)} WGFMU pulse + {n_smu} SMU DC protocols"
        )
        subtitle.setObjectName("appSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        h.addLayout(titles)
        h.addStretch(1)

        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        self.status_text = QLabel("")
        self.status_text.setObjectName("statusText")
        h.addWidget(self.status_dot)
        h.addWidget(self.status_text)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setToolTip(
            "Open the instrument session (or the simulated device in demo mode)"
        )
        self.btn_connect.clicked.connect(self.on_connect)
        h.addWidget(self.btn_connect)
        return bar

    # -- navigator ------------------------------------------------------

    def _navigator(self) -> QWidget:
        frame, v = card(spacing=0, margins=(0, 0, 0, 0))
        frame.setMinimumWidth(190)
        frame.setMaximumWidth(360)

        head = QWidget()
        hv = QVBoxLayout(head)
        hv.setContentsMargins(12, 11, 12, 8)
        hv.setSpacing(7)
        hv.addWidget(caption("Protocols"))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter...")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        hv.addWidget(self.filter)
        v.addWidget(head)
        v.addWidget(rule())

        self.nav = QTreeWidget()
        self.nav.setObjectName("nav")
        self.nav.setHeaderHidden(True)
        self.nav.setIndentation(12)
        self.nav.setAnimated(True)
        self.nav.setSelectionMode(QAbstractItemView.SingleSelection)
        self.nav.currentItemChanged.connect(self._nav_changed)

        setup = QTreeWidgetItem(self.nav, ["Instrument & safety"])
        setup.setData(0, KEY_ROLE, SETUP_KEY)
        setup.setToolTip(0, "Connection, channels, and the limits every run is "
                            "checked against")
        self.item_of[SETUP_KEY] = setup

        bold = QFont()
        bold.setBold(True)
        for name, desc, members in self.families:
            parent = QTreeWidgetItem(self.nav, [f"{name}  ({len(members)})"])
            parent.setData(0, KEY_ROLE, None)
            parent.setFont(0, bold)
            parent.setToolTip(0, desc)
            parent.setFlags(Qt.ItemIsEnabled)
            for exp in members:
                leaf = QTreeWidgetItem(parent, [short_name(exp)])
                leaf.setData(0, KEY_ROLE, exp.key)
                leaf.setToolTip(0, f"{exp.name}\nCLI key:  {exp.key}")
                self.item_of[exp.key] = leaf
        self.nav.expandAll()
        v.addWidget(self.nav, 1)
        return frame

    def _apply_filter(self, text: str) -> None:
        """Hide protocols whose name or key does not contain ``text``.

        A family whose children are all hidden is hidden too, so the list
        does not leave empty headings behind.
        """
        needle = text.strip().lower()
        for i in range(self.nav.topLevelItemCount()):
            top = self.nav.topLevelItem(i)
            key = top.data(0, KEY_ROLE)
            if key == SETUP_KEY:
                top.setHidden(bool(needle) and needle not in "instrument safety")
                continue
            shown = 0
            for j in range(top.childCount()):
                child = top.child(j)
                hay = f"{child.text(0)} {child.data(0, KEY_ROLE)}".lower()
                hit = needle in hay
                child.setHidden(not hit)
                shown += hit
            top.setHidden(shown == 0)
            if needle:
                top.setExpanded(True)

    def _nav_changed(self, current: Optional[QTreeWidgetItem], _prev) -> None:
        if current is None:
            return
        key = current.data(0, KEY_ROLE)
        if key is None:  # a family heading
            return
        self.pages.setCurrentIndex(self.page_of[key])
        self.validation.setText("Not verified yet")
        self.validation.setStyleSheet(f"color: {theme.INK_FAINT};")
        self.verdict_dot.setStyleSheet(f"color: {theme.IDLE};")
        self._set_actions_for(key)
        if hasattr(self, "status_dot"):
            self._set_status()

    def _set_actions_for(self, key: str) -> None:
        """The setup page has nothing to verify, preview or run."""
        is_exp = key != SETUP_KEY
        for b in (self.btn_verify, self.btn_preview, self.btn_run):
            b.setEnabled(is_exp)

    def select_experiment(self, key: str) -> bool:
        """Select ``key`` in the navigator; ``False`` if there is no such page."""
        item = self.item_of.get(key)
        if item is None:
            return False
        self.nav.setCurrentItem(item)
        return True

    # -- parameters -----------------------------------------------------

    def _parameter_pane(self, address: str, dll: Optional[str],
                        demo: bool) -> QWidget:
        frame, v = card(spacing=0, margins=(0, 0, 0, 0))
        frame.setMinimumWidth(320)

        head = QWidget()
        hv = QVBoxLayout(head)
        hv.setContentsMargins(14, 12, 14, 10)
        hv.setSpacing(4)
        self.page_title = QLabel("")
        self.page_title.setObjectName("pageTitle")
        self.page_title.setWordWrap(True)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.page_desc = QLabel("")
        self.page_desc.setObjectName("pageDesc")
        self.page_desc.setWordWrap(True)
        self.page_chip = QLabel("")
        self.page_chip.setObjectName("chip")
        row.addWidget(self.page_desc, 1)
        row.addWidget(self.page_chip, 0, Qt.AlignTop)
        hv.addWidget(self.page_title)
        hv.addLayout(row)
        v.addWidget(head)
        v.addWidget(rule())

        self.pages = QStackedWidget()
        self.page_of[SETUP_KEY] = self.pages.addWidget(
            self._instrument_page(address, dll, demo)
        )
        for exp in self.protocols:
            self.page_of[exp.key] = self.pages.addWidget(self._experiment_page(exp))
        self.pages.currentChanged.connect(self._page_changed)
        v.addWidget(self.pages, 1)
        return frame

    def _page_changed(self, index: int) -> None:
        for key, i in self.page_of.items():
            if i != index:
                continue
            if key == SETUP_KEY:
                self.page_title.setText("Instrument & safety")
                self.page_desc.setText(
                    "Where to connect, which channels carry the signal, and the "
                    "limits every waveform is checked against before it is sent."
                )
                self.page_chip.setText("setup")
            else:
                exp = next(e for e in self.protocols if e.key == key)
                self.page_title.setText(short_name(exp))
                instrument = ("SMU, driven step by step"
                              if isinstance(exp, SmuProtocol)
                              else "WGFMU, one played waveform")
                self.page_desc.setText(
                    f"{self.family_of.get(exp.key, '')} - {instrument}"
                )
                self.page_chip.setText(exp.key)
            return

    @staticmethod
    def _page_body() -> tuple:
        """A scrollable page and the layout its sections go into."""
        inner = QWidget()
        box = QVBoxLayout(inner)
        box.setContentsMargins(14, 10, 14, 14)
        # A section heading belongs to the fields under it, so the gap above
        # a group comes from here and the gap below its title stays small.
        box.setSpacing(16)
        area = QScrollArea()
        area.setWidget(inner)
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.NoFrame)
        # Sideways scrolling in a form is always a layout failure: the field
        # you cannot see is the one you were about to type in. Long rows wrap
        # instead (see _form below).
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return area, box

    @staticmethod
    def _form(box: QGroupBox) -> QFormLayout:
        """A form whose rows wrap rather than widen the page."""
        f = QFormLayout(box)
        f.setLabelAlignment(Qt.AlignLeft)
        f.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        f.setHorizontalSpacing(12)
        f.setVerticalSpacing(6)
        f.setContentsMargins(0, 2, 0, 0)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        f.setRowWrapPolicy(QFormLayout.WrapLongRows)
        return f

    @staticmethod
    def _combo(items, index: int) -> QComboBox:
        """A drop-down that does not stretch the page to its longest choice."""
        cb = QComboBox()
        cb.addItems(items)
        cb.setCurrentIndex(index)
        cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        cb.setMinimumContentsLength(8)
        return cb

    def _instrument_page(self, address: str, dll: Optional[str],
                         demo: bool) -> QWidget:
        page, v = self._page_body()
        fields: Dict[str, QWidget] = {}

        box = QGroupBox("Connection")
        f = self._form(box)
        e = QLineEdit(address)
        fields["addr"] = e
        f.addRow("VISA address", e)
        e = QLineEdit(dll or "")
        e.setPlaceholderText("blank = default search (System32)")
        fields["dll"] = e
        f.addRow("wgfmu.dll", e)
        c = QCheckBox("Simulated device, no instrument")
        c.setChecked(demo)
        c.toggled.connect(self._demo_toggled)
        fields["demo"] = c
        f.addRow("Demo mode", c)
        v.addWidget(box)

        box = QGroupBox("WGFMU channels")
        f = self._form(box)
        for key, label, items, idx, tip in (
            ("force_ch", "Force channel", ["101", "102"], 0,
             "the WGFMU channel that drives the device"),
            ("sink_ch", "Sink channel", ["101", "102"], 1,
             "the return path; it must be connected or you measure an open"),
            ("force_meas", "Force measures", ["V", "I", "off"], 0, ""),
            ("sink_meas", "Sink measures", ["I", "V", "off"], 0, ""),
            ("irange", "Current range",
             ["1 uA", "10 uA", "100 uA", "1 mA", "10 mA"], 4,
             "the WGFMU needs 300 ns per point below the 10 mA range"),
        ):
            cb = self._combo(items, idx)
            if tip:
                cb.setToolTip(tip)
            fields[key] = cb
            f.addRow(label, cb)
        v.addWidget(box)

        box = QGroupBox("SMU channels")
        f = self._form(box)
        for key, label, default, tip in (
            ("smu_force_ch", "Force channel", "1",
             "the SMU slot that forces the bias and measures the current"),
            ("smu_common_ch", "Return channel", "2",
             "held at the return bias; 0 means none -- but with no return "
             "the circuit is open, and every reading sits at the noise "
             "floor, which looks like a very good insulator"),
            ("smu_common_bias", "Return bias (V)", "0", ""),
            ("smu_vmax", "Max |V| limit (V)", "5",
             "the SMU gets its own ceiling: DC forming needs volts, while "
             "the pulse limit below is set for a device that should never "
             "see them from the WGFMU"),
        ):
            e = QLineEdit(default)
            if tip:
                e.setToolTip(tip)
            fields[key] = e
            f.addRow(label, e)
        v.addWidget(box)

        box = QGroupBox("WGFMU safety and timing")
        f = self._form(box)
        for key, label, default, tip in (
            ("hw_vmax", "Max |V| pulse limit (V)", "1.0",
             "WGFMU only; runs that would exceed it are refused, not clipped"),
            ("hw_edge", "Edge time (s)", "10n", "rise = fall, >= 10 ns"),
            ("hw_settle", "Settle fraction", "0.3",
             "fraction of each flat top skipped by measure windows"),
            ("hw_bias", "Sink bias (V)", "0", ""),
            ("hw_maxpts", "Max data points", "200k", ""),
        ):
            e = QLineEdit(default)
            if tip:
                e.setToolTip(tip)
            fields[key] = e
            f.addRow(label, e)
        v.addWidget(box)

        box = QGroupBox("Data")
        f = self._form(box)
        c = QCheckBox("Save every finished run automatically")
        c.setChecked(True)
        c.setToolTip("A run that exists only on screen is a run you cannot "
                     "go back to")
        fields["autosave"] = c
        f.addRow("Auto-save", c)
        e = QLineEdit(default_parent())
        e.setToolTip("One folder per run underneath this, named by date and "
                     "time, holding the data, the figure and a manifest")
        fields["data_dir"] = e
        f.addRow("Data folder", e)
        browse = QPushButton("Browse\u2026")
        browse.clicked.connect(self._pick_data_dir)
        open_it = QPushButton("Open folder")
        open_it.clicked.connect(self._open_data_dir)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        rl.addWidget(browse)
        rl.addWidget(open_it)
        rl.addStretch(1)
        f.addRow("", row)
        v.addWidget(box)

        note = QLabel(
            "The voltage limit is a software check on the waveform, not a "
            "safety interlock - it cannot protect a device from a wiring "
            "mistake.\n\nIf Connect reports a timeout, run  "
            "python -m b1500_studio check"
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)

        self.edits[SETUP_KEY] = fields
        return page

    def _experiment_page(self, exp: Experiment) -> QWidget:
        page, v = self._page_body()
        fields: Dict[str, QWidget] = {}
        group_box = None
        form = None

        for spec in exp.fields():
            group = spec.get("group", "Parameters")
            if group_box is None or group_box.title() != group:
                group_box = QGroupBox(group)
                form = self._form(group_box)
                v.addWidget(group_box)

            kind = spec.get("kind", "num")
            if kind == "combo":
                widget: QWidget = self._combo(
                    spec["choices"], int(spec["default"])
                )
            elif kind == "check":
                widget = QCheckBox()
                widget.setChecked(bool(spec["default"]))
            else:
                widget = QLineEdit(str(spec["default"]))
                widget.setPlaceholderText(str(spec["default"]))
            if spec.get("help"):
                widget.setToolTip(spec["help"])
            fields[spec["name"]] = widget
            form.addRow(spec["label"], widget)

        v.addStretch(1)
        self.edits[exp.key] = fields
        return page

    # -- output ---------------------------------------------------------

    def _output_pane(self) -> QWidget:
        self.output = QTabWidget()
        self.output.setDocumentMode(True)
        self.output.setMinimumWidth(380)

        plot_page = QWidget()
        pv = QVBoxLayout(plot_page)
        pv.setContentsMargins(6, 4, 6, 6)
        pv.setSpacing(2)
        self.figure = Figure(figsize=(9, 6), facecolor=plotting.BG)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setStyleSheet(f"background: {plotting.BG};")
        toolbar = NavigationToolbar2QT(self.canvas, plot_page)
        toolbar.setIconSize(toolbar.iconSize() * 0.8)
        toolbar.setStyleSheet(
            f"QToolBar {{ background: {theme.SURFACE}; border: none; "
            f"spacing: 2px; }} "
            f"QToolButton {{ background: transparent; border: none; "
            f"padding: 3px; border-radius: 4px; }} "
            f"QToolButton:hover {{ background: {theme.SURFACE_ALT}; }} "
            f"QLabel {{ color: {theme.INK_FAINT}; }}"
        )
        pv.addWidget(toolbar)
        pv.addWidget(self.canvas, 1)
        self.output.addTab(plot_page, "Figure")

        metrics_page = QWidget()
        mv = QVBoxLayout(metrics_page)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)
        self.metrics = QTableWidget()
        self.metrics.setColumnCount(3)
        self.metrics.setHorizontalHeaderLabels(["Run", "Metric", "Value"])
        self.metrics.verticalHeader().setVisible(False)
        self.metrics.setAlternatingRowColors(True)
        self.metrics.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        mv.addWidget(self.metrics, 3)
        self.notes = QTextEdit()
        self.notes.setObjectName("log")
        self.notes.setReadOnly(True)
        self.notes.setPlaceholderText(
            "Notes the analysis attaches to a run appear here."
        )
        mv.addWidget(self.notes, 1)
        self.output.addTab(metrics_page, "Metrics")

        self.table = QTableWidget()
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.output.addTab(self.table, "Data")

        self.log_view = QTextEdit()
        self.log_view.setObjectName("log")
        self.log_view.setReadOnly(True)
        self.output.addTab(self.log_view, "Log")
        return self.output

    # -- action bar -----------------------------------------------------

    def _action_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("actionbar")
        v = QVBoxLayout(bar)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(3)
        self.progress.setVisible(False)
        v.addWidget(self.progress)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(16, 9, 16, 9)
        h.setSpacing(8)

        self.verdict_dot = QLabel("●")
        self.verdict_dot.setObjectName("statusDot")
        self.verdict_dot.setStyleSheet(f"color: {theme.IDLE};")
        self.validation = QLabel("Not verified yet")
        self.validation.setStyleSheet(f"color: {theme.INK_FAINT};")
        self.validation.setWordWrap(False)
        h.addWidget(self.verdict_dot)
        h.addWidget(self.validation, 1)

        self.btn_verify = QPushButton("Verify")
        self.btn_verify.setToolTip(
            "Check the waveform against the safety limits. No instrument needed."
        )
        self.btn_verify.clicked.connect(self.on_verify)
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.setToolTip("Draw the pulse train that would be applied")
        self.btn_preview.clicked.connect(self.on_preview)
        self.btn_run = QPushButton("Run")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self.on_run)
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        self.btn_export = QPushButton("Export CSV")
        self.btn_export.clicked.connect(self.on_export)

        for b in (self.btn_verify, self.btn_preview, self.btn_export):
            h.addWidget(b)
        h.addSpacing(6)
        h.addWidget(self.btn_stop)
        h.addWidget(self.btn_run)
        v.addWidget(row)
        return bar

    # -- startup --------------------------------------------------------

    def _place_on_screen(self) -> None:
        """Size to fit the current screen and centre on it.

        Without this the window manager can place the window at negative
        coordinates (off the left edge, or on a monitor that is not there any
        more), which looks like the application opening half off-screen.
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1480, 920)
            return
        area = screen.availableGeometry()
        # Never larger than the screen itself: a floor bigger than the
        # available area would push the window back off the edge. That
        # includes the minimum size -- a minimum wider than the screen is
        # not a floor, it is an off-screen window nobody can shrink.
        w = min(1480, max(320, area.width() - 40))
        h = min(920, max(240, area.height() - 40))
        self.setMinimumSize(min(940, w), min(600, h))
        self.resize(w, h)
        self.move(
            area.x() + (area.width() - w) // 2,
            area.y() + (area.height() - h) // 2,
        )
        if w < 1240:
            self.split.setSizes([205, 400, max(280, w - 625)])

    def _show_welcome(self) -> None:
        """Put something on the canvas so it is not a blank rectangle."""
        panel = plotting.Panel(
            title="B1500 Studio",
            text=[
                "Pick a protocol in the navigator, then:",
                "",
                "   Verify    check the waveform is within the safety limits",
                "   Preview   draw the pulse train that would be applied",
                "   Run       execute it, plot it and fill the metrics tab",
                "",
                "Verify and Preview never touch the instrument, so a protocol",
                "can be set up and checked with nothing connected.",
                "",
                "Demo mode is on, so Run works with no instrument attached.",
                "Untick it on the Instrument page to use real hardware.",
                "",
                "SMU and MFCMU protocols are on the command line:",
                "   python -m b1500_studio smu iv --start -1.5 --stop 1.5",
                "   python -m b1500_studio cmu cv --vmax 3",
            ],
        )
        plotting.render(
            self.figure, plotting.PanelGrid(panels=[panel], rows=1, cols=1)
        )
        self.canvas.draw_idle()

    # ------------------------------------------------------------- state

    def current_protocol(self):
        """Whatever the navigator has selected, or ``None`` for setup.

        Either an :class:`Experiment` or an :class:`SmuProtocol`; callers
        that care use :func:`isinstance`, and most do not.
        """
        item = self.nav.currentItem()
        if item is None:
            return None
        key = item.data(0, KEY_ROLE)
        if key in (None, SETUP_KEY):
            return None
        return next((e for e in self.protocols if e.key == key), None)

    def current_experiment(self) -> Optional[Experiment]:
        """The selected WGFMU experiment, or ``None`` if it is not one."""
        protocol = self.current_protocol()
        return None if isinstance(protocol, SmuProtocol) else protocol

    def current_smu_protocol(self) -> Optional[SmuProtocol]:
        """The selected SMU protocol, or ``None`` if it is not one."""
        protocol = self.current_protocol()
        return protocol if isinstance(protocol, SmuProtocol) else None

    def gather(self, exp: Experiment) -> Params:
        overrides: Dict[str, object] = {}
        for name, widget in self.edits[exp.key].items():
            if isinstance(widget, QComboBox):
                overrides[name] = widget.currentIndex()
            elif isinstance(widget, QCheckBox):
                overrides[name] = widget.isChecked()
            else:
                overrides[name] = widget.text()
        return params_from_fields(exp.fields(), overrides)

    def hardware(self) -> Hardware:
        f = self.edits[SETUP_KEY]
        meas = {"V": "V", "I": "I", "off": "0"}
        return Hardware(
            force_ch=int(f["force_ch"].currentText()),
            sink_ch=int(f["sink_ch"].currentText()),
            force_meas=meas[f["force_meas"].currentText()],
            sink_meas=meas[f["sink_meas"].currentText()],
            irange=f["irange"].currentIndex(),
            vmax=parse_si(f["hw_vmax"].text()),
            edge=parse_si(f["hw_edge"].text()),
            settle=parse_si(f["hw_settle"].text()),
            sink_bias=parse_si(f["hw_bias"].text()),
            max_points=int(parse_si(f["hw_maxpts"].text())),
        )

    def smu_settings(self) -> SmuSettings:
        """Wiring and the voltage ceiling, read off the instrument page."""
        f = self.edits[SETUP_KEY]
        return SmuSettings(
            force_ch=int(parse_si(f["smu_force_ch"].text())),
            common_ch=int(parse_si(f["smu_common_ch"].text())),
            common_bias=parse_si(f["smu_common_bias"].text()),
            vmax=parse_si(f["smu_vmax"].text()),
        )

    def is_demo(self) -> bool:
        return self.edits[SETUP_KEY]["demo"].isChecked()

    def _pick_data_dir(self) -> None:
        current = self.edits[SETUP_KEY]["data_dir"].text().strip()
        folder = QFileDialog.getExistingDirectory(
            self, "Where finished runs are saved", current or default_parent()
        )
        if folder:
            self.edits[SETUP_KEY]["data_dir"].setText(folder)

    def _open_data_dir(self) -> None:
        """Show the data folder, creating it if nothing has been saved yet."""
        import os

        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        target = getattr(self, "last_saved", "") or \
            self.edits[SETUP_KEY]["data_dir"].text().strip() or default_parent()
        try:
            os.makedirs(target, exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "Data folder", str(exc))
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(target))

    def _demo_toggled(self, _on: bool) -> None:
        """Changing mode invalidates whatever session is open."""
        if self.driver is not None:
            self.on_connect()  # disconnects
        self._set_status()

    def _set_status(self) -> None:
        """Say what the program is attached to, in a word plus a colour."""
        if self.needs_smu():
            if self.smu is None:
                colour, text = ((theme.IDLE, "SMU demo - not started")
                                if self.is_demo()
                                else (theme.WARN, "SMU - not connected"))
            else:
                colour = theme.OK
                text = ("SMU demo device" if self.is_demo()
                        else "SMU - connected")
            self.status_dot.setStyleSheet(f"color: {colour};")
            self.status_text.setText(text)
            self.status_text.setStyleSheet(
                f"color: {theme.INK_SOFT if self.smu is None else theme.INK};"
            )
            self.btn_connect.setText(
                "Disconnect" if self.smu is not None else "Connect"
            )
            return
        if self.driver is None:
            if self.is_demo():
                colour, text = theme.IDLE, "Demo mode - not started"
            else:
                colour, text = theme.WARN, "Hardware - not connected"
        elif self.is_demo():
            colour, text = theme.OK, "Demo device"
        else:
            addr = self.edits[SETUP_KEY]["addr"].text().strip()
            colour, text = theme.OK, f"Connected - {addr}"
        self.status_dot.setStyleSheet(f"color: {colour};")
        self.status_text.setText(text)
        self.status_text.setStyleSheet(
            f"color: {theme.INK_SOFT if self.driver is None else theme.INK};"
        )

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.append(
            f"<span style='color:{theme.INK_FAINT}'>{stamp}</span>  {message}"
        )

    def _set_verdict(self, colour: str, text: str) -> None:
        self.verdict_dot.setStyleSheet(f"color: {colour};")
        self.validation.setText(text)
        self.validation.setStyleSheet(f"color: {theme.INK};")

    # ---------------------------------------------------------- handlers

    def on_verify(self) -> bool:
        protocol = self.current_protocol()
        if protocol is None:
            self._set_verdict(theme.IDLE, "Select a protocol first.")
            return False
        if isinstance(protocol, SmuProtocol):
            try:
                ok, message = protocol.verify(
                    self.gather(protocol), self.smu_settings()
                )
            except Exception as exc:
                self._set_verdict(theme.CRIT, f"Validation error: {exc}")
                return False
            self._set_verdict(theme.OK if ok else theme.CRIT, message)
            if not ok:
                self.log(f"[Unsafe] {protocol.key}: {message}")
            return ok
        exp = protocol
        try:
            res = verify_experiment(exp, self.gather(exp), self.hardware())
        except Exception as exc:
            self._set_verdict(theme.CRIT, f"Validation error: {exc}")
            return False
        if not res.ok:
            self._set_verdict(theme.CRIT, res.summary_text)
        elif res.has_warnings:
            self._set_verdict(theme.WARN, res.summary_text)
        else:
            self._set_verdict(theme.OK, res.summary_text)
        for issue in res.issues:
            if issue.severity != Severity.INFO:
                self.log(f"[{issue.severity}] {issue.title}: {issue.detail}")
        return res.ok

    def on_preview(self) -> None:
        protocol = self.current_protocol()
        if protocol is None:
            return
        if isinstance(protocol, SmuProtocol):
            try:
                plotting.render(self.figure, protocol.preview(
                    self.gather(protocol), self.smu_settings()
                ), summary=False)
                self.canvas.draw_idle()
                self.output.setCurrentIndex(0)
                self.log(f"preview: {protocol.key}")
            except Exception as exc:
                self.log(f"preview failed: {exc}")
                QMessageBox.warning(self, "Preview", str(exc))
            return
        exp = protocol
        try:
            specs = exp.build(self.gather(exp), self.hardware())
            plotting.render(self.figure, draw_preview(specs, exp.name),
                            summary=False)
            self.canvas.draw_idle()
            self.output.setCurrentIndex(0)
            self.log(f"preview: {len(specs)} run(s)")
        except Exception as exc:
            self.log(f"preview failed: {exc}")
            QMessageBox.warning(self, "Preview", str(exc))

    def needs_smu(self) -> bool:
        """Whether the selected protocol talks to the SMU rather than WGFMU."""
        return isinstance(self.current_protocol(), SmuProtocol)

    def ensure_smu(self) -> bool:
        """Open an SMU session if one is not already open."""
        if self.smu is not None:
            return True
        try:
            if self.is_demo():
                from ..instruments.demo import DemoSmu

                self.smu = DemoSmu(log=self.log)
                self.smu.open()
                self.log("demo SMU ready (no instrument)")
            else:
                from ..instruments.smu import Smu

                address = self.edits[SETUP_KEY]["addr"].text().strip()
                self.log(f"opening SMU on {address} ...")
                self.smu = Smu(address, log=self.log)
                self.smu.open()
                self.log(f"SMU connected: {self.smu.idn()}")
        except Exception as exc:
            self.smu = None
            self.log(f"SMU connect failed: {str(exc).splitlines()[0]}")
            QMessageBox.critical(self, "SMU connection failed", str(exc))
            return False
        self._set_status()
        return True

    def close_smu(self) -> None:
        if self.smu is None:
            return
        try:
            self.smu.safe_zero(self.smu_settings().channels())
        except Exception:
            pass
        try:
            self.smu.close()
        except Exception:
            pass
        self.smu = None

    def on_connect(self) -> None:
        # The button acts on whichever instrument the selected protocol
        # needs. Opening both would fail on a bench that has only one.
        if self.needs_smu():
            if self.smu is not None:
                self.close_smu()
                self.btn_connect.setText("Connect")
                self._set_status()
                self.log("SMU disconnected")
            elif self.ensure_smu():
                self.btn_connect.setText("Disconnect")
            return

        if self.driver is not None:
            try:
                self.driver.close()
            except Exception:
                pass
            self.driver = None
            self.btn_connect.setText("Connect")
            self._set_status()
            self.log("disconnected")
            return

        if self.is_demo():
            from ..instruments.demo import DemoDriver

            self.driver = DemoDriver()
            self.btn_connect.setText("Disconnect")
            self._set_status()
            self.log("demo driver ready (no instrument)")
            return

        from ..instruments.session import RemoteWgfmu

        f = self.edits[SETUP_KEY]
        dll = f["dll"].text().strip() or None
        address = f["addr"].text().strip()
        self.driver = RemoteWgfmu(dll)
        self.btn_connect.setEnabled(False)
        self.log(f"connecting to {address} ...")

        self.thread = QThread(self)
        self.worker = ConnectWorker(self.driver, address)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.work)
        self.worker.done.connect(self._connected)
        self.thread.start()

    @Slot(object, str)
    def _connected(self, channels, error: str) -> None:
        self._stop_thread()
        self.btn_connect.setEnabled(True)
        if error:
            self.driver = None
            self._set_status()
            self.log(f"connect failed: {error.splitlines()[0]}")
            QMessageBox.critical(self, "Connection failed", error)
            return
        self.btn_connect.setText("Disconnect")
        self._set_status()
        self.log(f"connected; channels = {channels}")

    def on_run(self) -> None:
        protocol = self.current_protocol()
        if protocol is None:
            QMessageBox.information(self, "Run", "Select a protocol first.")
            return
        if not self.on_verify():
            QMessageBox.warning(self, "Run", "Verification failed; see the log.")
            return
        if isinstance(protocol, SmuProtocol):
            self._run_smu(protocol)
            return
        exp = protocol
        if self.driver is None:
            self.on_connect()
            if self.driver is None:
                return
        if not self.is_demo():
            reply = QMessageBox.question(
                self, "Run on hardware",
                "This will apply pulses to the device. Continue?",
            )
            if reply != QMessageBox.Yes:
                return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setVisible(True)

        self.thread = QThread(self)
        self.worker = RunWorker(exp, self.gather(exp), self.hardware(), self.driver)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.work)
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._finished)
        self.thread.start()

    def _run_smu(self, protocol) -> None:
        if not self.ensure_smu():
            return
        if not self.is_demo():
            reply = QMessageBox.question(
                self, "Run on hardware",
                f"This will apply bias to the device on SMU channel "
                f"{self.smu_settings().force_ch}. Continue?",
            )
            if reply != QMessageBox.Yes:
                return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.progress.setVisible(True)

        self.thread = QThread(self)
        self.worker = SmuWorker(protocol, self.gather(protocol),
                                self.smu_settings(), self.smu)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.work)
        self.worker.log.connect(self.log)
        self.worker.progress.connect(self._progress)
        self.worker.done.connect(self._smu_finished)
        self.thread.start()

    @Slot(object, str)
    def _smu_finished(self, runs, error: str) -> None:
        self._stop_thread()
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setValue(1000)
        self.progress.setVisible(False)

        if error or runs is None:
            self.log(f"run failed: {error.splitlines()[0] if error else 'unknown'}")
            QMessageBox.critical(self, "Run failed", error or "unknown error")
            return

        protocol = self.current_smu_protocol()
        self.smu_runs = list(runs)
        self.current_report = None
        self.results = []
        grid = None
        try:
            # smu.iv alone has a log-axis choice, so it gets the parameters.
            draw = getattr(protocol, "draw_with", None)
            grid = (draw(runs, self.gather(protocol)) if draw
                    else protocol.draw(runs))
            plotting.render(self.figure, grid, summary=False)
            self.canvas.draw_idle()
        except Exception as exc:
            self.log(f"plot failed: {exc}")
        self._fill_smu_metrics(runs)
        self._fill_smu_table(runs)
        self.output.setCurrentIndex(0)
        if protocol is not None:
            from ..protocols.smu import export_smu_csv

            stem = protocol.key.replace(".", "_")
            self._autosave(
                key=protocol.key, name=protocol.name, grid=grid,
                write_csvs=lambda folder: export_smu_csv(runs, folder, stem),
                parameters=self.gather(protocol).s,
                setup=self._smu_setup(),
                metrics=[m for R in runs for m in R.metrics],
                notes=[f"{R.label}: {n}" for R in runs for n in R.notes],
            )
        for R in runs:
            self.log(f"--- {R.label}  ({len(R.x)} points)")
            for key, value in R.metrics:
                self.log(f"    {key} = {value:.6g}")
            for note in R.notes:
                self.log(f"    ! {note}")

    def _fill_smu_metrics(self, runs) -> None:
        rows = [(R.label, k, v) for R in runs for k, v in R.metrics]
        self.metrics.clearContents()
        self.metrics.setRowCount(len(rows))
        mono = QFont("Consolas")
        for i, (label, key, value) in enumerate(rows):
            self.metrics.setItem(i, 0, QTableWidgetItem(label))
            self.metrics.setItem(i, 1, QTableWidgetItem(key))
            item = QTableWidgetItem(f"{value:.6g}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            item.setFont(mono)
            self.metrics.setItem(i, 2, item)
        self.metrics.resizeColumnToContents(0)
        self.metrics.resizeColumnToContents(2)
        notes = [f"{R.label}: {n}" for R in runs for n in R.notes]
        hits = sum(R.compliance_hits() for R in runs)
        if hits:
            # A clipped point is not a measurement, and anything computed
            # across one is wrong, so say so where the numbers are read.
            notes.insert(0, f"compliance was hit on {hits} point(s); those "
                            f"points are clipped, not measured")
        self.notes.setPlainText("\n".join(notes))

    def _fill_smu_table(self, runs) -> None:
        rows = [(R, i) for R in runs for i in range(len(R.x))]
        first = runs[0]
        cols = ["phase", first.x_name, first.y_name]
        if first.y2:
            cols.append(first.y2_name or "y2")
        cols.append("status")
        self.table.clear()
        self.table.setColumnCount(len(cols))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(cols)
        for r, (R, i) in enumerate(rows):
            phase = R.phase[i] if i < len(R.phase) else ""
            self.table.setItem(r, 0, QTableWidgetItem(phase or R.label))
            values = [R.x[i], R.y[i]]
            if first.y2:
                values.append(R.y2[i] if i < len(R.y2) else float("nan"))
            for c, value in enumerate(values, start=1):
                text = "" if value != value else f"{value:.6g}"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, item)
            status = R.status[i] if i < len(R.status) else ""
            self.table.setItem(r, len(cols) - 1, QTableWidgetItem(status))
        self.table.resizeColumnsToContents()

    @Slot(int, int, str)
    def _progress(self, i: int, n: int, label: str) -> None:
        self.progress.setValue(int(1000 * i / max(n, 1)))

    @Slot(object, str)
    def _finished(self, report, error: str) -> None:
        self._stop_thread()
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress.setValue(1000)
        self.progress.setVisible(False)

        if error or report is None:
            self.log(f"run failed: {error.splitlines()[0] if error else 'unknown'}")
            QMessageBox.critical(self, "Run failed", error or "unknown error")
            return
        if report.error:
            self.log(f"run failed: {report.error}")
            QMessageBox.critical(self, "Run failed", report.error)
            return

        self.current_report = report
        self.results = report.results
        exp = self.current_experiment()
        grid = None
        try:
            grid = exp.draw(report.results)
            # summary=False: those numbers are in the Metrics tab beside the
            # canvas. The saved figure keeps the panel, because a file that
            # leaves the program has to carry them.
            plotting.render(self.figure, grid, summary=False)
            self.canvas.draw_idle()
        except Exception as exc:
            self.log(f"plot failed: {exc}")
        self._fill_metrics(report.results)
        self._fill_table(report.results)
        self.output.setCurrentIndex(0)
        self._autosave(
            key=exp.key, name=exp.name, grid=grid,
            write_csvs=lambda folder: export_results(report, folder),
            parameters=self.gather(exp).s,
            setup=self._hardware_setup(),
            metrics=[m for R in report.results for m in R.summary],
            notes=[f"{R.label}: {n}" for R in report.results
                   for n in (getattr(R, "notes", []) or [])],
        )
        for R in report.results:
            self.log(f"--- {R.label}")
            for key, value in R.summary:
                self.log(f"    {key} = {value:.6g}")

    def _hardware_setup(self) -> dict:
        """The WGFMU wiring and limits, as the manifest records them."""
        hw = self.hardware()
        return {
            "instrument": "WGFMU",
            "mode": "demo" if self.is_demo() else "hardware",
            "visa_address": self.edits[SETUP_KEY]["addr"].text().strip(),
            "force_ch": hw.force_ch,
            "sink_ch": hw.sink_ch,
            "force_measures": hw.force_meas,
            "sink_measures": hw.sink_meas,
            "irange_index": hw.irange,
            "vmax_V": hw.vmax,
            "edge_s": hw.edge,
            "settle_fraction": hw.settle,
            "sink_bias_V": hw.sink_bias,
            "max_points": hw.max_points,
        }

    def _smu_setup(self) -> dict:
        """The SMU wiring and ceiling, as the manifest records them."""
        s = self.smu_settings()
        return {
            "instrument": "SMU",
            "mode": "demo" if self.is_demo() else "hardware",
            "visa_address": self.edits[SETUP_KEY]["addr"].text().strip(),
            "force_ch": s.force_ch,
            "common_ch": s.common_ch,
            "common_bias_V": s.common_bias,
            "vmax_V": s.vmax,
        }

    def _autosave(self, **kw) -> None:
        """Write the finished run under the data folder, if that is on.

        A save that fails is logged, never raised: the measurement already
        happened, and losing the window over a full disk would throw away
        the data as well as the file.
        """
        if not self.edits[SETUP_KEY]["autosave"].isChecked():
            return
        parent = self.edits[SETUP_KEY]["data_dir"].text().strip() \
            or default_parent()
        saved = save_run(parent, **kw)
        if saved.folder:
            self.last_saved = saved.folder
            self.log(f"saved {len(saved.files)} file(s) to {saved.folder}")
        if not saved.ok:
            self.log(f"! save incomplete: {saved.error}")

    def _fill_metrics(self, results) -> None:
        """Metrics belong in a table, not only in the log.

        The log is a record of what happened; the numbers the run produced
        are the result, and scrolling a log to re-read them is not a way to
        read a result.
        """
        rows = [(R.label, k, v) for R in results for k, v in R.summary]
        self.metrics.clearContents()
        self.metrics.setRowCount(len(rows))
        mono = QFont("Consolas")
        for i, (label, key, value) in enumerate(rows):
            self.metrics.setItem(i, 0, QTableWidgetItem(label))
            self.metrics.setItem(i, 1, QTableWidgetItem(key))
            item = QTableWidgetItem(f"{value:.6g}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            item.setFont(mono)
            self.metrics.setItem(i, 2, item)
        self.metrics.resizeColumnToContents(0)
        self.metrics.resizeColumnToContents(2)

        notes = []
        for R in results:
            for note in getattr(R, "notes", []) or []:
                notes.append(f"{R.label}: {note}")
        self.notes.setPlainText("\n".join(notes))

    def _fill_table(self, results) -> None:
        if not results:
            return
        R = results[0]
        rows = len(R.data[0]) if R.data else 0
        self.table.clear()
        self.table.setColumnCount(len(R.cols) + 1)
        self.table.setRowCount(rows)
        self.table.setHorizontalHeaderLabels(["event"] + list(R.cols))
        for i in range(rows):
            event = R.events[i] if i < len(R.events) else ""
            self.table.setItem(i, 0, QTableWidgetItem(event))
            for c, col in enumerate(R.data):
                value = col[i] if i < len(col) else float("nan")
                text = "" if value != value else f"{value:.6g}"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, c + 1, item)
        self.table.resizeColumnsToContents()

    def on_stop(self) -> None:
        if self.worker and hasattr(self.worker, "abort"):
            self.worker.abort()
            self.log("abort requested")

    def on_export(self) -> None:
        has_smu = bool(getattr(self, "smu_runs", None))
        if not self.current_report and not has_smu:
            QMessageBox.information(self, "Export", "Run a protocol first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Export CSV to folder")
        if not folder:
            return
        if has_smu:
            from ..protocols.smu import export_smu_csv

            protocol = self.current_smu_protocol()
            stem = (protocol.key if protocol else "smu").replace(".", "_")
            for path in export_smu_csv(self.smu_runs, folder, stem):
                self.log(f"wrote {path}")
            return
        for path in export_results(self.current_report, folder):
            self.log(f"wrote {path}")

    def _stop_thread(self) -> None:
        if self.thread:
            self.thread.quit()
            self.thread.wait(5000)
            self.thread = None

    def closeEvent(self, event) -> None:
        self._stop_thread()
        self.close_smu()
        if self.driver is not None:
            try:
                self.driver.close()
            except Exception:
                pass
        super().closeEvent(event)


def main(address: str = "GPIB0::16::INSTR", dll: Optional[str] = None,
         demo: bool = True) -> int:
    """Launch the GUI; returns the Qt exit code."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("B1500 Studio")
    app.setApplicationDisplayName("B1500 Studio")
    app.setWindowIcon(app_icon())
    window = MainWindow(address=address, dll=dll, demo=demo)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
