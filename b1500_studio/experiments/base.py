"""Shared machinery for experiments: fitting, windows, tables, interfaces.

Faithful port of the anonymous-namespace helpers at the top of
``pkg/src/experiments.cpp``, plus the ``Params`` / ``RunSpec`` / ``RunResult``
/ ``Experiment`` types from ``pkg/src/experiments.h``.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# Params is shared with the SMU protocols, so it lives in core.fields and
# is re-exported here: every experiment already imports it from this
# module, and the GUI builds one for either kind of protocol.
from ..core.fields import Params  # noqa: F401
from ..core.driver import Hardware, Results
from ..core.plan import Event, Plan, Wave, ticks

__all__ = [
    "NaN",
    "eng",
    "window",
    "add_event",
    "read_pulse",
    "Lin",
    "linfit",
    "Fit",
    "fit_kww",
    "cross_time",
    "mean_of",
    "require_finite",
    "t_unit",
    "finish_table",
    "Params",
    "RunSpec",
    "RunResult",
    "Experiment",
    "Severity",
    "VerificationIssue",
    "VerificationResult",
]

NaN = float("nan")


# ------------------------------------------------------------- formatting


_ENG_TABLE = ((1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p"))


def eng(v: float, unit: str = "", digits: int = 4) -> str:
    """Engineering-notation formatter, matching the C++ ``eng()``."""
    if v is None or not math.isfinite(v):
        return "n/a"
    if v == 0:
        return f"0 {unit}"
    for factor, prefix in _ENG_TABLE:
        if abs(v) >= factor * 0.9995:
            return f"{v / factor:.{digits}g} {prefix}{unit}"
    return f"{v / 1e-12:.{digits}g} p{unit}"


# ------------------------------------------------------------- plan helpers


@dataclass
class Win:
    """A measurement window: start tick and length in ticks."""

    t: int
    len: int


def window(fs: int, fe: int, settle: float) -> Win:
    """Skip ``settle`` of a flat top, then measure the remainder."""
    length = fe - fs
    st = int(round(settle * float(length)))
    st = min(st, max(length - 1, 0))
    return Win(fs + st, max(length - st, 1))


def add_event(
    r: "RunSpec",
    name: str,
    t: int,
    pts: int,
    iv: int,
    avg: int,
    phase: int,
    idx: int,
    amp: float,
    w: float,
    measure_v: float = NaN,
) -> None:
    """Append one measure event plus its per-event metadata."""
    r.plan.ev.append(Event(name=name, t=int(t), points=int(pts), interval=int(iv), avg=int(avg)))
    r.ev_phase.append(phase)
    r.ev_idx.append(idx)
    r.ev_amp.append(amp)
    r.ev_width.append(w)
    r.ev_measure_v.append(measure_v)
    r.ev_swept.append(False)


def read_pulse(
    r: "RunSpec",
    W: Wave,
    hw: Hardware,
    vr: float,
    tread: int,
    name: str,
    phase: int,
    idx: int,
    amp: float,
    w: float,
) -> None:
    """Ramp up, flat top (one averaged point), ramp down."""
    e = ticks(hw.edge)
    W.to(vr, e)
    fs = W.now()
    W.to(vr, tread)
    fe = W.now()
    W.to(0.0, e)
    wn = window(fs, fe, hw.settle)
    add_event(r, name, wn.t, 1, wn.len, wn.len, phase, idx, amp, w, vr - hw.sink_bias)


# ------------------------------------------------------------------ fitting


@dataclass
class Lin:
    """Least-squares straight-line fit."""

    slope: float = NaN
    icpt: float = NaN
    r2: float = NaN


def linfit(x: Sequence[float], y: Sequence[float]) -> Lin:
    """Ordinary least squares, returning slope, intercept and R^2."""
    L = Lin()
    n = min(len(x), len(y))
    if n < 2:
        return L
    sx = sy = sxx = sxy = syy = 0.0
    for i in range(n):
        sx += x[i]
        sy += y[i]
        sxx += x[i] * x[i]
        sxy += x[i] * y[i]
        syy += y[i] * y[i]
    dn = float(n)
    den = dn * sxx - sx * sx
    if abs(den) < 1e-300:
        return L
    L.slope = (dn * sxy - sx * sy) / den
    L.icpt = (sy - L.slope * sx) / dn
    ss_tot = syy - sy * sy / dn
    ss_res = 0.0
    for i in range(n):
        e = y[i] - (L.icpt + L.slope * x[i])
        ss_res += e * e
    L.r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-300 else NaN
    return L


@dataclass
class Fit:
    """Stretched-exponential fit ``G(t) = Ginf + A exp(-(t/tau)^beta)``."""

    tau: float = NaN
    beta: float = 1.0
    ginf: float = NaN
    A: float = NaN
    rss: float = float("inf")
    ok: bool = False


def _fit_eval(t: Sequence[float], g: Sequence[float], tau: float, beta: float) -> Tuple[float, float, float]:
    """Residual sum of squares with (ginf, A) solved linearly. -> (rss, ginf, A)."""
    Sf = Sff = Sg = Sfg = 0.0
    n = float(len(t))
    for i in range(len(t)):
        try:
            f = math.exp(-((t[i] / tau) ** beta))
        except (OverflowError, ValueError):
            f = 0.0
        Sf += f
        Sff += f * f
        Sg += g[i]
        Sfg += f * g[i]
    det = n * Sff - Sf * Sf
    if abs(det) < 1e-30:
        return float("inf"), NaN, NaN
    A = (n * Sfg - Sf * Sg) / det
    ginf = (Sg - A * Sf) / n
    rss = 0.0
    for i in range(len(t)):
        try:
            f = math.exp(-((t[i] / tau) ** beta))
        except (OverflowError, ValueError):
            f = 0.0
        e = g[i] - ginf - A * f
        rss += e * e
    return rss, ginf, A


def _fit_beta(t: Sequence[float], g: Sequence[float], beta: float) -> Fit:
    """Golden-section search over log(tau) at fixed beta."""
    F = Fit()
    if len(t) < 5:
        return F
    tmin = max(t[0], 1e-9)
    tmax = t[-1]
    if tmin <= 0 or tmax <= 0:
        return F
    lo, hi = math.log(tmin * 0.05), math.log(tmax * 20.0)
    best, bl = float("inf"), lo
    N = 160
    for i in range(N + 1):
        l = lo + (hi - lo) * i / N
        r, _, _ = _fit_eval(t, g, math.exp(l), beta)
        if r < best:
            best, bl = r, l
    a, b = bl - (hi - lo) / N, bl + (hi - lo) / N
    phi = 0.6180339887
    for _ in range(60):
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        rc, _, _ = _fit_eval(t, g, math.exp(c), beta)
        rd, _, _ = _fit_eval(t, g, math.exp(d), beta)
        if rc < rd:
            b = d
        else:
            a = c
    F.tau = math.exp(0.5 * (a + b))
    F.rss, F.ginf, F.A = _fit_eval(t, g, F.tau, beta)
    F.beta = beta
    F.ok = math.isfinite(F.rss)
    return F


def fit_kww(t: Sequence[float], g: Sequence[float]) -> Fit:
    """Scan beta in [0.30, 1.60] and keep the best stretched-exponential fit."""
    best = Fit()
    b = 0.30
    while b <= 1.60001:
        f = _fit_beta(t, g, b)
        if f.ok and f.rss < best.rss:
            best = f
        b += 0.05
    return best


def cross_time(t: Sequence[float], y: Sequence[float], level: float) -> float:
    """First time a positive decaying series reaches ``level`` (log-interp)."""
    for i in range(len(y)):
        if y[i] <= level:
            if i == 0:
                return t[0]
            y0, y1 = y[i - 1], y[i]
            if y0 == y1:
                return t[i]
            f = (y0 - level) / (y0 - y1)
            if t[i - 1] <= 0 or t[i] <= 0:
                return t[i]
            return math.exp(math.log(t[i - 1]) + f * (math.log(t[i]) - math.log(t[i - 1])))
    return NaN


def mean_of(v: Sequence[float]) -> float:
    """Arithmetic mean, NaN for an empty sequence."""
    if not len(v):
        return NaN
    return sum(v) / float(len(v))


def require_finite(values: Sequence[float], label: str) -> None:
    """Raise unless every value is finite."""
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{label} values must be finite")


def t_unit(tmax: float) -> Tuple[float, str]:
    """Pick a time scale and label for an axis spanning ``tmax`` seconds."""
    if tmax >= 1.0:
        return 1.0, "s"
    if tmax >= 1e-3:
        return 1e-3, "ms"
    if tmax >= 1e-6:
        return 1e-6, "us"
    return 1e-9, "ns"


# -------------------------------------------------------------- data model


@dataclass
class RunSpec:
    """One waveform to execute, plus per-event metadata used by analysis."""

    label: str = ""
    plan: Plan = field(default_factory=Plan)
    ev_phase: List[int] = field(default_factory=list)
    ev_idx: List[int] = field(default_factory=list)
    ev_amp: List[float] = field(default_factory=list)
    ev_width: List[float] = field(default_factory=list)
    ev_measure_v: List[float] = field(default_factory=list)
    #: True where the event measures *along a ramp* rather than at one bias.
    #
    #  A read pulse has a single device voltage, so conductance and
    #  read-disturb are meaningful for it. A PUND ramp does not: the voltage
    #  sweeps across the event by design, and that sweep is the measurement.
    #  Marking it explicitly keeps "no bias was recorded" -- a real mistake --
    #  distinguishable from "a bias would be meaningless here", instead of
    #  overloading NaN to mean both.
    ev_swept: List[bool] = field(default_factory=list)
    v: Dict[str, float] = field(default_factory=dict)


@dataclass
class RunResult:
    """Analysis output: summary metrics, notes, a CSV table and plot arrays."""

    label: str = ""
    summary: List[Tuple[str, float]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    cols: List[str] = field(default_factory=list)
    data: List[List[float]] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    a: Dict[str, List[float]] = field(default_factory=dict)
    v: Dict[str, float] = field(default_factory=dict)


def finish_table(
    R: RunResult,
    r: RunSpec,
    t: Sequence[float],
    ei: Sequence[int],
    n: int,
    I: Sequence[float],
    extra_col: Sequence[float],
    extra_name: str,
) -> None:
    """Build the per-sample CSV table, in device order."""
    R.cols = [
        "sample",
        "event_idx",
        "phase",
        "stimulus_V",
        "width_s",
        "t_s",
        "I_A",
        "measure_V",
        extra_name,
    ]
    R.data = [[] for _ in R.cols]
    R.events = []
    for i in range(n):
        e = int(ei[i])
        R.events.append(r.plan.ev[e].name)
        R.data[0].append(float(i))
        R.data[1].append(float(r.ev_idx[e]))
        R.data[2].append(float(r.ev_phase[e]))
        R.data[3].append(r.ev_amp[e])
        R.data[4].append(r.ev_width[e])
        R.data[5].append(t[i])
        R.data[6].append(I[i])
        R.data[7].append(r.ev_measure_v[e])
        R.data[8].append(extra_col[i] if i < len(extra_col) else NaN)


# ----------------------------------------------------------- verification


class Severity:
    """Verification issue severity levels."""

    INFO = "Info"
    WARNING = "Warning"
    ERROR = "Error"


@dataclass
class VerificationIssue:
    """One problem found while dry-running an experiment."""

    severity: str = Severity.INFO
    title: str = ""
    detail: str = ""


@dataclass
class VerificationResult:
    """Outcome of a dry run: totals plus any issues."""

    ok: bool = True
    has_warnings: bool = False
    issues: List[VerificationIssue] = field(default_factory=list)
    max_abs_v: float = 0.0
    total_duration: float = 0.0
    total_points: int = 0
    run_count: int = 0
    summary_text: str = ""


# ------------------------------------------------------------- interface


class Experiment(abc.ABC):
    """Build plan(s) -> (driver runs) -> analyse -> plot."""

    #: Stable short key used by the CLI and the registry.
    key: str = ""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable title shown in the UI."""

    @abc.abstractmethod
    def fields(self) -> List[dict]:
        """Field descriptors: ``{name, label, default, help, kind, choices}``."""

    @abc.abstractmethod
    def build(self, p: Params, hw: Hardware) -> List[RunSpec]:
        """Turn parameters into one or more executable waveforms."""

    @abc.abstractmethod
    def analyze(self, r: RunSpec, res: Results, hw: Hardware) -> RunResult:
        """Turn measured data into metrics, notes, a table and plot arrays."""

    @abc.abstractmethod
    def draw(self, results: List[RunResult]) -> "PanelGrid":
        """Describe the figure for ``results`` as a panel grid."""

    def plot(self, fig, results: List[RunResult]) -> None:
        """Render :meth:`draw` onto a matplotlib ``Figure``."""
        from ..core.plotting import render

        render(fig, self.draw(results))

    @staticmethod
    def summary_lines(results: List[RunResult]) -> List[str]:
        """The monospace text block shared by every summary panel."""
        txt: List[str] = []
        for R in results:
            txt.append(R.label)
            for key, value in R.summary:
                txt.append(f"  {key} = {value:.4g}")
            for note in R.notes:
                txt.append(f"  ! {note}")
            txt.append("")
        return txt

    def pause_between_runs(self, p: Params) -> float:
        """Seconds to wait between consecutive runs."""
        return 0.0

    def defaults(self) -> Params:
        """A :class:`Params` filled with this experiment's default fields."""
        from .registry import params_from_fields

        return params_from_fields(self.fields(), {})
