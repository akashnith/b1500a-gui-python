"""What an SMU protocol is, and the settings every one of them shares.

A WGFMU experiment and an SMU protocol are different shapes, and pretending
otherwise would hurt both.  An :class:`~b1500_studio.experiments.base.Experiment`
*builds a waveform* and hands it to a driver that plays it; an SMU protocol
*drives the instrument step by step*, reading as it goes, because that is how
a DC measurement with real current compliance works.

What they do share is the part the user interface cares about: a key, a name,
a list of input fields, a preview, a run, and a figure.  So the two expose the
same five members and the GUI builds a page for either without knowing which
it has.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from ..core.fields import Params
from ..core.plotting import PanelGrid

__all__ = [
    "SmuSettings",
    "SmuProtocol",
    "CURRENT_RANGES",
    "current_range_labels",
    "current_range_code",
    "sweep_step_count",
]


# ------------------------------------------------------------ current ranges

#: FLEX Table 4-3 ranging codes, as ``(label, code)``.
#:
#: 0 is full autoranging; 11..18 are *limited* autoranging, which stops the
#: instrument dropping to a range so sensitive that a fast transition is
#: missed while it settles; the negatives are fixed ranges.
CURRENT_RANGES: Tuple[Tuple[str, int], ...] = (
    ("Auto, full range", 0),
    ("Auto, min 1 nA", 11),
    ("Auto, min 10 nA", 12),
    ("Auto, min 100 nA", 13),
    ("Auto, min 1 uA", 14),
    ("Auto, min 10 uA", 15),
    ("Auto, min 100 uA", 16),
    ("Auto, min 1 mA", 17),
    ("Auto, min 10 mA", 18),
    ("Fixed 1 nA", -11),
    ("Fixed 10 nA", -12),
    ("Fixed 100 nA", -13),
    ("Fixed 1 uA", -14),
    ("Fixed 10 uA", -15),
    ("Fixed 100 uA", -16),
    ("Fixed 1 mA", -17),
    ("Fixed 10 mA", -18),
)


def current_range_labels() -> List[str]:
    """The range names, in the order the drop-down shows them."""
    return [label for label, _code in CURRENT_RANGES]


def current_range_code(choice: int) -> int:
    """The FLEX code for drop-down index ``choice``; 0 (auto) if out of range.

    An index that is out of range must not become a *fixed* range by
    accident.  A fixed range below the real current reads as saturated, and
    a saturated reading looks like a measurement rather than like an error.
    """
    if not 0 <= choice < len(CURRENT_RANGES):
        return 0
    return CURRENT_RANGES[choice][1]


def sweep_step_count(magnitude: float, step: float) -> int:
    """How many steps of size ``step`` span ``magnitude``, at least one.

    The epsilon is what stops ``2.0 / 0.1`` -- which is 19.999999999999996 in
    binary floating point -- from rounding up to 21 steps and putting a
    duplicate point at the peak of every sweep.
    """
    if not (step > 0.0) or not (magnitude > 0.0):
        return 1
    return max(1, int(math.ceil(magnitude / step - 1.0e-9)))


# ---------------------------------------------------------------- settings


@dataclass
class SmuSettings:
    """Channel assignment and the limits every SMU protocol is held to.

    This is the SMU's counterpart to
    :class:`~b1500_studio.core.driver.Hardware`: the parts of the setup that
    belong to the *instrument and the wiring*, not to any one measurement.
    """

    #: The channel that forces the bias and measures the current.
    force_ch: int = 1
    #: The return path, held at ``common_bias``.  0 means "not used".
    #:
    #: A two-probe measurement still needs a return: with no second channel
    #: enabled the circuit is open and every reading sits at the noise floor,
    #: which looks like a very good insulator rather than like a mistake.
    common_ch: int = 0
    common_bias: float = 0.0
    #: Software ceiling on |V|.  A run that would exceed it is refused.
    #:
    #: Compliance, integration time and current range are *not* here: they
    #: differ per measurement -- a forming ramp and a read want different
    #: compliance by a factor of a thousand -- so each protocol asks for its
    #: own.  What lives here is the wiring, which does not change between
    #: protocols, and the one limit that must hold for all of them.
    vmax: float = 5.0

    def channels(self) -> List[int]:
        """Every channel this setup enables, force first."""
        return [self.force_ch] + ([self.common_ch] if self.common_ch else [])

    def check(self, peak_v: float) -> Optional[str]:
        """Why ``peak_v`` may not be applied, or ``None`` if it may."""
        if not math.isfinite(peak_v):
            return "the sweep peak is not a finite voltage"
        if abs(peak_v) > self.vmax:
            return (f"|V| = {abs(peak_v):g} V exceeds the safety limit "
                    f"{self.vmax:g} V (Instrument page)")
        if not self.common_ch:
            return None
        if self.common_ch == self.force_ch:
            return "the force and return channels are the same channel"
        return None


# ---------------------------------------------------------------- protocol


class SmuProtocol(abc.ABC):
    """One DC or slow-pulse measurement, driven step by step."""

    #: Stable key used by the CLI, the registry and the GUI navigator.
    key: str = ""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable title shown in the UI."""

    @abc.abstractmethod
    def fields(self) -> List[dict]:
        """Field descriptors, the same shape the experiments use."""

    @abc.abstractmethod
    def run(
        self,
        smu,
        p: Params,
        s: SmuSettings,
        log: Optional[Callable[[str], None]] = None,
        progress: Optional[Callable[[int, int, str], None]] = None,
    ) -> List["SmuRun"]:
        """Execute against a connected SMU and return the measured runs."""

    @abc.abstractmethod
    def draw(self, runs: Sequence["SmuRun"]) -> PanelGrid:
        """Describe the figure for ``runs``."""

    # -- optional ------------------------------------------------------

    def preview(self, p: Params, s: SmuSettings) -> PanelGrid:
        """The bias this protocol would apply, drawn without touching it.

        The default says so rather than drawing a guess: a protocol whose
        applied bias depends on what it measures -- forming stops at the
        event, write-verify stops at the target -- has no waveform to show
        in advance, and inventing one would show the operator a measurement
        that is not going to happen.  :class:`SmuIv` overrides it.
        """
        from ..core.plotting import Panel

        return PanelGrid(
            rows=1, cols=1,
            panels=[Panel(
                title=self.name,
                text=[
                    "This protocol has no fixed waveform to preview.",
                    "",
                    "Where it stops depends on what it measures, so the",
                    "bias it will apply is not known until it runs.",
                    "",
                    "Press Verify to check the limits, then Run.",
                ],
            )],
        )

    def peak_voltage(self, p: Params) -> float:
        """The largest |V| this protocol would apply, for the safety check."""
        return 0.0

    def verify(self, p: Params, s: SmuSettings) -> Tuple[bool, str]:
        """``(ok, message)`` -- the dry-run check, with no instrument."""
        try:
            peak = self.peak_voltage(p)
        except Exception as exc:
            return False, f"Parameter error: {exc}"
        problem = s.check(peak)
        if problem:
            return False, problem
        extra = ""
        if "icomp" in p.d:
            extra = f", compliance {p.num('icomp'):g} A"
        if "adc_plc" in p.d:
            extra += f", {p.num('adc_plc'):g} PLC"
        if not s.common_ch:
            extra += "  -- no return channel set"
        return True, f"Checked: max |V| = {abs(peak):g} V{extra}"


# Imported late: SmuRun lives beside the protocol functions, and importing it
# at module scope would make protocols.base and protocols.smu import each
# other.
from .smu import SmuRun  # noqa: E402,F401
