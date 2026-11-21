"""The DC staircase: how a bipolar I-V sweep is cut into legs.

Ported from ``patternbuilder.cpp``.  The two limbs are independent, because
a device can SET at +2 V and RESET at -4 V, and a sweep that forced them to
be the same size would be measuring the instrument's convenience rather than
the device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .base import sweep_step_count

__all__ = [
    "SweepLeg",
    "SWEEP_MODES",
    "POLARITIES",
    "staircase",
    "leg_voltages",
    "staircase_voltages",
]

#: Drop-down order for the ``sweep_mode`` field.
SWEEP_MODES = (
    "Single (0 -> V1)",
    "Dual (0 -> V1 -> V2 -> 0)",
    "Bipolar RS (0 -> V1 -> 0 -> V2 -> 0)",
)

POLARITIES = ("Positive first (+V)", "Negative first (-V)")


@dataclass
class SweepLeg:
    """One straight run of the staircase, start and stop inclusive."""

    start: float
    stop: float
    steps: int
    phase: str
    #: False where the previous leg already measured this starting point.
    include_start: bool = True

    @property
    def points(self) -> int:
        """Points the instrument is asked for, including the start."""
        return self.steps + 1

    def step_size(self) -> float:
        return abs(self.stop - self.start) / max(self.steps, 1)


def staircase(
    v_pos: float,
    v_neg: float,
    step: float,
    mode: int = 2,
    positive_first: bool = True,
    cycles: int = 1,
) -> List[SweepLeg]:
    """Cut a bipolar sweep into legs.

    ``v_pos`` and ``v_neg`` are **magnitudes**; the sign comes from which
    limb is being built.  Allowing a negative here would mean "V- max = -2"
    sweeping to +2, which is the sort of double negative that puts the wrong
    polarity on a device.

    The step size, not a point count, sets how many steps each limb gets.
    Deriving it per limb is the whole point of an asymmetric sweep: a single
    points-per-leg setting would silently give the shorter limb a finer step
    than the longer one, so the same transition would be resolved differently
    depending on which direction it was approached from.
    """
    v_pos, v_neg = abs(v_pos), abs(v_neg)
    cycles = max(1, int(cycles))
    first = v_pos if positive_first else -v_neg
    second = -v_neg if positive_first else v_pos

    legs: List[SweepLeg] = []

    def add(start: float, stop: float, phase: str, include_start: bool) -> None:
        if start == stop:
            return
        legs.append(SweepLeg(
            start, stop, sweep_step_count(abs(stop - start), step),
            phase, include_start,
        ))

    for cycle in range(cycles):
        # Only name the cycle when there is more than one, so a single sweep
        # keeps the plain phase labels.
        tag = f" (cycle {cycle + 1})" if cycles > 1 else ""
        # A later cycle starts on the zero the previous one ended at, so that
        # point is not measured twice.
        add(0.0, first, "first polarity sweep" + tag, cycle == 0)
        if mode == 1:
            add(first, second, "opposite polarity sweep" + tag, False)
            add(second, 0.0, "return to zero" + tag, False)
        elif mode >= 2:
            add(first, 0.0, "first polarity return to zero" + tag, False)
            add(0.0, second, "opposite polarity sweep" + tag, False)
            add(second, 0.0, "final return to zero" + tag, False)
    return legs


def leg_voltages(leg: SweepLeg) -> List[float]:
    """Every voltage ``leg`` applies, honouring ``include_start``."""
    n = leg.points
    values = [leg.start + (leg.stop - leg.start) * i / (n - 1) for i in range(n)]
    return values if leg.include_start else values[1:]


def staircase_voltages(legs: Sequence[SweepLeg]) -> List[float]:
    """The whole sweep as one list of applied voltages, in order."""
    out: List[float] = []
    for leg in legs:
        out.extend(leg_voltages(leg))
    return out
