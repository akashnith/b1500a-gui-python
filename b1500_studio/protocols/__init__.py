"""SMU protocols: DC and slow-pulse measurements, driven step by step.

The split from :mod:`b1500_studio.experiments` is physical, not
organisational.  Below roughly 500 ns -- and in practice below a few
microseconds -- a pulse has to come from the WGFMU, which plays a waveform it
was handed in advance.  At or above that, the SMU is the better instrument,
because it has real current compliance and an integrating ADC, and it is
driven one step at a time with a decision between the steps.

    smu         the measurements themselves, as plain functions
    sweep       the DC staircase: how a bipolar I-V is cut into legs
    base        SmuSettings and the SmuProtocol interface
    catalogue   the eight protocols as objects the GUI can show
"""

from __future__ import annotations

from .base import (
    CURRENT_RANGES,
    SmuProtocol,
    SmuSettings,
    current_range_code,
    current_range_labels,
    sweep_step_count,
)
from .catalogue import (
    SMU_PROTOCOL_KEYS,
    all_smu_protocols,
    get_smu_protocol,
)
from .smu import (
    SmuRun,
    dc_endurance,
    forming,
    ndr,
    probe_resistance,
    read_retention,
    slow_pulse_ltp_ltd,
    threshold_statistics,
)
from .sweep import POLARITIES, SWEEP_MODES, SweepLeg, staircase

__all__ = [
    # interface
    "SmuProtocol",
    "SmuSettings",
    "SmuRun",
    # registry
    "all_smu_protocols",
    "get_smu_protocol",
    "SMU_PROTOCOL_KEYS",
    # ranging
    "CURRENT_RANGES",
    "current_range_labels",
    "current_range_code",
    "sweep_step_count",
    # staircase
    "staircase",
    "SweepLeg",
    "SWEEP_MODES",
    "POLARITIES",
    # measurements
    "forming",
    "read_retention",
    "ndr",
    "threshold_statistics",
    "slow_pulse_ltp_ltd",
    "dc_endurance",
    "probe_resistance",
]
