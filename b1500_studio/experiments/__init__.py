"""Experiment definitions: build a plan, run it, analyse it, draw it.

Each module here owns one measurement: its UI fields, its waveform builder,
its analysis and its figure.  Nothing in this package touches the instrument,
so every experiment can be dry-run and unit-tested without hardware.
"""

from __future__ import annotations

from .base import (
    Experiment,
    Params,
    RunResult,
    RunSpec,
    Severity,
    VerificationIssue,
    VerificationResult,
    eng,
)
from .registry import (
    EXPERIMENT_FAMILIES,
    EXPERIMENT_KEYS,
    all_experiments,
    experiments_by_family,
    family_of,
    get_experiment,
    params_from_fields,
)

__all__ = [
    "Experiment",
    "Params",
    "RunSpec",
    "RunResult",
    "VerificationIssue",
    "VerificationResult",
    "Severity",
    "eng",
    "all_experiments",
    "get_experiment",
    "params_from_fields",
    "EXPERIMENT_KEYS",
    "EXPERIMENT_FAMILIES",
    "experiments_by_family",
    "family_of",
]
