"""Field descriptors, parameter gathering and the experiment registry.

In the Qt app the per-experiment input fields live in ``mainwindow.cpp``.
Here each experiment owns its own :meth:`Experiment.fields`, so the GUI, the
CLI and the tests all build parameters from one source of truth.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from ..core.fields import (  # noqa: F401
    Params,
    check_field,
    combo_field,
    grouped,
    num_field,
    params_from_fields,
    text_field,
)
from .base import Experiment

__all__ = [
    "num_field",
    "text_field",
    "combo_field",
    "check_field",
    "grouped",
    "params_from_fields",
    "all_experiments",
    "get_experiment",
    "EXPERIMENT_KEYS",
    "EXPERIMENT_FAMILIES",
    "family_of",
    "experiments_by_family",
]


# The field descriptors and the gatherer are shared with the SMU protocols,
# so they live in core.fields.  They are re-exported here because every
# experiment imports them from this module.


# -------------------------------------------------------------- registry


def all_experiments() -> List[Experiment]:
    """One fresh instance of every experiment, in UI tab order."""
    from .ferro_loops import FeAfe, FeFtj, FeIv, FePv
    from .ferro_pulsed import (
        FeAccumulation,
        FeAfePund,
        FeEndurance,
        FeSpeed,
    )
    from .ferroelectric import FePund
    from .integrate_fire import IntegrateFire
    from .neuron_extra import NeuronFi, NeuronIv, NeuronLeak, NeuronOscillator
    from .synapse_dynamics import (
        SynapseEndurance,
        SynapseMlc,
        SynapseRetention,
        SynapseSrdp,
        SynapseStp,
    )
    from .synapse_sweeps import (
        SynapseIncrementalAmplitude,
        SynapseIncrementalWidth,
        SynapseKinetics,
        SynapseMetrics,
        SynapseSpeed,
        SynapseWidthSweep,
    )
    from .ppf import SynapsePPF
    from .refractory import NeuronRefractory
    from .relaxation import Relaxation
    from .stdp import SynapseSTDP
    from .stochastic import NeuronStochastic
    from .synapse import Synapse
    from .synapse_epsc import SynapseEPSC

    return [
        Synapse(),
        SynapseEPSC(),
        SynapseSTDP(),
        Relaxation(),
        IntegrateFire(),
        NeuronStochastic(),
        SynapsePPF(),
        NeuronRefractory(),
        FePund(),
        FePv(),
        FeIv(),
        FeAfe(),
        FeFtj(),
        FeSpeed(),
        FeEndurance(),
        FeAccumulation(),
        FeAfePund(),
        SynapseWidthSweep(),
        SynapseIncrementalWidth(),
        SynapseIncrementalAmplitude(),
        SynapseMetrics(),
        SynapseKinetics(),
        SynapseSpeed(),
        SynapseSrdp(),
        SynapseStp(),
        SynapseMlc(),
        SynapseRetention(),
        SynapseEndurance(),
        NeuronFi(),
        NeuronLeak(),
        NeuronIv(),
        NeuronOscillator(),
    ]


#: Stable CLI keys, in the same order as :func:`all_experiments`.
EXPERIMENT_KEYS = [
    "synapse",
    "epsc",
    "stdp",
    "relaxation",
    "integrate_fire",
    "stochastic",
    "ppf",
    "refractory",
    "pund",
    "pv",
    "fe_iv",
    "afe",
    "ftj",
    "fe_speed",
    "fe_endurance",
    "fe_accumulation",
    "afe_pund",
    "width_sweep",
    "incremental_width",
    "incremental_amplitude",
    "metrics",
    "kinetics",
    "rs_speed",
    "srdp",
    "stp",
    "mlc",
    "retention",
    "rs_endurance",
    "fi",
    "leak",
    "neuron_iv",
    "oscillator",
]


#: The device each protocol is a measurement *of*, which is how an operator
#: looks for one -- nobody opens the program wanting "experiment 23", they
#: want "the retention measurement for a synapse".  The grouping drives the
#: GUI navigator and ``list``; the keys above stay the stable CLI names.
EXPERIMENT_FAMILIES = (
    (
        "Synapse",
        "Analog weight update, plasticity and the window it holds",
        (
            "synapse",
            "epsc",
            "stdp",
            "ppf",
            "width_sweep",
            "incremental_width",
            "incremental_amplitude",
            "metrics",
            "kinetics",
            "rs_speed",
            "srdp",
            "stp",
            "mlc",
            "retention",
            "rs_endurance",
        ),
    ),
    (
        "Neuron",
        "Threshold switching, integration and firing statistics",
        (
            "integrate_fire",
            "relaxation",
            "stochastic",
            "refractory",
            "fi",
            "leak",
            "neuron_iv",
            "oscillator",
        ),
    ),
    (
        "Ferroelectric",
        "Switched polarization, hysteresis and fatigue",
        (
            "pund",
            "pv",
            "fe_iv",
            "afe",
            "afe_pund",
            "ftj",
            "fe_speed",
            "fe_endurance",
            "fe_accumulation",
        ),
    ),
)


def family_of(key: str) -> str:
    """The family ``key`` belongs to, or ``"Other"`` if it is unclassified.

    An unclassified experiment is still reachable -- the navigator puts it
    under "Other" rather than hiding it, because a protocol that exists but
    cannot be found is worse than one that is filed oddly.
    """
    for name, _desc, keys in EXPERIMENT_FAMILIES:
        if key in keys:
            return name
    return "Other"


def experiments_by_family() -> List[tuple]:
    """``[(family, description, [Experiment, ...]), ...]`` in display order.

    Every registered experiment appears exactly once; any that no family
    claims is collected into a trailing "Other" group.
    """
    everything = {e.key: e for e in all_experiments()}
    out = []
    for name, desc, keys in EXPERIMENT_FAMILIES:
        members = [everything.pop(k) for k in keys if k in everything]
        if members:
            out.append((name, desc, members))
    if everything:
        out.append(("Other", "Not yet filed under a device family",
                    list(everything.values())))
    return out


def get_experiment(key: str) -> Experiment:
    """Look up one experiment by its :attr:`Experiment.key`."""
    k = key.strip().lower().replace("-", "_")
    for exp in all_experiments():
        if exp.key == k:
            return exp
    raise KeyError(
        f"unknown experiment {key!r}; choose from {', '.join(EXPERIMENT_KEYS)}"
    )
