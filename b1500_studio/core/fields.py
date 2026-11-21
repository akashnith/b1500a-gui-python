"""Input fields: what a protocol asks for, and what the answers parse to.

Every measurement in the program -- WGFMU pulse experiment or SMU DC
protocol -- declares its inputs as a list of these plain dictionaries, and
the GUI, the CLI and the tests all build their values through
:func:`params_from_fields`.  One source of truth: a field that exists has a
label, a default and a type wherever it is shown, and a default that does not
parse fails in the tests rather than in front of a device.

Descriptor keys::

    name     the parameter key the protocol reads
    label    what the operator sees
    default  the initial value (text, or an index for combo/check)
    help     tooltip, optional
    kind     "num" | "text" | "combo" | "check"
    choices  combo only
    group    form section caption, added by grouped()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from .plan import parse_si

__all__ = [
    "Params",
    "num_field",
    "text_field",
    "combo_field",
    "check_field",
    "grouped",
    "params_from_fields",
]


# ------------------------------------------------------------- descriptors


def num_field(name: str, label: str, default: str, help: str = "") -> dict:
    """A free-text field whose value is normally a number with SI suffix."""
    return {
        "name": name,
        "label": label,
        "default": default,
        "help": help,
        "kind": "num",
    }


def text_field(name: str, label: str, default: str, help: str = "") -> dict:
    """A field holding a list expression such as ``log(10u,100m,8)``."""
    return {
        "name": name,
        "label": label,
        "default": default,
        "help": help,
        "kind": "text",
    }


def combo_field(
    name: str, label: str, choices: List[str], default: int = 0, help: str = ""
) -> dict:
    """A drop-down; the parameter value is the selected index."""
    return {
        "name": name,
        "label": label,
        "default": default,
        "help": help,
        "kind": "combo",
        "choices": list(choices),
    }


def check_field(name: str, label: str, default: bool = False, help: str = "") -> dict:
    """A tick box; the parameter value is 0.0 or 1.0."""
    return {
        "name": name,
        "label": label,
        "default": bool(default),
        "help": help,
        "kind": "check",
    }


def grouped(group: str, *fields: dict) -> List[dict]:
    """Tag ``fields`` with a form-group caption."""
    out = []
    for f in fields:
        f = dict(f)
        f["group"] = group
        out.append(f)
    return out


# ------------------------------------------------------------------ values


@dataclass
class Params:
    """Field values from the UI: numbers plus the raw text of every field."""

    d: Dict[str, float] = field(default_factory=dict)
    s: Dict[str, str] = field(default_factory=dict)

    def num(self, k: str) -> float:
        if k not in self.d:
            raw = self.s.get(k)
            extra = f": '{raw}'" if raw is not None else ""
            raise ValueError(f"bad number in field '{k}'{extra}")
        return self.d[k]

    def integer(self, k: str) -> int:
        return int(round(self.num(k)))

    def flag(self, k: str) -> bool:
        return self.num(k) != 0.0

    def str(self, k: str) -> str:
        if k not in self.s:
            raise ValueError(f"missing field '{k}'")
        return self.s[k]

    def get(self, k: str, default: float) -> float:
        return self.d.get(k, default)


def params_from_fields(
    fields: List[dict], overrides: Optional[Mapping[str, object]] = None
) -> Params:
    """Build :class:`Params` from descriptors plus optional overrides.

    Mirrors the Qt ``gather()``: the raw text of every field always lands in
    ``Params.s``; it also lands in ``Params.d`` when it parses as a number.
    A field that does not parse only fails if a protocol actually asks for it
    as a number, which is what lets list expressions such as ``lin(a,b,n)``
    coexist with numeric fields.
    """
    overrides = dict(overrides or {})
    p = Params()
    for f in fields:
        name = f["name"]
        kind = f.get("kind", "num")
        raw = overrides.get(name, f["default"])

        if kind == "combo":
            choices = f.get("choices", [])
            if isinstance(raw, str) and not raw.lstrip("+-").isdigit():
                if raw not in choices:
                    raise ValueError(
                        f"field '{name}': {raw!r} is not one of {choices}"
                    )
                idx = choices.index(raw)
            else:
                idx = int(raw)
            if not 0 <= idx < max(len(choices), 1):
                raise ValueError(
                    f"field '{name}': index {idx} out of range for {choices}"
                )
            p.d[name] = float(idx)
            p.s[name] = choices[idx] if choices else str(idx)
            continue

        if kind == "check":
            if isinstance(raw, str):
                on = raw.strip().lower() in ("1", "true", "yes", "on")
            else:
                on = bool(raw)
            p.d[name] = 1.0 if on else 0.0
            p.s[name] = "1" if on else "0"
            continue

        text = str(raw)
        p.s[name] = text
        try:
            p.d[name] = parse_si(text)
        except Exception:
            pass  # list expressions and similar stay text-only

    # Allow overrides for keys not declared as fields (e.g. ad-hoc values).
    for k, v in overrides.items():
        if k not in p.s:
            p.s[k] = str(v)
            try:
                p.d[k] = parse_si(str(v))
            except Exception:
                pass
    return p
