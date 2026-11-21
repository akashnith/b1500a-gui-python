"""Command line interface.

    python -m b1500_studio check
    python -m b1500_studio list [KEY]
    python -m b1500_studio verify KEY [--set k=v ...]
    python -m b1500_studio preview KEY [--set k=v ...] [--out preview.png]
    python -m b1500_studio run KEY --demo  [--set k=v ...] [--out DIR] [--png]
    python -m b1500_studio run KEY --hardware --addr GPIB0::16::INSTR
    python -m b1500_studio gui
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

from .core.driver import Hardware
from .experiments import all_experiments, get_experiment, params_from_fields
from .experiments.base import Severity
from .core.plan import parse_si
from .core.runner import Runner, export_results
from .instruments.smu import DEFAULT_RESOURCE, I_RANGES
from .core.verification import verify_experiment

__all__ = ["main"]


def _hardware_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("instrument")
    g.add_argument("--addr", default="GPIB0::16::INSTR", help="VISA address")
    g.add_argument("--dll", default=None, help="path to wgfmu.dll")
    g.add_argument("--force-ch", type=int, default=101)
    g.add_argument("--sink-ch", type=int, default=102)
    g.add_argument("--force-meas", default="V", choices=["V", "I", "0"])
    g.add_argument("--sink-meas", default="I", choices=["V", "I", "0"])
    g.add_argument("--irange", type=int, default=4,
                   help="0..4 = 1uA,10uA,100uA,1mA,10mA")
    g.add_argument("--vmax", type=float, default=1.0,
                   help="safety limit on |V| (default 1.0)")
    g.add_argument("--edge", default="10n")
    g.add_argument("--settle", type=float, default=0.3)
    g.add_argument("--bias", type=float, default=0.0, help="sink bias (V)")
    g.add_argument("--maxpts", default="200k")


def _hardware_from(a) -> Hardware:
    return Hardware(
        force_ch=a.force_ch,
        sink_ch=a.sink_ch,
        force_meas=a.force_meas,
        sink_meas=a.sink_meas,
        irange=a.irange,
        vmax=a.vmax,
        edge=parse_si(a.edge),
        settle=a.settle,
        sink_bias=a.bias,
        max_points=int(parse_si(a.maxpts)),
    )


def _overrides(pairs: Optional[List[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------- commands


def cmd_list(a) -> int:
    if a.key:
        exp = get_experiment(a.key)
        print(f"{exp.key}: {exp.name}\n")
        group = None
        for f in exp.fields():
            if f.get("group") != group:
                group = f.get("group")
                print(f"  [{group}]")
            default = f["default"]
            if f["kind"] == "combo":
                default = f"{f['choices'][int(default)]}  (choices: {', '.join(f['choices'])})"
            print(f"    {f['name']:22} = {default}")
            if f.get("help"):
                print(f"    {'':22}   {f['help']}")
        return 0
    for exp in all_experiments():
        print(f"  {exp.key:16} {exp.name}")
    return 0


def cmd_check(a) -> int:
    from .instruments.diagnostics import diagnose

    d = diagnose(address=a.addr, dll_path=a.dll, timeout=a.timeout,
                 probe_instrument=not a.no_instrument)
    print(d.text())
    return 0 if d.ok else 1


def cmd_verify(a) -> int:
    exp = get_experiment(a.key)
    p = params_from_fields(exp.fields(), _overrides(a.set))
    res = verify_experiment(exp, p, _hardware_from(a))
    print(res.summary_text)
    for issue in res.issues:
        print(f"  [{issue.severity}] {issue.title}: {issue.detail}")
    print(
        f"  runs={res.run_count}  total={res.total_duration:.6g} s  "
        f"max|V|={res.max_abs_v:.4g} V  points/run<={res.total_points}"
    )
    return 0 if res.ok else 1


def cmd_preview(a) -> int:
    from .experiments.preview import draw_preview
    from .core.plotting import save

    exp = get_experiment(a.key)
    p = params_from_fields(exp.fields(), _overrides(a.set))
    hw = _hardware_from(a)
    specs = exp.build(p, hw)
    grid = draw_preview(specs, exp.name)
    out = a.out or f"{exp.key}_preview.png"
    print("wrote", save(grid, out))
    return 0


def cmd_run(a) -> int:
    exp = get_experiment(a.key)
    p = params_from_fields(exp.fields(), _overrides(a.set))
    hw = _hardware_from(a)

    res = verify_experiment(exp, p, hw)
    print(res.summary_text)
    for issue in res.issues:
        if issue.severity != Severity.INFO:
            print(f"  [{issue.severity}] {issue.title}: {issue.detail}")
    if not res.ok:
        return 1
    if res.has_warnings and a.hardware and not a.yes:
        reply = input("Warnings present. Continue on hardware? [y/N] ").strip()
        if reply.lower() not in ("y", "yes"):
            print("cancelled")
            return 1

    if a.hardware:
        from .instruments.session import RemoteWgfmu

        driver = RemoteWgfmu(a.dll)
    else:
        from .instruments.demo import DemoDriver

        driver = DemoDriver()

    runner = Runner(driver, hw, log=lambda m: print("   ", m))
    try:
        if a.hardware:
            print(f"connecting to {a.addr} ...")
            driver.open(a.addr)
            print("channels:", driver.channels())
        report = runner.run(exp, p)
    except KeyboardInterrupt:
        runner.abort()
        print("aborted")
        return 1
    finally:
        try:
            driver.close()
        except Exception:
            pass

    if report.error:
        print("ERROR:", report.error)
        return 1

    for R in report.results:
        print(f"\n--- {R.label}")
        for key, value in R.summary:
            print(f"    {key:34} = {value:.6g}")
        for note in R.notes:
            print(f"    ! {note}")

    if a.out:
        paths = export_results(report, a.out)
        for path in paths:
            print("wrote", path)
        if a.png:
            from .core.plotting import save

            png = os.path.join(a.out, f"{exp.key}.png")
            print("wrote", save(exp.draw(report.results), png))
    return 0


# ------------------------------------------------------------- SMU commands


def _open_smu(a):
    """Return an opened SMU: the real instrument, or the simulated one."""
    logger = (lambda m: print("   ", m)) if getattr(a, "verbose", False) else None
    if getattr(a, "hardware", False):
        from .instruments.smu import Smu

        smu = Smu(a.addr, log=logger)
    else:
        from .instruments.demo import DemoSmu

        smu = DemoSmu(log=logger)
    smu.open()
    return smu


def _release(smu, a) -> None:
    chans = [a.force_ch] + ([a.common_ch] if a.common_ch else [])
    try:
        smu.safe_zero(chans)
    finally:
        smu.close()


def cmd_smu_list(a) -> int:
    """Name every SMU protocol, or print one protocol's fields."""
    from .protocols import all_smu_protocols, get_smu_protocol

    if getattr(a, "key", None):
        protocol = get_smu_protocol(a.key)
        print(f"{protocol.key}  --  {protocol.name}")
        group = None
        for f in protocol.fields():
            if f.get("group") != group:
                group = f.get("group")
                print(f"\n  [{group}]")
            default = f["default"]
            if f["kind"] == "combo":
                default = f["choices"][int(default)]
            print(f"    {f['name']:<18} {f['label']:<32} = {default}")
            if f.get("help"):
                print(f"    {'':<18} {f['help']}")
        return 0
    for protocol in all_smu_protocols():
        print(f"  {protocol.key:<16} {protocol.name}")
    return 0


def cmd_smu_protocol(a) -> int:
    """Run any catalogued SMU protocol, overriding fields with --set."""
    import os

    from .core.fields import params_from_fields
    from .protocols import SmuSettings, get_smu_protocol
    from .protocols.smu import export_smu_csv

    protocol = get_smu_protocol(a.key)
    overrides = {}
    for item in a.set or []:
        if "=" not in item:
            print(f"refused: --set wants name=value, got {item!r}")
            return 1
        name, value = item.split("=", 1)
        overrides[name.strip()] = value.strip()

    known = {f["name"] for f in protocol.fields()}
    unknown = sorted(set(overrides) - known)
    if unknown:
        # A typo in a field name must not pass silently: the run would go
        # ahead with the default, and the operator would read the result as
        # if their value had been used.
        print(f"refused: {protocol.key} has no field(s) {', '.join(unknown)}")
        print(f"         try:  b1500 smu list {protocol.key}")
        return 1

    try:
        params = params_from_fields(protocol.fields(), overrides)
    except ValueError as exc:
        print(f"refused: {exc}")
        return 1

    settings = SmuSettings(
        force_ch=a.force_ch, common_ch=a.common_ch or 0,
        common_bias=parse_si(a.common_bias), vmax=a.vmax,
    )
    ok, message = protocol.verify(params, settings)
    print(message)
    if not ok:
        return 1

    smu = _open_smu(a)
    logger = (lambda m: print("   ", m)) if getattr(a, "verbose", False) else None
    try:
        runs = protocol.run(smu, params, settings, log=logger)
    finally:
        _release(smu, a)

    for run in runs:
        _report_run(run)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        stem = protocol.key.replace(".", "_")
        for path in export_smu_csv(runs, a.out, stem):
            print("wrote", path)
        if a.png:
            from .core.plotting import save

            draw = getattr(protocol, "draw_with", None)
            grid = draw(runs, params) if draw else protocol.draw(runs)
            print("wrote", save(grid, os.path.join(a.out, f"{stem}.png")))
    return 0


def cmd_smu_modules(a) -> int:
    smu = _open_smu(a)
    try:
        print(smu.idn())
        print()
        for m in smu.scan():
            chans = ", ".join(str(c) for c in m.channels)
            print(f"  slot {m.slot}  {m.model:8} {m.kind:6} {m.family:6} channels: {chans}")
    finally:
        smu.close()
    return 0


def cmd_smu_spot(a) -> int:
    icomp = parse_si(a.icomp)
    volts = parse_si(a.volts)
    smu = _open_smu(a)
    try:
        smu.setup(a.force_ch, a.common_ch, icomp, a.plc)
        value, status = smu.read_at(
            a.force_ch, volts, icomp, I_RANGES[a.rng], settle=parse_si(a.settle)
        )
        r = volts / value if abs(value) > 1e-15 else float("inf")
        flag = "   [COMPLIANCE]" if status in ("C", "T") else ""
        print(f"V = {volts:+.6g} V    I = {value:+.6g} A    R = {r:.6g} ohm"
              f"    status={status}{flag}")
    finally:
        _release(smu, a)
    return 0


def cmd_smu_iv(a) -> int:
    import os

    from .instruments.smu import export_sweep_csv, iv_panels

    icomp = parse_si(a.icomp)
    start, stop = parse_si(a.start), parse_si(a.stop)
    reach = max(abs(start), abs(stop))
    if a.hardware and reach > a.vmax:
        print(f"refused: |V| = {reach:g} V exceeds the --vmax limit {a.vmax:g} V")
        return 1

    smu = _open_smu(a)
    try:
        smu.setup(a.force_ch, a.common_ch, icomp, a.plc)
        smu.sweep_setup(a.force_ch, I_RANGES[a.rng], a.hold, a.delay, a.sdelay)
        result = smu.sweep(
            a.force_ch, start, stop, a.points, icomp, double=a.double,
            hold=a.hold, delay=a.delay, sdelay=a.sdelay, plc=a.plc,
            label=a.label or "",
        )
        smu.sweep_end()
    finally:
        _release(smu, a)

    res = [x for x in (abs(v) for v in result.resistance()) if x == x and x > 0]
    print(f"\n--- {result.label}")
    print(f"    points           = {len(result)}")
    print(f"    I max            = {max((abs(x) for x in result.i), default=0):.6g} A")
    if res:
        print(f"    R min            = {min(res):.6g} ohm")
        print(f"    R max            = {max(res):.6g} ohm")
        print(f"    on/off ratio     = {max(res) / min(res):.6g}")
    if result.compliance_hits:
        print(f"    ! compliance hit on {result.compliance_hits} point(s)")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        print("wrote", export_sweep_csv(result, os.path.join(a.out, "smu_iv.csv")))
        if a.png:
            from .core.plotting import save

            print("wrote", save(iv_panels([result]), os.path.join(a.out, "smu_iv.png")))
    return 0


def _open_cmu(a):
    """A connected MFCMU, real or simulated."""
    logger = (lambda m: print(" ", m)) if getattr(a, "verbose", False) else None
    if getattr(a, "hardware", False):
        from .instruments.cmu import Cmu

        cmu = Cmu(a.addr, channel=a.cmu_ch, log=logger)
    else:
        from .instruments.cmu import DemoCmu

        cmu = DemoCmu(log=logger)
    cmu.open()
    return cmu


def _report_cv(result, a) -> None:
    from .core.ferro import permittivity

    print(f"\n--- {result.label}  ({len(result)} points)")
    eps = result.permittivity()
    if eps:
        print(f"    eps_r              = {min(eps):.4g} .. {max(eps):.4g}")
    caps = [p.capacitance_f for p in result.points]
    if caps:
        print(f"    C                  = {min(caps):.4g} .. {max(caps):.4g} F")
    window = result.memory_window_v()
    if window == window:  # not NaN
        print(f"    memory_window_V    = {window:.4g}")


def cmd_cmu_cv(a) -> int:
    from .core.plan import parse_si

    cmu = _open_cmu(a)
    try:
        result = cmu.cv_sweep(
            vmax=parse_si(a.vmax),
            steps=a.steps,
            frequency_hz=parse_si(a.freq),
            ac_level_v=parse_si(a.ac),
            hold_s=parse_si(a.hold),
            step_delay_s=parse_si(a.delay),
            area_cm2=parse_si(a.area),
            thickness_nm=parse_si(a.thickness) * 1.0e9,
        )
    finally:
        cmu.close()
    _report_cv(result, a)
    return 0


def cmd_cmu_rayleigh(a) -> int:
    from .instruments.cmu import rayleigh_fit
    from .core.plan import parse_list, parse_si

    levels = parse_list(a.levels)
    cmu = _open_cmu(a)
    try:
        result = cmu.rayleigh_sweep(
            ac_levels_v=levels,
            dc_bias_v=parse_si(a.bias),
            frequency_hz=parse_si(a.freq),
            step_delay_s=parse_si(a.delay),
            area_cm2=parse_si(a.area),
            thickness_nm=parse_si(a.thickness) * 1.0e9,
        )
    finally:
        cmu.close()
    _report_cv(result, a)
    fit = rayleigh_fit(
        [p.ac_level_v for p in result.points],
        [p.capacitance_f for p in result.points],
        result.area_cm2,
        result.thickness_nm,
    )
    for key, value in fit.items():
        print(f"    {key:<18} = {value:.6g}")
    if not fit:
        print("    (not enough usable points to fit the Rayleigh law)")
    return 0


def _report_run(run) -> int:
    """Print one SmuRun the way every other command prints its result."""
    print(f"\n--- {run.label}  ({len(run.x)} points)")
    for name, value in run.metrics:
        print(f"    {name:<28} = {value:.6g}")
    hits = run.compliance_hits()
    if hits:
        print(f"    compliance_hits              = {hits}")
    for note in run.notes:
        print(f"    ! {note}")
    return 0


def _smu_protocol(a, call) -> int:
    """Open the SMU, run one protocol function, release the channels."""
    from .core.plan import parse_si

    smu = _open_smu(a)
    try:
        smu.setup(a.force_ch, a.common_ch, parse_si(a.icomp), a.plc)
        run = call(smu, parse_si(a.icomp))
    finally:
        _release(smu, a)
    return _report_run(run)


def cmd_smu_forming(a) -> int:
    from .core.plan import parse_si
    from .protocols.smu import forming

    return _smu_protocol(a, lambda smu, ic: forming(
        smu, a.force_ch, parse_si(a.vstart), parse_si(a.vstop),
        parse_si(a.step), ic, parse_si(a.settle)))


def cmd_smu_read(a) -> int:
    from .core.plan import parse_si
    from .protocols.smu import read_retention

    return _smu_protocol(a, lambda smu, ic: read_retention(
        smu, a.force_ch, parse_si(a.vread), ic, parse_si(a.duration),
        a.points))


def cmd_smu_ndr(a) -> int:
    from .core.plan import parse_si
    from .protocols.smu import ndr

    return _smu_protocol(a, lambda smu, ic: ndr(
        smu, a.force_ch, parse_si(a.vmax), parse_si(a.step), ic,
        parse_si(a.settle)))


def cmd_smu_threshold(a) -> int:
    from .core.plan import parse_si
    from .protocols.smu import threshold_statistics

    return _smu_protocol(a, lambda smu, ic: threshold_statistics(
        smu, a.force_ch, parse_si(a.vmax), parse_si(a.step), ic, a.trials,
        parse_si(a.vreset)))


def cmd_smu_pulse(a) -> int:
    from .core.plan import parse_si
    from .protocols.smu import slow_pulse_ltp_ltd

    return _smu_protocol(a, lambda smu, ic: slow_pulse_ltp_ltd(
        smu, a.force_ch, parse_si(a.vpot), parse_si(a.vdep),
        parse_si(a.width), a.pulses, a.cycles, parse_si(a.vread), ic))


def cmd_smu_endurance(a) -> int:
    from .core.plan import parse_si
    from .protocols.smu import dc_endurance

    return _smu_protocol(a, lambda smu, ic: dc_endurance(
        smu, a.force_ch, parse_si(a.vset), parse_si(a.vreset),
        parse_si(a.width), a.cycles, a.read_every, parse_si(a.vread), ic))


def cmd_smu_resistance(a) -> int:
    from .core.plan import parse_si
    from .protocols.smu import probe_resistance

    return _smu_protocol(a, lambda smu, ic: probe_resistance(
        smu, a.force_ch, parse_si(a.vread), ic, a.repeats,
        a.four_probe, a.sense_ch))


def cmd_gui(a) -> int:
    from .gui.app import main as gui_main

    return gui_main(
        address=a.addr, dll=a.dll, demo=not a.hardware
    )


# -------------------------------------------------------------------- main


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="b1500-studio",
        description="B1500 Studio - Keysight B1500A: SMU DC, MFCMU capacitance, WGFMU pulses",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list experiments, or one's fields")
    p_list.add_argument("key", nargs="?")
    p_list.set_defaults(func=cmd_list)

    p_check = sub.add_parser("check", help="diagnose the VISA / WGFMU connection")
    p_check.add_argument("--addr", default="GPIB0::16::INSTR")
    p_check.add_argument("--dll", default=None)
    p_check.add_argument("--timeout", type=float, default=25.0)
    p_check.add_argument("--no-instrument", action="store_true",
                         help="skip the WGFMU_openSession probe")
    p_check.set_defaults(func=cmd_check)

    p_verify = sub.add_parser("verify", help="dry-run checks, no instrument")
    p_verify.add_argument("key")
    p_verify.add_argument("--set", action="append", metavar="k=v")
    _hardware_args(p_verify)
    p_verify.set_defaults(func=cmd_verify)

    p_prev = sub.add_parser("preview", help="save a waveform preview PNG")
    p_prev.add_argument("key")
    p_prev.add_argument("--set", action="append", metavar="k=v")
    p_prev.add_argument("--out", default=None)
    _hardware_args(p_prev)
    p_prev.set_defaults(func=cmd_preview)

    p_run = sub.add_parser("run", help="run an experiment")
    p_run.add_argument("key")
    p_run.add_argument("--set", action="append", metavar="k=v")
    mode = p_run.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", default=True)
    mode.add_argument("--hardware", action="store_true")
    p_run.add_argument("--out", default=None, help="folder for CSV output")
    p_run.add_argument("--png", action="store_true", help="also save the figure")
    p_run.add_argument("--yes", action="store_true", help="skip the warning prompt")
    _hardware_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # ---- SMU (pyvisa / FLEX): DC spot readings and staircase I-V sweeps
    p_smu = sub.add_parser("smu", help="SMU measurements (DC / staircase I-V)")
    smu_sub = p_smu.add_subparsers(dest="smu_cmd", required=True)

    def smu_common(sp, channels: bool = True) -> None:
        sp.add_argument("--addr", default=DEFAULT_RESOURCE, help="VISA address")
        mode = sp.add_mutually_exclusive_group()
        mode.add_argument("--demo", action="store_true", default=True,
                          help="simulated memristor (default)")
        mode.add_argument("--hardware", action="store_true")
        sp.add_argument("--verbose", action="store_true",
                        help="echo every FLEX command")
        if channels:
            sp.add_argument("--force-ch", type=int, default=1,
                            help="SMU channel that forces (default 1)")
            sp.add_argument("--common-ch", type=int, default=None,
                            help="SMU channel held at 0 V")
            sp.add_argument("--icomp", default="1m", help="current compliance")
            sp.add_argument("--plc", type=float, default=1.0,
                            help="integration time in power line cycles")
            sp.add_argument("--rng", default="Auto", choices=list(I_RANGES),
                            help="current measurement range")

    p_cmu = sub.add_parser("cmu", help="MFCMU capacitance measurements")
    cmu_sub = p_cmu.add_subparsers(dest="cmu_cmd", required=True)

    def cmu_common(sp) -> None:
        sp.add_argument("--addr", default=DEFAULT_RESOURCE, help="VISA address")
        mode = sp.add_mutually_exclusive_group()
        mode.add_argument("--demo", action="store_true", default=True,
                          help="simulated capacitor (default)")
        mode.add_argument("--hardware", action="store_true")
        sp.add_argument("--verbose", action="store_true",
                        help="echo every FLEX command")
        sp.add_argument("--cmu-ch", type=int, default=0,
                        help="MFCMU slot; 0 finds the B1520A from UNT?")
        sp.add_argument("--freq", default="1M", help="measurement frequency")
        sp.add_argument("--delay", default="10m", help="settle before each point")
        sp.add_argument("--area", default="1e-4", help="electrode area in cm^2")
        sp.add_argument("--thickness", default="10n", help="film thickness")

    p_cv = cmu_sub.add_parser("cv", help="C-V butterfly (DC bias sweep)")
    cmu_common(p_cv)
    p_cv.add_argument("--vmax", default="3", help="sweep +/- this bias")
    p_cv.add_argument("--steps", type=int, default=101)
    p_cv.add_argument("--ac", default="30m", help="AC oscillator level")
    p_cv.add_argument("--hold", default="100m", help="hold before the sweep")
    p_cv.set_defaults(func=cmd_cmu_cv)

    p_ray = cmu_sub.add_parser(
        "rayleigh", help="permittivity against AC drive amplitude")
    cmu_common(p_ray)
    p_ray.add_argument("--levels", default="lin(10m,200m,10)",
                       help="AC oscillator levels to step through")
    p_ray.add_argument("--bias", default="0", help="fixed DC bias")
    p_ray.set_defaults(func=cmd_cmu_rayleigh)

    p_plist = smu_sub.add_parser(
        "list", help="name every SMU protocol, or show one's fields")
    p_plist.add_argument("key", nargs="?", help="e.g. iv, forming, ndr")
    p_plist.set_defaults(func=cmd_smu_list)
    smu_common(p_plist, channels=False)

    p_proto = smu_sub.add_parser(
        "protocol",
        help="run any SMU protocol by key, the same code the GUI runs")
    p_proto.add_argument("key", help="e.g. iv, forming, threshold")
    p_proto.add_argument(
        "--set", action="append", metavar="NAME=VALUE",
        help="override one field; repeat for more (b1500 smu list KEY)")
    p_proto.add_argument("--common-bias", default="0",
                         help="bias held on the return channel")
    p_proto.add_argument("--out", default=None, help="folder for CSV / PNG")
    p_proto.add_argument("--png", action="store_true", help="also save a figure")
    p_proto.add_argument("--vmax", type=float, default=5.0,
                         help="software ceiling on |V| (default 5)")
    smu_common(p_proto)
    p_proto.set_defaults(func=cmd_smu_protocol)

    p_mod = smu_sub.add_parser("modules", help="*IDN? and the installed modules")
    smu_common(p_mod, channels=False)
    p_mod.set_defaults(func=cmd_smu_modules)

    p_spot = smu_sub.add_parser("spot", help="one biased current reading")
    smu_common(p_spot)
    p_spot.add_argument("--volts", default="0.1", help="read bias (V)")
    p_spot.add_argument("--settle", default="0", help="settle time before reading")
    p_spot.set_defaults(func=cmd_smu_spot)

    p_form = smu_sub.add_parser("forming", help="ramp until the device forms")
    smu_common(p_form)
    p_form.add_argument("--vstart", default="0.5")
    p_form.add_argument("--vstop", default="5")
    p_form.add_argument("--step", default="0.05")
    p_form.add_argument("--settle", default="10m")
    p_form.set_defaults(func=cmd_smu_forming)

    p_read = smu_sub.add_parser("read", help="retention / read disturb")
    smu_common(p_read)
    p_read.add_argument("--vread", default="0.1")
    p_read.add_argument("--duration", default="10")
    p_read.add_argument("--points", type=int, default=20)
    p_read.set_defaults(func=cmd_smu_read)

    p_ndr = smu_sub.add_parser("ndr", help="negative differential resistance")
    smu_common(p_ndr)
    p_ndr.add_argument("--vmax", default="1.5")
    p_ndr.add_argument("--step", default="0.02")
    p_ndr.add_argument("--settle", default="5m")
    p_ndr.set_defaults(func=cmd_smu_ndr)

    p_thr = smu_sub.add_parser("threshold", help="threshold voltage statistics")
    smu_common(p_thr)
    p_thr.add_argument("--vmax", default="1.5")
    p_thr.add_argument("--step", default="0.02")
    p_thr.add_argument("--vreset", default="-1")
    p_thr.add_argument("--trials", type=int, default=10)
    p_thr.set_defaults(func=cmd_smu_threshold)

    p_pulse = smu_sub.add_parser("pulse", help="slow-pulse LTP / LTD")
    smu_common(p_pulse)
    p_pulse.add_argument("--vpot", default="0.7")
    p_pulse.add_argument("--vdep", default="-0.7")
    p_pulse.add_argument("--width", default="100u")
    p_pulse.add_argument("--vread", default="0.1")
    p_pulse.add_argument("--pulses", type=int, default=20)
    p_pulse.add_argument("--cycles", type=int, default=1)
    p_pulse.set_defaults(func=cmd_smu_pulse)

    p_end = smu_sub.add_parser("endurance", help="DC SET/RESET endurance")
    smu_common(p_end)
    p_end.add_argument("--vset", default="1")
    p_end.add_argument("--vreset", default="-1")
    p_end.add_argument("--width", default="1m")
    p_end.add_argument("--vread", default="0.1")
    p_end.add_argument("--cycles", type=int, default=50)
    p_end.add_argument("--read-every", type=int, default=5)
    p_end.set_defaults(func=cmd_smu_endurance)

    p_res = smu_sub.add_parser("resistance", help="2-/4-probe resistance")
    smu_common(p_res)
    p_res.add_argument("--vread", default="0.1")
    p_res.add_argument("--repeats", type=int, default=10)
    p_res.add_argument("--four-probe", action="store_true",
                       help="use --sense-ch for the voltage across the device")
    p_res.add_argument("--sense-ch", type=int, default=None)
    p_res.set_defaults(func=cmd_smu_resistance)

    p_iv = smu_sub.add_parser("iv", help="staircase I-V sweep")
    smu_common(p_iv)
    p_iv.add_argument("--start", default="-1", help="start voltage")
    p_iv.add_argument("--stop", default="1", help="stop voltage")
    p_iv.add_argument("--points", type=int, default=101)
    p_iv.add_argument("--double", action="store_true",
                      help="sweep up then back down (hysteresis loop)")
    p_iv.add_argument("--hold", type=float, default=0.0, help="hold time (s)")
    p_iv.add_argument("--delay", type=float, default=0.0, help="step delay (s)")
    p_iv.add_argument("--sdelay", type=float, default=0.0, help="step source delay (s)")
    p_iv.add_argument("--vmax", type=float, default=2.0,
                      help="safety limit on |V| for hardware sweeps")
    p_iv.add_argument("--label", default=None)
    p_iv.add_argument("--out", default=None, help="folder for CSV output")
    p_iv.add_argument("--png", action="store_true", help="also save the figure")
    p_iv.set_defaults(func=cmd_smu_iv)

    p_gui = sub.add_parser("gui", help="launch the graphical interface")
    p_gui.add_argument("--addr", default="GPIB0::16::INSTR")
    p_gui.add_argument("--dll", default=None)
    p_gui.add_argument("--hardware", action="store_true")
    p_gui.set_defaults(func=cmd_gui)

    a = ap.parse_args(argv)
    try:
        return a.func(a)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
