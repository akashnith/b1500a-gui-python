"""Open this in Thonny and press F5.

A script, not a GUI: Thonny is where you keep a measurement you want to read
and change line by line, and the parts of this package that matter there are
the protocols and the analysis, not the window.

It runs against the **simulator** as written, so it works with nothing
connected. Change DEMO to False when the instrument is on the bus.

If Thonny says "No module named 'b1500_studio'", it is running from the
wrong folder -- see the note at the bottom.
"""

import os
import sys

# Thonny runs a script from wherever the file is, so make the repository
# root importable no matter where this file was opened from.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from b1500_studio.core.fields import params_from_fields          # noqa: E402
from b1500_studio.core.storage import save_run                   # noqa: E402
from b1500_studio.protocols import SmuSettings, get_smu_protocol  # noqa: E402

# --------------------------------------------------------------- settings

DEMO = True          # False talks to the real instrument
ADDRESS = "GPIB0::16::INSTR"

# The wiring and the one limit that holds for every protocol. The force
# channel drives and measures; the return channel must be connected, or the
# circuit is open and every reading sits at the noise floor -- which looks
# like a very good insulator rather than like a mistake.
SETUP = SmuSettings(force_ch=1, common_ch=2, vmax=3.0)

# Which measurement, and what to change about it. Everything not named here
# keeps the protocol's own default; `b1500 smu list iv` prints them all.
PROTOCOL = "iv"
OVERRIDES = {
    "sweep_v_pos_v": "1.5",     # magnitude of the positive limb
    "sweep_v_neg_v": "1.5",     # magnitude of the negative limb, entered positive
    "sweep_step_v": "0.05",
    "step_dwell_s": "10m",
    "icomp": "1m",              # compliance: the real protection
    "read_voltage_v": "0.2",    # where the ON/OFF ratio is read off the loop
}

# Where finished runs are written. One folder per run, holding the data, the
# figure and a manifest of what produced them.
DATA_FOLDER = os.path.join(os.path.expanduser("~"), "B1500-Studio-Data")


# ------------------------------------------------------------------- run

def main() -> int:
    protocol = get_smu_protocol(PROTOCOL)
    params = params_from_fields(protocol.fields(), OVERRIDES)

    # Check the limits before anything is applied. This needs no instrument,
    # so it is also the fastest way to find a typo in OVERRIDES.
    ok, message = protocol.verify(params, SETUP)
    print(message)
    if not ok:
        return 1

    if DEMO:
        from b1500_studio.instruments.demo import DemoSmu

        smu = DemoSmu()
    else:
        from b1500_studio.instruments.smu import Smu

        smu = Smu(ADDRESS)

    smu.open()
    try:
        runs = protocol.run(smu, params, SETUP, log=lambda m: print(" ", m))
    finally:
        # Never leave a device biased, whatever happened above.
        smu.safe_zero(SETUP.channels())
        smu.close()

    for run in runs:
        print(f"\n--- {run.label}  ({len(run.x)} points)")
        for name, value in run.metrics:
            print(f"    {name:<22} = {value:.6g}")
        hits = run.compliance_hits()
        if hits:
            # A clipped point is not a measurement, and anything computed
            # across one is wrong.
            print(f"    ! compliance hit on {hits} point(s)")
        for note in run.notes:
            print(f"    ! {note}")

    # The figure the GUI would have drawn, saved rather than shown -- Thonny
    # has no canvas, and a file is what you want from a script anyway.
    draw = getattr(protocol, "draw_with", None)
    grid = draw(runs, params) if draw else protocol.draw(runs)

    from b1500_studio.protocols.smu import export_smu_csv

    saved = save_run(
        DATA_FOLDER,
        key=protocol.key,
        name=protocol.name,
        write_csvs=lambda folder: export_smu_csv(
            runs, folder, protocol.key.replace(".", "_")),
        grid=grid,
        parameters=params.s,
        setup={"mode": "demo" if DEMO else "hardware",
               "force_ch": SETUP.force_ch, "common_ch": SETUP.common_ch,
               "vmax_V": SETUP.vmax},
        metrics=[m for r in runs for m in r.metrics],
        notes=[n for r in runs for n in r.notes],
    )
    print(f"\nsaved to {saved.folder}")
    for path in saved.files:
        print("   ", os.path.basename(path))
    if not saved.ok:
        print("  ! save incomplete:", saved.error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------- if stuck
#
# "No module named 'b1500_studio'"
#     Thonny is using an interpreter that cannot see this repository. The
#     sys.path line at the top handles the usual case; if it persists, run
#     this once in Thonny's shell, with the repository path as it is on your
#     machine:
#
#         >>> import sys; sys.path.insert(0, r"C:\path\to\b1500-studio-python")
#
#     Or install the package into Thonny's interpreter, which fixes it for
#     good:  Tools > Open system shell..., then
#
#         pip install -e "C:\path\to\b1500-studio-python"
#
# "No module named 'PySide6'"
#     Only the window needs it; everything in this script does not. To use
#     the GUI from Thonny's interpreter:  pip install PySide6
#
# The instrument does not answer
#     python -m b1500_studio check      (reports, in evidence order, why)
