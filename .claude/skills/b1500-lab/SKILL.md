---
name: b1500-lab
description: Drive a Keysight B1500A through this project - SMU (FLEX, DC spot and staircase I-V) and B1530A WGFMU (nanosecond pulses, synapse and neuron tests). Use when the user asks to check the instrument or sample, run or tune a measurement, sweep I-V, pulse a device, interpret results, or diagnose a connection failure.
---

# B1500A lab automation

Everything goes through one CLI, run from the project root:

```
python -m b1500_studio <command> [...]
```

Two measurement paths share the one instrument:

| path | module | what it does |
|---|---|---|
| **SMU** | `smu.py` (pyvisa + FLEX) | DC spot readings, staircase I-V, ms pulses |
| **WGFMU** | `wgfmu_dll.py` + `session.py` (ctypes -> `wgfmu.dll`) | ns-µs pulse trains, synapse/neuron protocols |

Every command takes `--demo` (simulated device, the default) or `--hardware`.
**Never pass `--hardware` without the user's explicit approval for that
session**, and ask again before raising the voltage.

## Always start here

```
python -m b1500_studio check
```

It probes, in the order failures cascade: 64-bit Python -> `wgfmu.dll` present
-> Keysight IO services running -> `pyvisa viOpenDefaultRM()` -> `WGFMU_openSession()`.
Each probe runs in a killable subprocess, so a broken VISA stack reports a
timeout instead of hanging.

**If it reports stopped Keysight IO services, stop and tell the user.** Every
VISA call will block forever until they are started from an Administrator
PowerShell; no code change can work around it, and this breaks the SMU path,
the WGFMU path and Keysight Connection Expert alike.

## SMU workflow

```
python -m b1500_studio smu modules                  # *IDN? + installed slots
python -m b1500_studio smu spot --volts 0.2         # one biased reading
python -m b1500_studio smu iv --start -1 --stop 1 --points 101 --double \
       --icomp 1m --rng "Fixed 1 mA" --out results --png
```

- `--force-ch` forces, `--common-ch` sits at 0 V, `--icomp` is the compliance.
- `--double` gives the hysteresis loop; `--vmax` refuses sweeps beyond a limit.
- Read `on/off ratio`, `R min`/`R max`, and **always check for
  `! compliance hit`** - points in compliance are clipped, not measured.
- Current ranges come from `I_RANGES`; a fixed range is quieter than `Auto`
  when the current span is known.

## WGFMU workflow

```
python -m b1500_studio list                   # the eight experiments
python -m b1500_studio list synapse           # that experiment's parameters
python -m b1500_studio verify synapse --set syn_pot_v0=0.8 --vmax 1.0
python -m b1500_studio preview synapse --out preview.png
python -m b1500_studio run synapse --hardware --vmax 1.0 --out results --png
```

**Always `verify` before `run`.** Verification is a pure dry run: it builds
the waveforms, checks them against `--vmax`, the 10 ns grid and the point
limit, and applies physics checks (read-disturb, read/write ratio, duty
cycle, polarity). It never touches the instrument.

Experiments: `synapse`, `epsc`, `stdp`, `relaxation`, `integrate_fire`,
`stochastic`, `ppf`, `refractory`.

Parameters accept SI suffixes (`1u`, `500n`, `10m`, `200k`) and list
expressions (`0.5,0.7`, `lin(a,b,n)`, `log(a,b,n)`). Override any field with
`--set name=value`; names come from `list <key>`.

### Reading the metrics

- **Synapse**: `Gmax/Gmin` is the dynamic range (want >= 5-10). LTP/LTD `R2`
  near 1 means linear updates. Compare LTP and LTD slopes for symmetry.
- **PPF**: above 100 % is facilitation; `tau PPF` is the short-term memory time.
- **Relaxation**: compare `tau exp` with `tau KWW`; `beta` well below 1 means
  a broad distribution of relaxation times, not a single process.
- **Stochastic**: use the Wilson bounds, not the bare probability - below ~30
  trials the interval is wide and the run notes say so.
- **Refractory**: `t_abs` is where the second spike fails entirely, `t_rel`
  where it recovers to 90 %.

## Safety

- Get explicit approval before the first `--hardware` run of a session, and
  again before raising voltage or running long stress sequences.
- `--vmax` is a hard limit on |V| at the device; never raise it to get a run
  through without saying so plainly.
- **The WGFMU has no current compliance.** Resistive-switching devices need a
  series resistor or a suitable current range.
- On any failure - a non-zero exit, a `! compliance hit`, or a verification
  error - stop and report. Do not retry blindly at higher voltage.

## Checking the software

```
test.bat          (or: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest)
```

164 tests, no hardware required. They must pass after any code change. The
env var is needed only because an unrelated installed plugin (`nengo`) is
incompatible with this pytest version.
