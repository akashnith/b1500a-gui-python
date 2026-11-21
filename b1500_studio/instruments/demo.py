"""Simulated device and demo driver.

Faithful port of ``pkg/src/sim_device.cpp``: a non-volatile synapse and a
leaky threshold-switching neuron, so the whole application can be exercised
with no instrument attached.

The random stream differs from the C++ ``std::mt19937`` (numpy's generator is
used instead), so traces are not bit-identical to the Qt app -- only
statistically equivalent.
"""

from __future__ import annotations

import bisect
import math
import threading
import time
from typing import List

import numpy as np

from ..core.driver import Channel, Driver, Hardware, LogFn, Results, irange_amps
from ..core.plan import GRID, Plan

__all__ = ["simulate", "DemoDriver", "DemoSmu"]


# --- the simulated ferroelectric capacitor -------------------------------
#
# HZO-like, matching the device the Qt demo backend simulates: Ps 20 uC/cm^2,
# Vc about 1 V at 10 nm. Fixed, deliberately: it is one specific pretend
# device, so changing the `area` field of an experiment rescales the REPORTED
# polarization without changing what the simulator does -- exactly as changing
# the area you type would do against a real capacitor of fixed size.
FERRO_PS_UC_CM2 = 20.0    #: saturation polarization  [uC/cm^2]
# No coercive voltage constant: with Merz kinetics there isn't one. The
# apparent Vc is wherever the switching time happens to match the ramp, so it
# moves with frequency -- which is what a real ferroelectric does, and why a
# loop measured at 1 kHz and one at 1 MHz do not report the same Vc.
FERRO_AREA_CM2 = 1.0e-4   #: simulated electrode area [cm^2]
# Merz law, tau(V) = tau0 * exp(Va/|V|). The prefactor is chosen so that
# switching takes a useful fraction of a microsecond ramp at these
# amplitudes, which is what makes partial switching -- and therefore
# fe.speed and fe.accumulation -- observable at all.
#
# It also has to keep the switching CURRENT inside the WGFMU's 10 mA
# ceiling. 4 nC switched in 100 ns is 40 mA, which the instrument would
# clip, and a clipped spike integrates to the wrong charge. That is not a
# simulation artefact: it is why real PUND ramps are slow.
FERRO_TAU = 3.3e-8        #: switching time prefactor tau0  [s]
FERRO_VA = 5.45           #: Merz activation voltage, tau = tau0*exp(Va/|V|)
#
# Chosen so one microsecond at 0.8 V switches a few percent -- slow
# enough to be genuinely sub-coercive, fast enough that a train of
# them accumulates -- while 2.5 V switches within a microsecond ramp.

# --- the simulated ANTIFERROELECTRIC -------------------------------------
#
# Selected by ``plan.hint == "afe"``. The difference from the ferroelectric
# above is not a parameter change, it is a different state machine: the
# field-induced ferroelectric phase is only stable while the field holds it,
# so the device falls back on the way down. That is what produces a double
# loop with near-zero remanence, and it is why a conventional PUND measures
# nothing on one of these -- the induced phase is gone before U arrives.
AFE_V_FORWARD = 2.0   #: AFE -> FE, going out
AFE_V_BACKWARD = 1.2  #: FE -> AFE, coming back (lower: this is the hysteresis)
FERRO_CAP_F = 2.0e-10     #: linear dielectric capacitance [F]
FERRO_RLEAK = 2.0e8       #: ohmic leakage [ohm]


#: Short-term plasticity: how fast the fast component forgets, and how
#: much conductance it can add at full strength.
STP_TAU = 300e-6
STP_GAIN = 4.0e-5


def simulate(plan: Plan, hw: Hardware, seed: int = 7) -> Results:
    """Integrate the toy device model over ``plan`` and sample it."""
    neuron = plan.hint == "neuron"
    afe = plan.hint == "afe"
    ferro = plan.hint == "ferro" or afe
    x, y, m = 0.05, 0.0, 0.0  # non-volatile state, LRS filament, membrane
    stp = 0.0                 # short-term (fast, decaying) synaptic state
    pol = -1.0                # ferroelectric state, normalised to +/-1
    # Charge and elapsed time since the previous sample, so the reported
    # current is the BOX AVERAGE over the sample interval.
    #
    # Not the instantaneous value: a WGFMU sample is an average over its
    # averaging window, and a switching spike narrower than that window is
    # genuinely averaged, not caught at its peak. Reporting the instantaneous
    # current instead made the integral depend on whether a sample happened to
    # land on the spike -- a 20 uC/cm^2 device came back as 65, and the error
    # moved when the sample count changed. This is also why Keysight requires
    # rise_time / sample_interval > 10.
    q_accum = 0.0
    t_accum = 0.0
    rng = np.random.default_rng(seed)

    # Breakpoints of the piecewise-linear force waveform, in ticks.
    bt: List[float] = [0.0]
    bv: List[float] = [0.0]
    acc = 0.0
    for v in plan.vec:
        acc += float(v.n)
        bt.append(acc)
        bv.append(v.v)

    def v_at(t: float) -> float:
        j = bisect.bisect_right(bt, t)
        if j == 0:
            return bv[0]
        if j >= len(bt):
            return bv[-1]
        span = max(bt[j] - bt[j - 1], 1e-9)
        f = (t - bt[j - 1]) / span
        return bv[j - 1] + (bv[j] - bv[j - 1]) * f

    # Sample centres in ticks, in device order.
    st: List[float] = []
    for e in plan.ev:
        for k in range(e.points):
            st.append(float(e.t + k * e.interval) + e.avg / 2.0)

    grid = sorted(set(bt + st))

    n_samples = len(st)
    I = [0.0] * n_samples
    Vm = [0.0] * n_samples
    si = 0
    tc = 0.0

    for g in grid:
        if g > tc:
            dt = (g - tc) * GRID
            nsub = max(1, min(4000, int(math.ceil(dt / 2e-6))))
            h = dt / nsub
            for k in range(nsub):
                tm = tc + (k + 0.5) * (g - tc) / nsub
                V = v_at(tm)
                a = abs(V)
                if ferro:
                    # Bistable switching: the device flips up above +Vc and
                    # down below -Vc, and holds in between. That hold is what
                    # makes it ferroelectric rather than merely non-linear,
                    # and it is what lets U and D measure a device that P and
                    # N have already switched.
                    if afe:
                        # Induced, not remanent. The polarization follows the
                        # field up past V_forward and falls back below
                        # V_backward, so at zero field there is nothing left:
                        # near-zero remanence, two loops instead of one.
                        if V > AFE_V_FORWARD:
                            target = 1.0
                        elif V < -AFE_V_FORWARD:
                            target = -1.0
                        elif abs(V) < AFE_V_BACKWARD:
                            target = 0.0
                        else:
                            target = pol
                    else:
                        # Any field drives the polarization toward alignment;
                        # what stops it below "the coercive voltage" is not a
                        # threshold but the switching TIME, which Merz's law
                        # makes astronomically long at low field.
                        #
                        # Modelling Vc as a hard cutoff instead made
                        # sub-coercive pulses do exactly nothing, so
                        # accumulative switching could not be simulated at
                        # all -- and it is the measurement that decides
                        # whether a material works as an analog weight.
                        target = 1.0 if V > 0.0 else (-1.0 if V < 0.0 else pol)
                    # Merz law: the switching time falls exponentially with
                    # field, tau(V) = tau0 * exp(Va/|V|). A fixed tau would
                    # make switching speed independent of amplitude, which
                    # would turn the whole fe.speed measurement into a step
                    # function and leave nothing for a t50 to land on.
                    # The antiferroelectric's fall back to zero happens at LOW
                    # field, so it must not be gated on |V| > Vc the way the
                    # ferroelectric's switching is -- that gate would freeze
                    # the induced phase in place and turn it into a
                    # ferroelectric with remanence.
                    moving = target != pol
                    if moving:
                        tau = FERRO_TAU * math.exp(FERRO_VA / max(abs(V), 0.5))
                        step = (target - pol) * (1.0 - math.exp(-h / tau))
                    else:
                        step = 0.0
                    pol += step
                    # i = A dP/dt + C dV/dt + V/R. The first term is the
                    # switching current the measurement is after; the other
                    # two are what the P-U subtraction exists to remove.
                    # tm is in TICKS, h is in seconds: the substep width has
                    # to be expressed in ticks to step v_at(), and the divide
                    # has to be in seconds to give V/s.
                    half = 0.5 * (g - tc) / nsub
                    dv_dt = (v_at(tm + half) - v_at(tm - half)) / h
                    i_now = (
                        FERRO_AREA_CM2 * FERRO_PS_UC_CM2 * 1.0e-6 * (step / h)
                        + FERRO_CAP_F * dv_dt
                        + V / FERRO_RLEAK
                    )
                    q_accum += i_now * h
                    t_accum += h
                elif not neuron:
                    if V > 0.5:
                        x += (1 - x) * (1 - math.exp(-4e4 * (V - 0.5) * h))
                    elif V < -0.5:
                        x -= x * (1 - math.exp(-4e4 * (-V - 0.5) * h))
                    x -= (x - 0.2) * (1 - math.exp(-h / 5.0))  # retention loss
                    # Short-term plasticity: a fast component that each pulse
                    # kicks and that relaxes in a few hundred microseconds.
                    #
                    # Without it the device had only the non-volatile state,
                    # whose retention constant is seconds, so a pulse-train
                    # STP measurement saw a perfectly flat recovery -- the
                    # protocol ran and measured nothing, which is worse than
                    # failing.
                    if abs(V) > 0.5:
                        stp += (1.0 - stp) * (1 - math.exp(-2e4 * (abs(V) - 0.5) * h))
                    stp *= math.exp(-h / STP_TAU)
                else:
                    if a > 0.4:
                        m += (1 - m) * (1 - math.exp(-9e5 * (a - 0.4) * h))
                    m *= math.exp(-h / 30e-6)
                    if a > 0.7:
                        y += (1 - y) * (
                            1 - math.exp(-1e5 * math.exp((a - 0.7) / 0.05) * h)
                        )
                    if m > 0.6:
                        y, m = 1.0, 0.0
                    y *= math.exp(-h / 50e-6)  # LRS -> HRS relaxation
            tc = g

        while si < n_samples and st[si] <= g + 1e-9:
            V = v_at(st[si])
            if ferro:
                # Already an absolute current, not a conductance times bias:
                # a ferroelectric's signal is displacement current, which has
                # no meaning as G*V and is largest where V is merely passing
                # through, not where V is largest.
                cur = (q_accum / t_accum) if t_accum > 0.0 else 0.0
                q_accum = 0.0
                t_accum = 0.0
            else:
                # The short-term state adds to the non-volatile one, so a
                # train facilitates and then relaxes back toward whatever
                # the non-volatile state alone supports.
                G = (1e-8 + 1e-4 * y) if neuron else (
                    5e-6 + 9.5e-5 * x + STP_GAIN * stp
                )
                cur = G * V * (1 + 0.3 * V * V)
            cur += float(rng.normal()) * (0.01 * abs(cur) + 2e-9)
            lim = irange_amps(hw.irange)
            I[si] = max(-lim, min(lim, cur))
            Vm[si] = V + float(rng.normal()) * 2e-4
            si += 1

    tt, _ = plan.samples()
    r = Results()
    if hw.sink_meas != "0":
        r.ch[hw.sink_ch] = Channel(
            t=list(tt), y=I if hw.sink_meas == "I" else [0.0] * len(I)
        )
    if hw.force_meas != "0":
        r.ch[hw.force_ch] = Channel(
            t=list(tt), y=Vm if hw.force_meas == "V" else I
        )
    return r


class DemoDriver(Driver):
    """Driver that runs :func:`simulate` instead of touching an instrument."""

    def __init__(self, realistic_delay: bool = True) -> None:
        self._abort = threading.Event()
        self._realistic_delay = realistic_delay

    @property
    def name(self) -> str:
        return "DEMO"

    def open(self, address: str = "") -> None:  # no-op
        pass

    def close(self) -> None:  # no-op
        pass

    def channels(self) -> List[int]:
        return [101, 102]

    def abort(self) -> None:
        self._abort.set()

    def run(self, plan: Plan, hw: Hardware, log: LogFn = None) -> Results:
        self._abort.clear()
        plan.validate(hw.vmax, hw.max_points)
        results = simulate(plan, hw)
        if self._realistic_delay:
            wait_s = min(1.5, plan.duration() * 200.0 + 0.15)
            deadline = time.monotonic() + wait_s
            while time.monotonic() < deadline and not self._abort.is_set():
                time.sleep(0.025)
        return results


class DemoSmu:
    """Simulated B1500A SMU: a bipolar memristor, no instrument needed.

    Implements the same surface as :class:`b1500_studio.smu.Smu` so the CLI,
    the tests and demonstrations can exercise the whole SMU path offline.

    Device model: internal state ``x`` in [0, 1] sets the resistance between
    ``r_on`` and ``r_off``. Above ``v_set`` the device SETs, below
    ``-v_reset`` it RESETs, which produces the usual pinched hysteresis loop.
    """

    def __init__(
        self,
        resource: str = "DEMO",
        timeout_ms: int = 60_000,
        log=None,
        r_on: float = 2e3,
        r_off: float = 2e5,
        v_set: float = 0.6,
        v_reset: float = 0.6,
        seed: int = 11,
    ) -> None:
        self.resource = resource
        self.timeout_ms = timeout_ms
        self._log = log
        self.r_on, self.r_off = r_on, r_off
        self.v_set, self.v_reset = v_set, v_reset
        self._x = 0.0  # 0 = HRS, 1 = LRS
        self._open = False
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ plumbing

    @property
    def name(self) -> str:
        return "SMU-DEMO"

    @property
    def connected(self) -> bool:
        return self._open

    def log(self, message: str) -> None:
        if self._log:
            self._log(message)

    def open(self, resource=None) -> None:
        self.resource = resource or self.resource
        self._open = True
        self.log(f"demo SMU ready ({self.resource})")

    def close(self) -> None:
        self._open = False

    def __enter__(self):
        if not self._open:
            self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write(self, cmd: str) -> None:
        self.log(f">> {cmd}")

    def query(self, cmd: str) -> str:
        if cmd.startswith("*IDN"):
            return self.idn()
        if cmd.startswith("ERR"):
            return "0,0"
        return "0"

    def errors(self):
        return []

    def checked(self, cmd: str, fatal: bool = True) -> None:
        self.write(cmd)

    def idn(self) -> str:
        return "Agilent Technologies,B1500A,DEMO,A.00.00 (simulated)"

    def scan(self):
        from .smu import Module

        return [
            Module(1, "B1511B", "SMU", "MPSMU", [1]),
            Module(2, "B1511B", "SMU", "MPSMU", [2]),
            Module(5, "B1530A", "WGFMU", "", [501, 502]),
        ]

    def setup(self, force_ch, common_ch=None, icomp=1e-3, plc=1.0) -> None:
        self.log(f"demo setup: force={force_ch} common={common_ch}")

    def safe_zero(self, chans=()) -> None:
        pass

    def sweep_setup(self, ch, rng=0, hold=0.0, delay=0.0, sdelay=0.0) -> None:
        pass

    def sweep_end(self) -> None:
        pass

    # --------------------------------------------------------- the device

    def _conduct(self, v: float, dt: float = 1e-3) -> float:
        """Advance the state under bias ``v`` and return the current."""
        if v > self.v_set:
            self._x += (1 - self._x) * (1 - math.exp(-40.0 * (v - self.v_set) * dt / 1e-3))
        elif v < -self.v_reset:
            self._x -= self._x * (1 - math.exp(-40.0 * (-v - self.v_reset) * dt / 1e-3))
        self._x = min(1.0, max(0.0, self._x))
        g = self._x / self.r_on + (1.0 - self._x) / self.r_off
        current = g * v
        current += float(self._rng.normal()) * (0.01 * abs(current) + 1e-12)
        return current

    @staticmethod
    def _clamp(value: float, icomp: float) -> tuple:
        if abs(value) >= icomp:
            return math.copysign(icomp, value), "C"
        return value, "N"

    # ------------------------------------------------------ measurement

    def spot_i(self, ch, rng=0):
        return self._clamp(self._conduct(0.1, 0.0), 1e-3)

    def spot_v(self, ch, rng=0):
        return 0.0, "N"

    def read_at(self, ch, v_read, icomp, rng=0, settle=0.0):
        # The settle time is how long the device is held at this bias, so it
        # is exactly the dt the state should advance over. Passing 0.0 here
        # meant a device could be sat at 5 V for ten milliseconds and not
        # change at all, which made forming and threshold measurements
        # impossible to simulate -- they would run and report that nothing
        # ever switched.
        return self._clamp(self._conduct(v_read, settle), icomp)

    def pulse(self, ch, amplitude, width, icomp, hardware=True):
        value = self._conduct(amplitude, width)
        if not hardware:
            return float("nan"), "?"
        return self._clamp(value, icomp)

    def sweep(self, ch, start, stop, points, icomp, double=False,
              hold=0.0, delay=0.0, sdelay=0.0, plc=1.0, label=""):
        from .smu import SweepResult

        if points < 2:
            raise ValueError("a sweep needs at least 2 points")
        levels = [start + (stop - start) * k / (points - 1) for k in range(points)]
        if double:
            levels += levels[-2::-1]

        step = delay + sdelay + plc * 0.02 + 0.005
        t, v, i, st = [], [], [], []
        for k, level in enumerate(levels):
            value, status = self._clamp(self._conduct(level, step), icomp)
            t.append(hold + k * step)
            v.append(level)
            i.append(value)
            st.append(status)
        self.log(f"** demo sweep {start:+.3g} -> {stop:+.3g} V: {len(i)} points")
        return SweepResult(
            t=t, v=v, i=i, status=st,
            label=label or f"{start:+.3g} to {stop:+.3g} V",
        )
