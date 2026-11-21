# What the simulator caught

The device simulator is not a stub that returns plausible numbers. It is a
bipolar memristor, a ferroelectric and an antiferroelectric with real
switching kinetics, and every protocol in the catalogue runs against it in
the test suite.

That is why the bugs below were found on a laptop rather than on a device.
Each one is recorded with what it would have cost, because the cost is the
argument for having the simulator at all.

---

## Figures that showed one panel and hid the rest

`Panel.row` and `Panel.col` both defaulted to `0`, so any `draw()` that did
not name a cell stacked all of its panels on the first one. **Twelve
figures** were affected - SRDP, STP, MLC, retention, endurance, f-I, leak,
pulsed I-V, oscillator and the three sweep protocols.

The existing tests asserted that `draw()` returned panels, which it did. A
figure hiding two thirds of itself looked exactly like a passing figure.

An unstated cell now means "the next free cell", explicit cells are reserved
first, and a test asserts that no two panels in any of the 32 figures share
one.

## A settle time that was accepted and ignored

`DemoSmu.read_at` passed `dt = 0` to the device model, so the `settle`
argument did nothing. A simulated device could sit at 5 V for 10 ms and not
change state.

This made forming and threshold switching **impossible to simulate**: the
protocols ran, completed, and reported that nothing had ever switched. The
failure mode is the dangerous one - not a crash, but a confident negative
result.

## `*CLS` fighting the error-queue drain

`smu.py` sent `*CLS` in two places. This mainframe answers
`+100 "Undefined GPIB command."` to it.

In `setup()` that was actively counterproductive: the very next line calls
`errors()` to *drain* the error queue, and the `*CLS` above it was putting a
fresh error in. Both are gone.

## An f-I curve with a negative slope

`neuron.fi` counted spikes in a window placed **after** the drive train. A
neuron fires while it is driven; afterwards there is one monotonic
relaxation and nothing else. The protocol reported exactly one "spike" at
every amplitude, and the fitted gain came out **negative** - the opposite of
the thing the measurement exists to show.

Reads are now interleaved with the drive, and the rate is computed from the
sample times rather than from the averaging window.

## A peak-to-valley ratio made of noise

The NDR detector reported `pvr = 1.0003` on a monotonic sweep. A noisy rise
always contains one point lower than the one before it, so "a point lower
than the last" is not evidence of negative differential resistance.

It now requires a real 5 % drop, and says there is no NDR when there is
none. A figure of merit that looks like a result but is not is worse than no
number at all.

## A short-term plasticity measurement with no short term

The simulated synapse had only a seconds-scale decay constant, so the `stp`
protocol - facilitation during a train, then its decay - saw a perfectly
flat recovery. It ran, and measured nothing.

## A bias check that rejected zero volts

Verification required every read event to have a non-zero bias, on the
grounds that conductance is undefined at 0 V. But zero is the **origin of an
I-V curve**, where the current is the measurement and conductance is simply
not what is being asked for. The check made `neuron_iv` unrunnable, and the
division it was guarding is not performed anywhere.

It now requires the bias to be finite, and exempts events that measure along
a ramp: a swept bias has no single value, and inventing one would make the
conductance wrong at every point but one.

---

## Ferroelectric analysis

Four separate errors, each of which produced a number that was physically
impossible rather than merely wrong - which is how they were spotted.

| symptom | cause |
|---|---|
| 102 % energy efficiency from a passive capacitor | the antiferroelectric energy integrated `P-dV` instead of `V-dP` |
| negative hysteresis on an AFE double loop | quarter signs taken from the voltage's sign rather than the direction polarization moves |
| leakage share of 14231 % | referenced to the slow leg instead of the fast one; now also guarded against an invalid separation |
| 65 µC/cm2 read from a 20 µC/cm2 device | instantaneous current where the instrument box-averages over the sample interval - and the error moved with the sample count, which is what gave it away |

## A destructive readout nobody had accounted for

`fe_accumulation` applies sub-coercive pulses and reads the accumulated
switching with PUND. **A PUND read is destructive**: it leaves the device
fully switched to the read polarity. Reading once and continuing the train
therefore measures the first point and then a flat line of zeros.

The protocol now resets before each train and applies the cumulative count,
which makes the run quadratic in step count. That cost is inherent to a
destructive readout, not a design choice.

The C++ implementation has the same gap.
