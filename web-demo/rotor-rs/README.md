# rotor-rs

A branchless, dependency-free, scalar-generic Rust port of [RotorPy]'s
single-vehicle quadrotor dynamics — the ground-truth simulator for the
drone-jepa web demo (see `docs/WEB_DEMO_PLAN.md`, P1).

## What it ports

Only the single-vehicle physics + the two control abstractions the project uses,
from `rotorpy/vehicles/multirotor.py`:

- rigid-body dynamics: thrust, parasitic drag, rotor drag (H-force), translational
  lift, flapping moment, yaw moment, gyroscopic term;
- first-order motor lag (rotor speed is integrated state);
- SO(3): `quat_dot`, scipy-compatible quat→rotmat, `hat_map`;
- control abstractions `cmd_motor_thrusts` (rotor force) and `cmd_ctbr` (collective
  thrust + body rates with an inner P rate loop).

**Skipped** (not used by the project): `BatchedMultirotor`, the other 5 control
abstractions, ground contact, motor noise, wind *profiles* (wind is supported as a
live state input — only the auto-evolving profiles are dropped), sensors,
estimators, trajectories.

## Design

- **Zero runtime dependencies.** All linear algebra is hand-rolled (`linalg.rs`).
  `serde`/`serde_json` are dev-only (fixture loading).
- **Generic over `Scalar` (`scalar.rs`).** The whole kernel is written once against
  `S: Scalar`. Instantiate with `f64` today; implement `Scalar` for a SIMD/array
  lane type later and the *same* monomorphized kernel steps a batch of drones.
- **Branchless.** No data-dependent control flow in the hot path — clips/signs are
  arithmetic (`clamp`, `copysign`). The control abstraction is a generic
  (`Multirotor<S, C>`), not a runtime branch.
- **Fixed-step RK4 integrator.** RotorPy uses scipy adaptive RK45; the adaptive
  accept/reject loop is the only branchy part and blocks vectorization. A fixed
  number of RK4 substeps converges to the same ODE solution (validated to ε below).

## Validation

RotorPy ships no tests, so correctness is established by **differential golden
testing** against Python RotorPy. `scripts/export_fixtures.py` (in the repo root)
dumps fine-rate `(params, state, command, s_dot, next_state)` trajectories across
both abstractions, nominal/OOD/edge domains, and nonzero wind to
`web-demo/fixtures/sim/*.json`. `tests/golden.rs` then asserts three gates:

1. **derivative gate** — Rust `state_derivative` vs RotorPy's 20-dim `s_dot`
   (integrator-independent; pins transcription bugs) — matches to ~1e-6;
2. **per-step gate** — one step from each truth state;
3. **rollout gate** — free-running trajectory.

## Run

```bash
# (re)generate fixtures from Python RotorPy
.venv/bin/python scripts/export_fixtures.py

# run unit + golden tests
cd web-demo/rotor-rs && cargo test
```

## Example

```rust
use rotor_rs::{Multirotor, RotorForce, QuadParamsInput, State, Vec3, Quat};

let veh: Multirotor<f64, RotorForce> = Multirotor::with_substeps(&params_input, 8);
let next = veh.step(&state, &[f0, f1, f2, f3], 0.005);
```

[RotorPy]: https://github.com/spencerfolk/rotorpy
