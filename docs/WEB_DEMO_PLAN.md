# drone-jepa — interactive real-time web demo: plan

Goal: a **real-time, in-browser demo that showcases the power of the SkyJEPA
world model** — a drone flying while the model predicts its future, adapts to
changing dynamics, beats the autoregressive baseline, and can be *used* for
control (MPC) — all client-side, zero server.

This doc is the project plan. **Near-term focus: the Rust port of the RotorPy
dynamics (`rotor-rs`) for WASM embedding** (Section 6). Everything else is
scoped here so the port is built against the right target.

---

## 1. What we're showcasing (the story)

Don't show "a drone flying" — show what only a *world model* can do. Hero loop,
ranked by impact:

1. **Forecast ghost** ⭐ — the live drone flies; JEPA paints its predicted next
   ~1 s as a translucent ghost trail glued to the real drone. "It sees the
   future." The ghost staying locked to reality is the whole pitch.
2. **Break it / adaptivity** — a slider doubles mass / adds wind / drops a
   payload mid-flight; the ghost re-locks within a few steps with **no
   retuning** (the model infers dynamics from history). The killer feature.
3. **Compounding showdown** — toggle a second (red) ghost: the autoregressive
   baseline, which visibly **diverges** while JEPA stays locked. The scientific
   claim, animated.
4. **Fly-to-click (MPC)** — user clicks a 3D target; the drone flies there via
   JEPA-in-MPPI (or **gradient-MPC**), with candidate rollouts drawn as a faint
   "spray of imagined futures."

Ship 1+2+3 as the hero; 4 as the interactive finale.

Framing badges: **"~30K params · real-time · runs in your browser"** — small &
fast *is* the power (embedded / sim-to-real).

Honesty rules: lead with prediction + adaptivity + baseline contrast (rock
solid). Present MPC tracking (~0.24 m CTBR) as "and it controls too," not the
headline. Don't oversell tracking precision.

---

## 2. Architecture decision — Path B (fully client-side)

Two paths were considered:
- **Path A** — Python (FastAPI + WebSocket) backend running RotorPy + torch,
  streaming to a Three.js frontend. Fastest to a demo, reuses everything, but
  needs a server per user.
- **Path B (CHOSEN)** — **everything in the browser**: dynamics + model both run
  client-side (WASM), deploy as a **static site** (Vercel/Pages), infinite scale,
  zero server, zero latency.

**Decisions:**
- **No ONNX.** The model is ~6 trivial ops + a custom DKI integrator (per-step
  loop with SO(3) matrix-exp) — exactly what ONNX exports badly. We implement the
  model directly.
- **Model in Rust via [Candle](https://github.com/huggingface/candle).** Tensor
  ops + safetensors loading + WASM + **autodiff in the browser** (→ gradient-MPC
  runs client-side, a showpiece ONNX can't do). ~150 lines forward pass.
- **Sim in Rust (`rotor-rs`)**, compiled to WASM. Hand-ported rigid-body
  dynamics matching RotorPy (NOT the model's DKI — must be a distinct, richer
  "ground truth" with the true randomized params so prediction-vs-reality is
  non-trivial). nalgebra → WASM.
- **Frontend: Three.js / react-three-fiber.** 60 fps render; sim/model tick at
  fixed rate, interpolate.

**Big synergy:** the **SO(3)/rigid-body math is shared** between `rotor-rs`
(ground-truth sim) and the model's DKI decoder (~80% overlap). Write it once.

A small sim-to-sim gap (model trained on Python RotorPy, demo runs the Rust port)
is **on-brand**, not a bug — it's the sim-to-real robustness the JEPA is built
for (domain randomization makes it transfer).

---

## 3. System components

```
web-demo/
  rotor-rs/        # Rust: ground-truth quad dynamics (port of RotorPy) -> WASM
    src/so3.rs       #   shared SO(3)/quaternion/rigid-body math  <-- also used by model DKI
    src/multirotor.rs#   dynamics: thrust + gravity + aero drag + 1st-order motor lag
    src/control.rs   #   cmd_motor_thrusts, cmd_ctbr inner loop (the abstractions we use)
    src/lib.rs       #   wasm-bindgen API: new(params), step(state, action, dt) -> state
  jepa-rs/           # Rust+Candle: the world model forward pass -> WASM
    src/model.rs     #   TCN encoders, GRU predictor, MLP prober (Candle)
    src/dki.rs       #   DKI integrator (reuses so3.rs); differentiable
    src/mpc.rs       #   MPPI (gaussian) + gradient-MPC (Candle autodiff)
    weights/*.safetensors
  web/               # Three.js / react-three-fiber frontend
    drone, ghost(s), rollout spray, knobs (mass/wind/payload), fly-to-click
  fixtures/          # golden reference data (Section 4) — the validation backbone
```

---

## 4. Validation foundation (build FIRST — it unblocks everything)

Since RotorPy has **no test suite**, validate every port by **golden /
differential testing** against the Python originals.

`scripts/export_fixtures.py` (pure Python, reuses `drone_jepa/`):
- **Sim goldens:** sample `(params, initial_state, action_sequence)` across
  domains, **each control abstraction** (`cmd_motor_thrusts`, `cmd_ctbr`), and
  edge cases (high tilt, motor saturation, slow/fast `tau_m`). Run RotorPy → dump
  trajectories to JSON/npz.
- **Model goldens:** dump `(state_hist, action_window) → predicted trajectory`
  (and intermediate latents) from the PyTorch CTBR model.
- **Weights:** `state_dict` → **safetensors** (`safetensors.torch.save_file`),
  stable tensor names.

Then `rotor-rs` and `jepa-rs` assert match within ε (~1e-4; looser if the
integrator differs). This catches the classic transcription bugs (Section 6/8).

---

## 5. Phases / milestones

- **P0 — Fixtures** (Python): sim goldens + model goldens + safetensors export.
- **P1 — `rotor-rs` core** (FOCUS): SO(3) + rigid-body dynamics + the two
  control abstractions, validated against sim goldens, native first.
- **P2 — `rotor-rs` → WASM**: wasm-bindgen API, runs in a JS harness.
- **P3 — `jepa-rs` model** (Candle): forward pass + DKI, validated against model
  goldens, native then WASM.
- **P4 — frontend MVP**: drone + forecast ghost + mass slider + baseline toggle.
- **P5 — MPC**: MPPI + gradient-MPC client-side; fly-to-click; rollout spray.
- **P6 — polish + deploy**: static hosting, mobile, the "power" badges.

---

## 6. FOCUS: `rotor-rs` (the Rust RotorPy port)

### Scope (what to port, what to skip)
Port **only the single-vehicle dynamics + the control abstractions we use**.
Skip: the batched/torch `BatchedMultirotor`, plotting, sensors, estimators, wind
fields (start with zero/constant wind), the other 5 control abstractions, the
trajectory zoo (the demo generates its own references).

Source: `rotorpy/vehicles/multirotor.py` (single-vehicle parts; ~150–300 LOC of
real math out of 1180) + `rotorpy/controllers/quadrotor_control.py` (SE3, 307
LOC, only if we want closed-loop SE3 in-browser — optional for v1).

### Dynamics to implement (`multirotor.rs`)
State: `pos(3), vel(3), quat(4), omega(3), rotor_speeds(4)` (motor lag is state!).
Per fine step `dt_fine`:
- motor first-order lag: `rotor_speeds += (cmd_speeds - rotor_speeds)·dt/tau_m`
- forces: thrust `= k_eta·Σ rotor_speeds²` along body-z; **aero drag** (`c_Dx/y/z`);
  optional rotor drag/inflow (`k_d,k_z,k_h,k_flap`) — start minimal, add to match
  goldens.
- rotational: `omega_dot = J⁻¹(torques − omega × J·omega)`; torques from the
  allocation matrix (`f_to_TM`).
- integrate: scipy uses **adaptive RK45** (`solve_ivp`). Rust: use `ode_solvers`
  Dopri5 to match, OR fixed-step RK4 (simpler, small ε). Attitude via quaternion
  / SO(3) exp (shared `so3.rs`).

### Control abstractions (`control.rs`) — only the two we use
- `cmd_motor_thrusts`: force → `rotor_speeds = sign·sqrt(|f/k_eta|)`.
- `cmd_ctbr`: `[thrust, body_rates]` → inner rate loop `wdot_cmd = -k_w·(w − w_cmd)`
  → allocation → rotor_speeds.

### Dependencies
`nalgebra` (vectors, `Rotation3`/`UnitQuaternion`, 3×3), an ODE integrator
(`ode_solvers`) or hand RK4, `wasm-bindgen` (+`serde`/`serde-wasm-bindgen` for the
JS boundary). All WASM-friendly.

### Gotchas (the golden fixtures catch these)
1. **Adaptive RK45 vs Rust integrator** — #1 mismatch source. Match method+tol or
   accept ε.
2. **Quaternion convention**: scipy `[x,y,z,w]` vs nalgebra `[w,x,y,z]`.
3. **Motor lag is integrated state**, not an instantaneous map.
4. **Allocation matrix** (`f_to_TM`) sign/axis conventions; `cmd_ctbr` gain `k_w`.
5. Rotor coefficient terms (`k_d,k_z,k_h`) — add incrementally until goldens match.

### Validation
Native Rust test: load `fixtures/sim/*.json`, run `multirotor.step` over each
action sequence, assert `‖rust − python‖ < ε` per state component. Cover both
abstractions + edge cases. Only WASM-ize after native goldens pass.

### Why this pays off 3×
- **Browser demo** (nalgebra → WASM).
- **Faster data gen** (Python RotorPy ran ~4 traj/s; the CTBR set took ~35 min).
- **RL throughput** — PPO was stuck at ~1,500 SPS because RotorPy-in-Python was
  the bottleneck (the PufferLib lesson). A vectorized Rust sim is the fix.

---

## 7. `jepa-rs` model (Candle) — summary (P3)

Forward pass (~150 LOC): TCN causal-Conv1d + GELU + LayerNorm + residual;
GRUCell (≈20 LOC from gate equations); 3-layer MLP prober; **DKI reusing
`so3.rs`**. Load `weights/*.safetensors`. GELU must match PyTorch's **erf**
variant. MPPI + **gradient-MPC via Candle autodiff** (client-side backprop to the
action — the showpiece).

---

## 8. Open decisions
- v1 control: MPPI (gaussian, robust, 0.24 m) is the safe default; gradient-MPC
  is the flashy autodiff demo. Ship both, default gaussian.
- Reference: fly-to-click target vs scripted circle vs free user-drive.
- Hosting: static (Vercel/GH Pages). Weights as a fetched safetensors asset.
- Native-app reuse: `rotor-rs` + `jepa-rs` also run native (faster data/RL) —
  keep the crates `no_std`-friendly where cheap.

## Status
- [x] Plan written (this doc).
- [x] P0 fixtures — `scripts/export_fixtures.py` dumps fine-rate goldens (params,
      state, command, `s_dot`, next_state) for both abstractions + nominal/OOD/edge
      domains + nonzero wind → `web-demo/fixtures/sim/*.json`.
- [x] **P1 `rotor-rs`** — single-vehicle dynamics ported, native. Zero runtime
      deps, generic over a `Scalar` trait (batch later for free), branchless,
      fixed-step RK4. Validated against Python RotorPy by `tests/golden.rs`:
      derivative gate ~1e-6, per-step + free-running rollout gates pass for all 9
      fixtures (`cargo test`). Found+fixed the `M_force` sign (RotorPy's einsum
      transpose cancels its leading minus). SE3 controller NOT ported (optional;
      the demo replays/closes the loop itself). See `web-demo/rotor-rs/README.md`.
- [x] **P2 `rotor-rs` → WASM** — `--features wasm` builds a `WasmDrone` (wasm-bindgen):
      `new(params, n_sub)` (partial JS params → hummingbird defaults), `step_rotor_force` /
      `step_ctbr`, `set_wind` / `set_mass` (the "break it" knobs), `state()` (17) /
      `state_jepa()` (18-dim canonical for the model). `wasm-pack build --target web`
      → `web-demo/web/pkg/` (82 KB); minimal harness in `web-demo/web/` (`index.html`+`main.js`).
      Native stays zero-dep (wasm deps gated behind the feature).
- [x] **Inlining / branchless verified.** `#[inline(always)]` on the hot path; a single
      step disassembles to **straight-line, zero-jump** code (objdump of `examples/asm.rs`
      `step_scalar_once`: no `b./cbz/tbz`, one basic block).
- [x] **SIMD batch via `Scalar`.** `F64x<L>` lane type implements `Scalar` (branchless,
      autovectorizes — `F64x<4>` add → `fadd.2d`); `integrate::<F64x<L>>` steps L drones at
      once, validated == per-lane scalar to <1e-12 (`simd::tests`). Batch helpers
      `pack_params`/`pack_state` in `src/simd.rs`.
- PERF (Apple Silicon / NEON 2×f64, `target-cpu=native`, after inlining):
  scalar step **0.76 M/s** @N_SUB=8 (1.3 µs), **5.5 M/s** @N_SUB=1. SIMD batch only
  **~1.1×** here — because the scalar step is *already* SLP-autovectorized to 2-wide `.2d`
  (551 vector ops in one step), so NEON's 2-wide f64 units are near-saturated; explicit
  batching just reshuffles. The batch payoff is expected to be larger on wide-SIMD x86
  (AVX-512, 8×f64). Python RotorPy reference: ~1k steps/s → Rust scalar is still ~600–4500×.
- [x] **Autonomous gate racing (new hero task).** `gates.rs` (Gate = center+normal+radius,
      Course with plane-cross+within-radius pass detection), `mppi.rs` (CTBR-action MPPI),
      `rng.rs` (zero-dep xorshift+Box–Muller). Pluggable by design:
      `RolloutModel` (`TrueDynamics` now → swap in the JEPA model for #1) and `Controller`
      (`MppiController` now → an `RlPolicy` later for #2). `examples/race.rs` flies a
      4-gate slalom autonomously: **all gates in 2.70 s, ~11 m/s** through gates.
  - Plan in **CTBR** (not rotor forces): the inner rate loop tames attitude (the measured
    ~3× tracking win); race uses an aggressive `k_w=16` rate gain.
  - MPPI lessons baked in: AR(1) (temporally-correlated) noise so rollouts make *sustained*
    tilt; velocity-toward-gate + altitude-hold cost; NaN-cost sanitization. KEY gotcha: the
    rollout model's fixed-step `n_sub` must be fine enough (8) for the stiff `k_w`+`tau_m`
    dynamics, else every rollout diverges to NaN and MPPI goes rudderless.
- [x] **P3 `jepa-rs` (Candle inference).** New crate `web-demo/jepa-rs` (candle-core, depends on
      rotor-rs). Neural net (TCN encoders, GRU predictor, MLP prober) in Candle f32 loaded from
      safetensors; parameter-free DKI (semi-implicit Euler + SO(3) Rodrigues) in Rust f64.
      `scripts/export_jepa.py` dumps weights→safetensors + config + predict() goldens.
      **Validated: Candle == PyTorch to 1.1e-6** over the full 20-step rollout (`tests/golden.rs`).
      Decision: Candle for inference, **PyTorch stays for training** for now.
- [x] **JEPA wired into the racer (#1).** `RolloutModel` gained a default `rollout_batch` + an
      `observe` hook (history for the encoders); `jepa-rs::rollout::JepaRollout` implements it
      (quat State→18-dim, rolling H-history, batched `predict_batch`). `examples/race_jepa.rs` puts
      the learned model in the MPPI loop end-to-end. It RUNS but **crashes / 0 gates** — the bundled
      checkpoint is **out-of-distribution for aggressive racing** (trained on gentle SE3 data, k_w=1;
      racer is k_w=16 @ ~11 m/s). Not a wiring/port bug (port matches to 1e-6) — it's the model.
- [x] **JEPA-MPPI flies the course (slowly).** At k_w=1 (matching the training plant) with a
      temporary speed cap (v_max=2 m/s) to fence MPPI inside the model's valid regime, the learned
      model races **all 4 gates in 6.05 s** (`examples/race_jepa.rs`). Open-loop the model predicts
      the true dynamics to ~0.04–0.10 m over 1 s in-distribution (`examples/predict_check.rs`).
      Without the cap it crashes (~2 s) — **model exploitation**: MPPI seeks the model's
      overconfident OOD blind spots. The cap is a band-aid; the real fix is data.
- [ ] **NEXT — better training data (paper's recipe) → retrain → drop the cap.** The paper trains on
      **closed-loop NMPC/MPPI** trajectories ("complementary action distributions"), NOT SE3/random —
      the current checkpoint's whole OOD problem is that it only saw gentle SE3 data. Plan:
      generate closed-loop MPPI data with `rotor-rs` (fast), with DR over **mass/inertia/motors/
      frame AND `k_w`** (the encoder infers dynamics from history → adaptivity → kills OOD AND powers
      the "break it" demo), a **slow→fast** speed sweep, and **wind**; more samples. Retrain in
      PyTorch, re-export weights (drop-in), remove `v_max`/`w_speed`.
- [ ] Later: RL policy `Controller` (#2, PPO on the fast batched sim); browser race demo (Three.js);
      Candle for training and/or gradient-MPC if a fully-Rust stack is wanted.
