# Can a tiny JEPA world model fly a *fleet* of different drones?

Experiment log for the writeup. Central question that emerged: **a ~9.5K-param SkyJEPA
world model drives an MPPI racer well on one drone — does it hold up when the drone
itself is randomized over a wide range (mass, size, shape)?** Short answer: **no, and the
reason is instructive.** This doc records every experiment, what it tested, and what it
showed.

## Setup

- **Demo**: browser gate-racer, everything client-side in WASM. Ground-truth quad sim
  (`rotor-rs`, a branchless Rust port of RotorPy) + a hand-rolled f32 JEPA forward pass,
  both compiled to WASM. Four drones race the same course: true-dynamics MPPI, JEPA-MPPI
  (CTBR), a PPO RL policy, and (added this round) a rotor-force JEPA-MPPI.
- **Models, all ~9.5K params**: SkyJEPA = TCN encoders [8,8,16]/[4,4,8] → GRU predictor
  (hidden 24) → 3-layer MLP prober → parameter-free DKI integrator. RL = reactive MLP
  policy (obs 21 → 85 → 85 → 4), PPO via PufferLib on a vectorized `rotor-rs` env.
- **The wide drone distribution** (the demo's "randomize" toggle, and what we train on):
  mass **0.2–2 kg** (10× range), **asymmetric** arm lengths **5–40 cm**, inertia scaled
  physically `I ∝ m·r²`, thrust coeff derived from a sampled thrust-to-weight ∈ [2,4]
  (guarantees flyability), randomized drag / motor time-constant / rate-loop gain.
- **Evaluation harnesses** (native Rust, read the same `.jblob`/`.rlb` the demo embeds):
  - `examples/jepa_fly` — one fixed hummingbird, counts gates + floor crashes.
  - `examples/rotor_fly` — the rotor-force drone on sky gates.
  - `examples/fleet_fly` — **16 randomized wide drones**, races true-MPPI / RL / a
    JEPA-under-test (`JTEST=<blob>`) on each, reports gates/crashes/respawns. Drone
    distribution and planner are env-configurable (`MASS_LO/HI`, `SYM`, `ARM_LO/HI`;
    `HORIZON`, `TRUST`, `SIGMA`, `VMAX`).
  - **Methodology lesson, established early and repeatedly: training `val_pos` does NOT
    predict flight quality. Only the fly-tests do.** Several times the lowest-val model
    flew the worst.

## The trigger: the JEPA drone dives into the floor

User observation: the JEPA drone tends to accelerate toward the ground, try to recover,
sometimes crash. Initial hypothesis — a **data-coverage gap**: the data-collection expert
MPPI never dives or recovers, so the world model is OOD for descents.

### Exp 1 — Trajectory augmentation (the "v2dive" dataset)

- **Method**: regenerate 60k trajectories with the expert's actions perturbed to cover the
  missing regime: AR(1) action noise (±25% thrust, ±2 rad/s rates), dynamic *descending*
  initial states, 8% large-tilt (34–80°) starts, occasional thrust→0 "stall bursts",
  v_max up to 13 m/s. Probe confirmed the intended coverage (16% fast descents, 20% of
  frames >60° tilt). Retrained CTBR + rotor-force JEPA.
- **Result**: **backfired.** Worse than the old model everywhere.
  | | ctbr crashes/race, gates | rotor crashes/12, gates |
  |---|---|---|
  | old | 0.5, 2.6 | 1, 2.3 |
  | v2dive | 2.0, 0.5 | 12, 0.0 |
- First (wrong) read: augmentation diluted a capacity-limited model.

### Exp 2 — Clean retrain (the "v2clean" dataset) — *the controlled comparison*

- **Method**: revert ALL augmentation (expert flight only), keep the new wide distribution.
  Isolates "augmentation" from "distribution". Retrain.
- **Result**: **v2clean is even worse** (ctbr 3.75 crashes, 0 gates; rotor 12/12).
- **Conclusion**: the augmentation was a **red herring**. v2dive ≈ v2clean (rotor
  *identical*: 12/12). The regression is the **wide drone distribution itself**, not the
  trajectories. `val_pos` actively misled: v2clean had the *lowest* val (0.34) and the
  *worst* flight.

## The fleet test: RL vs JEPA on the same 16 wide drones

| controller (~9.5K each) | gates/race | hard crashes | respawns |
|---|---|---|---|
| true-MPPI (*perfect* model) | 1.3 | **0** | 0 |
| RL-old | 2.6 | 5 | 48 |
| **RL-v2 (wide-trained)** | **4.8** | **0** | **0** |
| JEPA-old | 0.8 | 11 | 102 |
| JEPA-v2clean | 0.1 | 15 | 140 |

- **RL-v2 flies the whole fleet** (4.8/5 gates, zero crashes, 0.32–1.9 kg, symmetric or
  lopsided). The JEPA models crash constantly.
- The perfect-model MPPI **never crashes** (nothing to exploit) but is conservative on
  extreme drones (1.3 gates) — so the JEPA crashes are *model error*, not planner greed.

## Why the paper's tiny model was fine — and ours isn't

Paper (SkyJEPA) Table I domain randomization: **mass ±50%, inertia ±30%, motor τ, drag,
thrust/torque coeffs — around ONE nominal symmetric quad. No arm-length or shape
randomization.** That's ~3× mass span on a single airframe. Our demo went to 10× mass +
asymmetric frames — **3–4× past the paper's scope.** The paper's 9K model works *because*
its distribution is focused; we pushed far outside it.

## How RL "knows" the weight (it mostly doesn't)

The RL **network is weight-blind** — its observation has no mass/inertia. Mass enters at
exactly one point: action de-normalization, `thrust = (mass·g)·(1+a0)`, so "do nothing"
means "hover for *this* drone." Everything else is **feedback**: a heavy drone sags → it
sees downward velocity → adds thrust. It never needs to *predict*; it reacts. This is the
seed of the whole explanation.

## Campaign: four attempts to make JEPA-MPPI fly the wide fleet

| # | experiment | method | result |
|---|---|---|---|
| A | **Tame the MPPI** | v2clean + 5 conservative planners (short horizon, 5× trust region, low exploration, slow, combined) — no retrain | **no help** (0–0.1 gates, 11–15 crashes). Model is *wrong*, not exploited — no planner flies a wrong model. |
| B | **Train longer** | wide data, 16k/16k steps (2×) | **no help** (0 gates). Never under-trained (val had plateaued). |
| C | **Narrow the distribution** | symmetric frames, mass 0.4–0.9 kg, arms 0.12–0.22 m (~paper-width); train + test on the matching fleet | **JEPA RECOVERS: 3.2 gates/race, 0 hard-crashes** (vs 0.1/15 wide). Out-gates conservative true-MPPI (1.4), approaches RL (4.1). → **it's distribution WIDTH.** |
| D | **Feed drone params** | condition predictor+prober on a 10-dim param vector [mass, inertia, k_eta, k_m, τ, k_w, drag, arm]; +1.2K params → 10.7K | **NULL.** Held-out open-loop pos RMSE barely moves: |

Exp D detail (held-out drones, open-loop position RMSE):

| horizon | uncond | cond (params fed) | gain |
|---|---|---|---|
| 0.25 s | 0.124 m | 0.118 m | +4% |
| 0.50 s | 0.385 m | 0.364 m | +5% |
| 1.00 s | 1.046 m | 1.011 m | +3% |

- Pre-registered logic: big gain ⇒ the model wasn't extracting the drone from history;
  small gain ⇒ it had the info but can't *use* it. We got the small gain.
- **Conclusion**: the wide-fleet failure is **predictive capacity, not
  drone-identification.** Even *told exactly* which drone it is, a ~10K world model can't
  predict the wide dynamics accurately enough for MPPI (~1 m error at 1 s ≫ 0.85 m gate
  radius). "Infer from history" and "told explicitly" hit the same wall.

### Exp E — Capacity sweep

- **Method**: train plain (unconditioned) JEPA at width 1.25 / 1.5 / 2 (**14K / 20K /
  34K** params) on the full wide dataset; wide-fleet-test each. Required adding fractional
  `--width-mult` support. Directly tests whether the capacity wall is cheap to climb.
- **Result**: **completely flat.** More capacity does *nothing*.

  | model | params | gates/race | crashes (of 16) | val_pos |
  |---|---|---|---|---|
  | baseline | 9.5K | 0.1 | 15 | ~0.34 |
  | w1.25 | 14.2K | 0.0 | 16 | ~0.36 |
  | w1.5 | 19.8K | 0.0 | 16 | ~0.37 |
  | w2 | 33.8K | 0.0 | 16 | ~0.37 |

- **Conclusion**: scaling **3.5× changes nothing** — the *exact same* 0.0 gates / 16
  crashes / 144 respawns at every size, on the same drones. `val_pos` barely moves
  (~0.34→0.37), so bigger nets fit the data marginally better and fly identically. The
  wide-fleet wall is **not a capacity problem you can scale out of** (at least not 2–3.5×).
  Combined with Exp A–D this closes every cheap lever: not planning (A), not training (B),
  not data coverage (Exp 1/2), not drone-ID/params (D), not capacity (E). Only narrowing
  the distribution (C) works.

### Exp F — Open-loop error vs flight (the metric that finally explains it)

The capacity fleet metric *saturated* (every model fails all 16 drones to the respawn
cap → identical 0/16/144), so it can't rank models. Measured the finer thing — **held-out
open-loop position RMSE** — and a per-window error distribution at the 1 s horizon:

| model (held-out) | mean@1s | p90 | p99 | frac > 2 m |
|---|---|---|---|---|
| WIDE 9.5K *(crashes fleet)* | 1.04 m | 1.87 | 3.21 | 8.1% |
| WIDE 34K *(crashes fleet)* | 0.98 m | 1.76 | 3.04 | 6.6% |
| NARROW 9.5K *(flies, 3.2 gates)* | 0.91 m | 1.61 | 2.65 | 4.3% |

- Capacity **does** help prediction, but glacially: ~6% lower 1 s error for 3.5× params.
  Error is ~0.12 m @0.25 s → ~1 m @1 s = **autoregressive compounding**, which capacity
  barely touches.
- **The bombshell**: the model that FLIES (narrow, 0.91 m) and the models that CRASH
  (wide, 0.98–1.04 m) have **nearly identical open-loop error.** A 12% mean-error gap
  separates "flies the whole fleet" from "crashes every drone." **Open-loop accuracy does
  NOT determine flight.** (Even held-out RMSE joins `val_pos` on the list of metrics that
  don't predict flight — only the fly-tests do.)

**Reinterpretation — it's exploitability, not accuracy.** MPPI is an *adversary*: it
samples hundreds of plans and executes the one the model rates best, so it systematically
selects the model's **optimistic** errors (a dive the model wrongly thinks is safe). On the
wide fleet the model can't identify the specific drone, so it carries a **consistent,
exploitable per-drone bias** under aggressive actions — the planner drives into it (the
heavier >2 m tail, 8% vs 4%, is its fingerprint). Narrowing doesn't make the model more
accurate; it makes the errors **uniform/benign**, so there's nothing systematic to exploit.
This is why every model-quality lever (capacity / params / data / training) failed while
narrowing succeeded — the failure was never about average error.

### Exp G — Policy-guided MPPI (the world model IS usable when guided)

Direct test of the rank-fidelity diagnosis: instead of sampling MPPI plans around hover,
re-center the search each step on the RL policy's action (`set_nominal_const`), so the
JEPA model only RANKS plans near a known-good, in-distribution action. Wide model
(skyjepa_ctbr_v2clean), 16-drone wide fleet:

| controller | gates/race | crashes | respawns |
|---|---|---|---|
| true-MPPI (perfect model) | 1.3 | 0 | 0 |
| RL-v2 (reactive) | 4.8 | 0 | 0 |
| JEPA alone (hover-centered MPPI) | 0.1 | 15 | 140 |
| **JEPA + RL-proposal** | **3.4** | 1 | 10 |

- **0.1 → 3.4 gates, 15 → 1 crashes** from a one-line change (re-center the search). The
  wide 9K world model is **usable and adapts across the fleet** — it was never broken as a
  *model*, only as an *open-loop planner*. Confirms Exp F: the model ranks fine among
  good plans, terribly among arbitrary ones.
- **But the hybrid (3.4) does NOT beat pure RL (4.8).** On a task the reactive policy
  already nails, the model's lookahead adds no value over its own guide (and its still-
  imperfect ranking + the speed-capped JEPA cfg cost a little). The world model would pay
  off where the proposal is weaker / lookahead matters (longer horizon, obstacles,
  constraints) — not here.

### Exp H — Spline / contouring cost + seeded search (the world model as a REFINER)

Idea (user): stop asking the model to PLAN toward a gate (where its rank-fidelity
collapses) and instead give it a model-free reference SPLINE that arrives perpendicular
to the gate (a fan of 4, varying approach tension), cost rollouts by how well they track
it (contouring + along-spline velocity + progress) — MPCC-style. Then seed MPPI's search
in-distribution so the model only RANKS locally. Wide fleet, skyjepa_ctbr_v2clean:

| controller | gates/race | crashes |
|---|---|---|
| JEPA alone (gate cost) | 0.1 | 15 |
| JEPA + spline cost, hover-centered | 0.0 | 15 |
| SE3 geometric seed only (no model) | 0.8 | 14 |
| JEPA + spline + SE3 seed | 1.5 | 9 |
| JEPA + spline + RL seed | 3.4 | 1 |

Findings:
- **A better cost alone does nothing** (0.0 vs 0.1): the spline cost is evaluated on the
  model's predicted rollouts, and for arbitrary hover-centered plans the model mispredicts
  which ones track the path. Cost shape was never the bottleneck — the candidate
  *distribution* (where the model predicts reliably) is.
- **The model IS a useful refiner**: JEPA+spline+SE3 (1.5 gates, 9 crashes) beats the raw
  SE3 seed (0.8, 14) — the lookahead improves the proposal. Not dead weight.
- **Performance is seed-dominated**: SE3's fixed gains are mismatched to a 0.2-2kg fleet
  (14/16 crashes alone), so the model can only refine a bad seed so far. The adaptive RL
  seed lands higher (3.4) for the same reason.

**Synthesis**: the wide-fleet world model is a *refiner*, never an *open-loop planner*.
Its value is bounded by the proposal it refines; getting a good proposal on a wide
distribution is the hard part (fixed-gain geometric = mediocre, learned policy = good).
Backprop-through-JEPA for candidates is the WORST option (gradient_mpc.py / NOTES: the
most aggressive optimizer exploits the model hardest — diverges ~26m vs ~0.22m Gaussian);
the right "optimize for a seed" is through RELIABLE nominal physics (SE3/flatness), then
let the model rank locally.

### Exp I — Recurrent (LSTM) RL vs reactive

Does memory let the policy do in-context drone system-ID (the thing the JEPA history
encoder failed at)? Capacity-matched LSTM policy (h=32, single-layer encoder + LSTM =
9,284 actor params ≈ JEPA's 9,534), trained on the wide env, eval on the wide fleet
(deterministic, `eval_fleet`):

| policy | mean return | median | ep length |
|---|---|---|---|
| reactive (RL-v2) | 84.65 | 87.07 | 429.5 |
| recurrent (LSTM) | 86.78 | 92.71 | 429.3 |

Memory helps only **marginally** (+2.5% mean, +6.5% median; same survival). The LSTM does
a little in-context adaptation (more consistent across the fleet) but reactive feedback
already handles the fleet, so memory isn't a big lever. (Note: PufferLib's LSTM training
path deadlocked at ~88M/120M steps — result is from the 88M checkpoint; trend is clear.)

## The thesis (revised)

The wide-fleet failure is a **model-exploitation / robust-control** problem, not a
prediction-quality one:

- A world model + MPPI fails because an imperfect *shared* model has per-drone biases, and
  the planner **adversarially amplifies** them. More capacity lowers average error ~6%/3.5×
  but doesn't remove the exploitable structure.
- A reactive RL policy succeeds at the *same* param count because **there is no planner to
  exploit it** — its errors are corrected by feedback on the next step, not amplified by a
  search over plans.
- Fixes that work target *exploitability*, not accuracy: shrink the distribution so errors
  are benign (Exp C, narrow → 3.2 gates), or drop the forward model for reactive control
  (RL → 4.8 gates). Predicting the fleet was never the bottleneck.

## The thesis (original, superseded by the above)

For a **wide** drone fleet at a fixed tiny parameter budget, a **reactive policy beats a
world-model + MPPI**, because **predicting is a harder function than reacting**:

- RL must learn "given my current error, what corrective action?" — low-complexity, one
  small net covers all drones, errors self-correct via feedback.
- JEPA-MPPI must learn "for *any* action sequence on *any* drone, the exact future
  trajectory" — a much richer function, which MPPI then actively probes for its weakest
  point. Same params, very different demands.

JEPA-MPPI is not broken — it reproduces the paper's behavior **at the paper's distribution
width** (Exp C). It just doesn't scale to a fleet the paper never asked it to model, and
neither more data (Exp 1/2), more training (B), gentler planning (A), nor explicit params
(D) fix that. The open question Exp E answers: how much *capacity* would.

## Reproducibility / artifacts

- Datasets: `racing_v2*.bin` (wide, clean), `racing_narrow.bin` (paper-width),
  `racing_pc.bin(+.params)` (wide + per-drone param sidecar). Gen distribution is
  env-configurable (`MASS_LO/HI`, `SYM`, `ARM_LO/HI`) in `examples/gen_dataset`.
- Models: `skyjepa_ctbr_v2clean` (wide), `exp_narrow` (paper-width), `exp_w{125,15,2}`
  (capacity sweep), param-conditioning via `--n-params`/`drone_jepa/train_pc.py`.
- Code added: `src/rotor_mppi.rs` + 4th drone, `examples/fleet_fly.rs`, env knobs,
  fractional width, param-conditioning (`SkyJEPA(n_params=…)`).
- Related memory notes: `wide-distribution-hurts-jepa`, `param-conditioning-null-result`,
  `bigger-jepa-races-worse`, `mppi-racer-stability-levers`, `rl-drone-pufferlib`.
