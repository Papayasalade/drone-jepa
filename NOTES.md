# Implementation notes — spec choices & deviations

This file records every place the implementation had to *decide* something the
paper leaves unspecified, plus deviations forced by the sim — the "what we
actually built and why."

## ⚠️ Datasets ↔ models: use the matching trajectories

There are TWO action spaces, and a JEPA must be trained AND deployed on the
trajectories that match its action mode — they are NOT interchangeable (the
4-dim action means different things, and the DKI physics differs):

| Model | `action_mode` | Train/eval data | Deployment (MPPI) |
|---|---|---|---|
| **rotor-force / "MPPI" JEPA** (`skyjepa.pt` 2×, `skyjepa_1x.pt`) | `rotor_force` | `dataset.pt` (collected with `--action-mode rotor_force`; action = 4 rotor forces) | RotorPy `cmd_motor_thrusts`; MPPI samples rotor forces |
| **CTBR JEPA** (`skyjepa_ctbr.pt` 2×, `skyjepa_ctbr_1x.pt`) | `ctbr` | `dataset_ctbr.pt` (collected with `--action-mode ctbr`; action = [thrust, wx, wy, wz]) | RotorPy `cmd_ctbr` inner rate loop; MPPI samples [thrust, body-rates] |

Rules of thumb:
- Train the **CTBR JEPA on `dataset_ctbr.pt`**, the **rotor-force JEPA on `dataset.pt`**. Crossing them is meaningless (the prober/DKI for `ctbr` predicts `omega_dot`; for `rotor_force` it predicts the rotor→torque map `K`).
- The eval/MPPI helpers read `action_mode` from the checkpoint config (`SkyJEPA.from_checkpoint`) and pick the right vehicle abstraction + action bounds automatically — so always load via `from_checkpoint`, never hand-build the model.
- The OOD set (`dataset_ood.pt`) is rotor-force; collect a CTBR OOD set with `--action-mode ctbr` if you want an OOD CTBR check.

## Model zoo & control stack (what this grew into)

Beyond the Tier-1 reproduction, the repo now holds a small zoo of models and
controllers that can be raced head-to-head in the evaluator (`app.py`).

**World models (have `predict`; appear in all tabs):**
| model | action space | params | notes |
|---|---|---|---|
| rotor-force JEPA 1× / 2× | 4 rotor forces | 9.9K / 34.5K | `dataset.pt`; cmd_motor_thrusts |
| CTBR JEPA 1× / 2× | [thrust, body-rates] | 9.5K / 33.8K | `dataset_ctbr.pt`; cmd_ctbr inner loop |
| AR baseline (naive 1-step / strong multi-step) | rotor forces | 6.4K | the "thing JEPA beats" |
| probabilistic CTBR JEPA | CTBR | ~36K | variance head + NLL (aleatoric σ) |
| ensemble of K CTBR JEPAs | CTBR | K× | disagreement = epistemic uncertainty |

**Controllers (race-only; no world model):**
| controller | how it flies |
|---|---|
| MPC + JEPA (MPPI) | samples action plans, rolls them through a world model, picks best |
| RL policy (PPO) | one network forward pass per step (`drone_jepa/rl/`) |
| PID (classical) | cascaded position→thrust+body-rate, nominal mass + integral (`control/pid.py`) |

**MPPI deployment improvements (all in `control/mppi.py`):**
- **Noise smoothing** (temporal EMA of candidate actions): ~halves tracking error
  vs jerky white noise (matches the smooth training distribution).
- **Integral action** (`ki`): drives steady-state offset (e.g. altitude) → 0, like
  a real flight controller. Best ~0.1; cut altitude offset ~half + improved RMSE.
- **Per-channel action bounds / σ**: thrust vs body-rate live on different scales.
- **Samplers:** `gaussian` (robust, default) vs `icem` (colored noise + spline
  basis + elite refit). **Finding:** on this imperfect learned model the aggressive
  iCEM exploits model bias and DIVERGES (~26 m) while Gaussian stays at ~0.22 m —
  a textbook "aggressive planner + imperfect model = exploitation". The bottleneck
  is model accuracy, not sampling.
- **Uncertainty-aware planning** (`unc_lambda`): penalize plans by ensemble
  disagreement or predicted variance. (Results: see "Uncertainty-aware planning".)

### Closed-loop tracking leaderboard (circle, CTBR domains, ki=0.1)
| controller | RMSE |
|---|---|
| CTBR JEPA 2× (MPC, inner loop) | **0.24 m** |
| CTBR JEPA 1× | 0.53 m |
| RL policy (PPO) | ~0.4 m (brittle: crashes 1/4 domains) |
| rotor-force 2× MPPI | 0.70 m |
| PID (classical) | 0.95 m |

The big lesson: **the realistic cascaded control (CTBR + inner rate loop) beats raw
rotor-force MPPI ~3×** — fighting fast attitude dynamics with raw motor forces was
the problem; the inner loop (like PX4) fixes it.

## Uncertainty-aware planning (does it fix the iCEM divergence?)

The aggressive iCEM sampler exploits the imperfect model and diverges (~26 m).
We tested the two standard cures. **Neither robustly fixes it here — and *why* is
the interesting part.**

**(a) Ensemble (epistemic) — `model/ensemble.py`, `unc_lambda` penalty.**
Penalize plans the members *disagree* on. Two variants:
- *Cheap* (4 probers sharing stage-1): **no help at any λ.** The members share the
  encoders/predictor AND the deterministic DKI, so they agree even on bad actions.
- *Fully independent* (4 separate stage-1+2, seeds 0/11/12/13): at a *large* λ=40 it
  **rescues 2 of 3 domains** (iCEM RMSE 0.5–0.8 m, competitive!) but a 3rd still
  diverges (35 m). So independence helps and the signal is real, but it's **weak and
  inconsistent**: the exploited dimension is thrust→altitude through the *hard-coded
  DKI physics that's identical across members*, so they only disagree on the small
  learned residual, not on the dynamics the planner actually exploits.

| iCEM + independent-ensemble penalty | λ=0 | λ=10 | λ=40 |
|---|---|---|---|
| mean tracking RMSE [m] | 16.1 | 23.6 | 12.3 (2/3 ≈0.6 m) |

**(b) Probabilistic prober (aleatoric) — `--probabilistic`, variance head + NLL.**
- **Calibration is excellent**: `corr(predicted σ, actual error) = 0.87` — the
  NLL-trained variance genuinely tracks where the model is wrong (predicted σ 0.03→
  actual 0.05 m; σ 0.63→ 1.64 m). A clean success, useful for monitoring/safety.
- **As a control penalty it barely helps** (best mean 9.9 m, still diverges) —
  exactly as theory predicts: a single model's variance is *aleatoric* (data noise),
  not the *epistemic* (OOD) signal the exploitation needs.

**(B-trust-region) SIGReg latent trust region — `trust_lambda` in `mppi.py`.**
SIGReg makes latents ~N(0,1) (in-dist per-dim energy is a *very* tight 0.83, max
0.84), so penalize plans whose predicted latents leave the manifold. **Also no
help** (~20 m at every λ) — and the diagnostic is decisive: during an exploiting
flight the drone climbs (z 1.8→2.3 m) while the **latent energy stays flat at
0.82**. The predicted latents *never leave the manifold*; the exploitation lives
entirely in the **prober+DKI decoder**, not the latent (consistent with the
original teacher-forcing diagnostic). So a latent trust region targets the wrong
place. An **action-space** trust region (penalize OOD actions) would be the cheap
correct version; **Fix A** (learn the thrust-scale in the prober so it varies
across members) is the principled one.

**Two targeted fixes (the "both" the deep-dive ended on):**
- **Action-space trust region** (`act_trust_lambda`, penalize candidate actions
  beyond the training action distribution): **the best result** — first thing to
  reliably *stop the runaway* (26 m → ~5.5 m, no seed diverges). Bounds the symptom
  but is still loose (can't both constrain actions and track tightly).
- **Fix A — learned thrust-scale** (`--learn-thrust`: the DKI thrust→accel inverse-
  mass becomes a per-member learned output, so the *exploited dimension* varies
  across an ensemble). **No improvement** over the plain independent ensemble
  (12.6 m @λ=40 vs the plain ensemble's 2/3≈0.6 m) — and that nails the root cause.

**Gradient-MPC (`control/gradient_mpc.py`) — backprop through the differentiable
model+DKI to the actions** (the opposite extreme from MPPI's sampling). Surprise:
it does **not** diverge — ~1.0 m tracking (CTBR, lr=0.3, 12 iters), much better than
iCEM, worse than Gaussian. Why: **few gradient steps from the hover warm-start =
implicit trust region** (early stopping keeps actions near the nominal). The
exploitation thesis still holds, just *gracefully*: **more optimization (lower lr /
more iters) makes it WORSE** (1.0 → 1.7 → 2.4 m), not better. So gradient descent +
early stopping is a self-limiting aggressive optimizer — a nice counterpoint to the
"gradient would diverge hardest" intuition. (Selectable as a race contestant.)

**Final unifying conclusion (the deepest finding of the project).** The iCEM
divergence is a **systematic, *in-distribution* model bias amplified by greedy
optimization** — NOT exploitation of OOD regions. The proof: the predicted latents
stay on-manifold (energy 0.82) and the exploit actions stay within ~1.2σ of the
training distribution while the drone diverges. That's why *every* uncertainty/
disagreement signal fails — **all ensemble members share the same bias** (same data,
same physics structure), so they agree on the biased prediction and produce no
disagreement to penalize; learning the thrust-scale (Fix A) doesn't help because the
members still converge to the same biased function. Only two things work: a
**conservative optimizer** (Gaussian averaging — doesn't amplify the bias → 0.22 m,
the winner) or **fixing the bias** (a more accurate/calibrated model). An action
trust region bounds the symptom but doesn't cure it. Corollary: **gradient-based
MPC would diverge even harder** (it's the most aggressive optimizer of all).

**Older conclusion (still true):** Uncertainty-aware planning is the theoretically-correct direction,
and we confirmed both halves of the textbook story (epistemic-ensemble is the right
*kind* of signal; aleatoric-variance calibrates but doesn't fix exploitation). But
on *this* physics-structured model neither is a robust cure, because the exploited
dimension lives in the shared deterministic DKI. The **robust Gaussian sampler
(0.22 m) remains the practical winner**; the principled fix would be a **trust region
in action/data space** (constrain candidates to the training distribution) rather
than an uncertainty penalty. Solid, honest negative-with-nuance result.

## Model (drone_jepa/model/)

| Item | Paper | Our choice | Rationale |
|---|---|---|---|
| State latent dim | "≈16" (inferred) | **16** | TCN final channel [8,8,**16**]. |
| Action latent dim | "≈8" (inferred) | **8** | TCN final channel [4,4,**8**]. |
| Predictor I/O | "GRU hidden 24"; note 16+8=24 | GRUCell **input = concat(s,z)=24**, hidden 24, linear readout 24→16 | Most literal reading of `Pred(s_t,z_t)` + the 16+8=24 identity. |
| TCN kernel / dilation | not given | kernel **3**, dilations **1,2,4** | Standard causal-TCN recipe; receptive field covers H=10. |
| Prober MLP width | depth 3 only | hidden **40** | Lands total params at **9023 ≈ 9K** (the paper's stated count). |
| DKI rotational model | "K = residual angular accel" | `omega_dot = K @ a` (fully learned allocation; no nominal inertia term) | True inertia is domain-randomized/unknown, so the residual subsumes allocation⁻¹·inertia⁻¹. |
| DKI translational | nominal thrust+gravity + residual | `v_dot = -g e3 + (T/m_nom) R e3 + dvdot` | `m_nom`=0.5 (hummingbird); `dvdot` absorbs true-mass/drag mismatch. |
| Attitude integration | SO(3) matrix exp | `R_{t+1}=R exp(hat(omega)dt)` | Eq. 15-16. Reprojection OFF: exp-map stays on SO(3) to ~1e-15 over T=20, and SVD of an exact rotation (singular values 1,1,1) fails to converge. |
| SIGReg | 17 spline knots, Epps–Pulley | empirical CF matched at **17 frequency knots** vs N(0,1) CF, 256 random projections | Differentiable, collapse-proof equivalent of the spline goodness-of-fit. |
| Collapse prevention | SIGReg, **not** EMA | no stop-grad on the target latent by default | Matches the explicit "not EMA/momentum" note. |
| Stage-2 loss scaling | "L2 on physical state" | MSE in **normalized** state space (÷ per-dim std) | Pos/vel/rot/omega live on different scales; normalizing avoids hand-tuned weights. |

## Data quality — the accuracy lever (keep the paper's ~9K model!)

The paper hits good accuracy with ~9K params, so when our prediction looked
"foreign" to ground truth (~1 m at 1 s), the fix was the **data**, not a bigger
model. Two coupled problems, both from running RotorPy's SE3 controller naively:

1. **Chattering actions.** SE3 *feedback at 20 Hz* bang-bangs the rotor forces:
   step-to-step change ≈ **2.5 N on a 3.8 N signal** (67%!). With motor lag the
   command≠applied, so the action→next-state map is near-unlearnable for a tiny
   model. Fix = run the controller at a **fine rate (200 Hz)** and subsample to
   20 Hz, logging the per-window **mean** command (the ZOH-equivalent). Exactly
   the paper's "resampled to 20 Hz." → chatter drops ~25×.
2. **Slow-motor oscillation.** SE3Control doesn't model motor lag, so for slow
   motors (high `tau_m`) it oscillates regardless of rate (chatter 1.7 N at
   `tau_m`=0.083 vs 0.018 N at 0.015). The paper's NMPC models the motor; ours
   can't, so we cap Table I's `tau_m` range to **[0.01, 0.03]**. Documented
   deviation, forced by the simpler controller — not a model change.

Combined (fine-rate control + `tau_m`≤0.03), step-to-step action change is
≈0.1 N (25× smoother), and the same ~9K model predicts far more accurately.
`simulate_trajectory(dt_control=0.005)` implements the substepping.

## Simulator (drone_jepa/data_gen/)

- **Vehicle: RotorPy `hummingbird` (0.5 kg), not a 1.3 kg drone.** RotorPy's
  bundled `SE3Control` has fixed gains tuned for the hummingbird; the px4-sihsim
  (1 kg) and crazyflie param sets make the controller command moments that
  massively saturate the rotors (60 N on a 5 N rotor) → loss of control. For a
  Tier-1 *sim-only* repro the absolute scale is irrelevant; what matters is
  consistent randomized dynamics + the latent-prediction claim.
- **Control input** = `cmd_motor_thrusts` (four rotor forces) — a 1:1 match to
  the paper's action space.
- **Data-collection controller** = SE(3) geometric controller (RotorPy builtin)
  tracking random sum-of-sines references. The paper uses NMPC+MPPI for
  "complementary action distributions"; SE3 + persistently-exciting references
  is the pragmatic Tier-1 substitute. (Adding NMPC via acados/do-mpc is future.)
- **References** = per-axis sum of sinusoids (flatness-friendly: all derivatives
  closed-form) instead of the paper's GP sampler. Same role: smooth, diverse,
  persistently exciting full-state references.
- **Control at 20 Hz with zero-order-hold action.** Hover is exactly stable;
  the SE3 loop tracks dynamic references fine at 20 Hz for the hummingbird, so
  no fine-rate substepping is needed and the action stays consistent between
  training and (20 Hz) MPPI deployment. Diverged rollouts (≈10-15%, hard
  domains) are detected by `TrajResult.is_valid` and resampled.
- **Domain randomization (Table I)**: mass ±50%, inertia ±30%, motor τ ∈[0.01,0.1]s,
  drag absolute (tamed to [0.05,0.30] vs paper's [0.1,0.5] for sim stability),
  thrust coeff ±50%, torque coeff ±50%. One param set per domain.

## Scale

Defaults are laptop-sized (≈150 domains × 20 trajs). The paper is 500 domains ×
20000 trajs × 10 s. Pass larger `--domains/--per-domain` to scale up; collection
runs at ≈4 trajs/s.

## Prober inputs — the decisive design choice (diagnosed empirically)

The paper says the prober maps the latent → DKI residuals. A literal latent-only
prober **fails badly**: open-loop position RMSE explodes (≈10 m at 2 s, ≈40 m at
5 s) — far worse than the autoregressive baseline. We diagnosed why with a
teacher-forced ablation (decode from *encoded-true* latents): it was just as bad,
so the predictor's latent rollout was fine and the **prober+DKI decoder** was the
bottleneck. Two compounding root causes, fixed in order:

1. **Thrust-dependent residual.** The dominant error is the mass mismatch, whose
   acceleration `(1/m_nom − 1/m_true)·T·R e3` scales with thrust `T` (the action).
   A latent-only `dvdot` is action-independent and can't track it under aggressive
   (non-hover) thrust. → feed the **action** to the prober. (Helped a little.)
2. **Drift-blindness.** The prober reads the latent (which tracks the *true*
   trajectory), but the DKI integrates its *own* state, which drifts in open loop.
   Reading only the latent, the prober is blind to that drift and can't correct
   it, so the double-integration amplifies. → feed the DKI's **running state** to
   the prober. (Decisive: in-dist long-horizon RMSE 10.8 m → 1.5 m, now ≤ the
   strong baseline; compounds slower past the training horizon.)

So our prober reads **(latent, running state, action)**. This is a deviation from
the strict latent-only reading, but physically grounded (residual accel depends
on state & action; the latent supplies the domain/mass that a single state can't)
and necessary for the DKI decode to be stable. The variants are now a first-class
flag: `--prober-inputs {full,latent_action,latent}` (saved in the checkpoint config).

RECONFIRMED under ideal conditions (2026-07-02, bigquad unified pipeline): with the
SAME frozen stage-1 (`--init-from bigquad_unified`) and the best data (base+DAgger+SE3),
retraining only the prober: full = 0.59 m @1 s / 12-12 races; latent_action = 2.05 m
(rank-corr −0.17, actively anti-correlated); latent-only = 2.28 m / 0-12 races, worse
than the naive AR baseline at 2 s. CAVEAT: the latent_action RACE cell is not validly
measured — the Rust rollout used to infer the prober input width from the weight shape
and silently fed that blob latent+state[0..4] instead of latent+action (the latent-only
race was correct by coincidence: first 16 features = the latent). jepa.rs now asserts
input width instead of silently slicing. Open-loop/probe numbers (PyTorch) unaffected. The deviation is STRUCTURAL (drift-blindness — no
dataset can tell the prober where its own integrator drifted), not a data artifact,
and the running-state input is the load-bearing one.

## Result summary (qualitative, this sim)

- **vs a strong baseline** (residual-delta, multi-step-trained autoregressive):
  in-distribution JEPA is ~2× better at 1 step, ~tied mid-horizon, slightly better
  at long horizon, and grows slower past the training horizon. OOD (domains beyond
  Table I), this baseline is unusually robust and beats JEPA at long horizon —
  partly because the fixed `m_nom` is a poor anchor far OOD.
- **vs a naive autoregressive baseline** (1-step-trained, rolled out): the classic
  compounding failure the paper highlights — JEPA tames it (see eval output).
- Exact RMSE differs from the paper (different sim/vehicle); the trends are the point.

### Milestone-4 numbers (in-distribution, horizon 40 = 2 s, train T=20), pos RMSE [m]

| k | JEPA | naive AR (1-step) | strong AR (multi-step) |
|---|------|-------------------|------------------------|
| 1 | 0.03 | 0.06 | 0.07 |
| 20 | 0.99 | 5.53 | 0.96 |
| 40 | 1.54 | 15.90 | 1.63 |

JEPA vs naive autoregressive: **10× lower** error at 2 s and compounds far slower
(1.57× vs 2.88× past the training horizon) — the paper's qualitative claim. ✓

## Deployment MPPI status

`control/mppi.py` + `scripts/demo_mppi.py` run the full deployment loop and the
drone flies and tracks short-term, but long-horizon tracking RMSE is poor (~6 m).
Root cause (not a bug): the world model is trained only on **SE3-controller**
actions — a narrow distribution — so MPPI's randomly-sampled candidate action
sequences are out-of-distribution and the model ranks them unreliably. The paper
collects **NMPC+MPPI** trajectories for exactly this reason ("complementary
action distributions"). Adding a diverse/Gaussian-perturbed action data source is
the clear next step to make MPPI track well.

## The fragility investigation — "open-loop error is blind; flyability is a weight-init lottery"

This is the deepest result of the project (a candidate blog post). It started from an
observation: a **width-2 rotor-force JEPA trained on the "full mix"** (base MPPI data +
2000 smooth SE3 + 800 aggressive-recovery SE3 = 12,800 trajs) **crashes every race**
(0/12, ~35% of steps inverted), while the same recipe on the "small mix" (base + 500
smooth + 200 recovery) flies 11/12. Seven experiments traced *why*. All flight numbers
are the deterministic `rotor_fly` race (12 seeded sky-gate courses); artifacts under
`artifacts/blog_cliff/RESULTS*.txt` and checkpoints `artifacts/blog_*.pt`.

**The one-line finding:** *closed-loop flyability is decided at weight initialization —
but it is a genuine multi-basin loss-landscape property, not a fixable init-recipe bug —
and it is completely invisible to the open-loop rollout error that world-model papers report.*

### Exp 1 — open-loop prediction error is blind to the crash
All w2 models — and all seeds of the crashing full-mix — have **statistically identical**
open-loop position RMSE on the shared `dataset_rf_mppi.pt` test split, yet flight ranges
from 0/12 to 12/12. (`python -m drone_jepa.eval.openloop --jepa <ckpt> --data dataset_rf_mppi.pt`)

| model | 1-step | 20-step | 40-step (2 s) | flight |
|---|---|---|---|---|
| base w2 | 0.0086 | 0.582 | 1.604 | 8–11/12 |
| small-mix w2 | 0.0076 | 0.577 | 1.597 | 11/12 |
| full-mix w2 seed 0 | 0.0086 | 0.578 | 1.596 | **0/12** |
| full-mix w2 seed 2 | 0.0085 | 0.583 | 1.606 | **12/12** |

The lethal model is the 2nd-best *predictor*. The standard metric cannot see the catastrophe.

### Exp 2 — there is no "aggressive-data cliff"
Holding smooth=500 and sweeping recovery ∈ {0,200,400,800} (w2, seed 0): 9/11/9/11 of 12,
**zero crashes**. Recovery volume is not the lever — the tidy "too much aggressive data
→ exploitation" hypothesis is **refuted**. (rec800 flies 11/12; the full mix that crashes
adds only ~1500 more *smooth* trajectories.)

### Exp 3 — the crash is a seed lottery
Same full-mix data + recipe, vary the seed: **most seeds crash, some fly.**
seeds 41(orig)/0/1 → 0/12; seed 2 → 12/12; at w1, seed 0 → 1/12 (milder — width amplifies
the instability but doesn't create it).

### Exp 4 — base & small-mix never crash; only the full mixture has a crash basin
base(seeds 0/1/2)=11/8/10 of 12; small-mix(0/1/2)=11/9/11 of 12. Both robustly flyable,
**zero crashes across seeds**. The `mix_small` win over base is real, not a lucky seed.

### Exp 5 — the lottery lives in STAGE-1 (latent dynamics), not the prober
Via `--init-from` (reuse a trained stage-1, retrain only the prober):
- crashing stage-1 (seed 0) + prober seeds {1,2,3} → **0/12, 0/12, 0/12** (all crash)
- flying stage-1 (seed 2) + prober seeds {1,2,3} → **10/12, 11/12, 10/12** (all fly)

A crashing stage-1 can't be rescued by any prober; a flying one flies with any prober.
This also mechanistically explains the older uncertainty finding ("cheap ensemble sharing
stage-1 = no help; independent stage-1 ensemble helps") — exploitation is a stage-1 property.

### Exp 6 — it's WEIGHT-INIT, not data-order (decoupled factorial)
Added `--init-seed` / `--data-seed` to `train.py` to separate the two things the global
seed controls. Full-mix w2 factorial:
- **vary init (data fixed at 0):** init 0→0/12, 1→0/12, 2→**11/12 fly**, 3→0/12 (stable-but-
  timid, 0.6 gates, only 0.9% flipped) — init spans the full outcome *spectrum*.
- **vary data-order (init fixed at 0):** data 0/1/2/3 → **all 0/12** (28–39% flipped every time).

Data shuffle cannot rescue a bad init; init alone reproduces crash/fly/timid. **Weight
init is the sole driver.** The mechanism is the GRU recurrent matrix's seed-dependent
spectral radius under PyTorch's default `uniform(±1/√H)` init.

### Exp 7 — principled init does NOT fix it (clean negative result)
Built `drone_jepa/model/weight_init.py` + `--init-scheme orthogonal`: He-normal conv/linear,
**orthogonal per-gate GRU recurrent** (spectral radius pinned to 1.0 every seed, vs default's
seed-dependent 0.55–0.65), Xavier input-hidden, and the **update-gate carry-bias +1**
(Jozefowicz 2015, the GRU forget-bias analog). Crash-rate over matched full-mix w2 seeds:

| init scheme | fly rate |
|---|---|
| PyTorch default | 1/5 |
| orthogonal + He (no gate bias) | 0/2 |
| **complete literature recipe** (orthogonal + He + gate bias) | **1/4** |

All ~the same ~20% base rate. Pinning the spectral radius and biasing the gates toward
stability — the two textbook stabilizers — **do not move the crash rate**. So the basin is
*set* at init but is **not an init-recipe artifact**: full-mix + this tiny model is a genuine
multi-basin landscape where ~80% of inits flow to an MPPI-exploitable minimum.

### What actually works (and the takeaway)
Neither a smarter init distribution, a bigger model (width-2 is *worse* than the paper's
~9K), nor open-loop-error selection fixes this. The only levers that don't depend on a
lucky init:
1. **Model selection on a control-relevant metric** (a cheap MPPI/flight-proxy score) —
   `train.py` currently ships the *final* step blindly, and `val_pos` is proven blind (Exp 1).
2. **Warm-start from a known flyer** (untested coda: does a good basin survive full-mix training?).
3. **Keep the model tiny + the data narrow** (base / small-mix never enter the crash basin).

Blog thesis: a tiny world model + MPPI reproduces SkyJEPA, but its fragility lives in the
**coupling**, not the network — flyability is a weight-init lottery in the latent-dynamics
stage, amplified by capacity and wide data, and invisible to the metric everyone reports.
New training knobs from this work: `--init-scheme {default,orthogonal}`, `--init-seed`,
`--data-seed`.

## Fragility campaign follow-up (E1–E9)

The open questions above (why the lottery, can it be detected/fixed) were answered by a
nine-experiment campaign — see **docs/EXPERIMENTS_fragility_campaign.md**. Headlines: a
2-second attitude probe on perturbed plans predicts fly/crash (`scripts/probe_plans.py`);
basins are loss-separated yet loss-equivalent at their optima; warm-starting inherits the
basin both ways; re-windowing the dataset flips basins (the lottery is init × pipeline
mechanics); EMA targets and early culling fail; DAgger data helps but isn't universal;
and on the wide fleet the analytic mass-aware DKI (`mass_aware=True`) beats
param-conditioning 8× (18× vs unconditioned) — Exp D's "null" was an artifact of the
blind open-loop metric.

## Known gaps / future work
- NMPC data-collection controller (acados/do-mpc) for complementary actions.
- GP reference sampler + differential flatness (we use sum-of-sines).
- Larger-scale training matching the paper's 50 epochs / batch 2048.
- Tier 2 (sim-to-real) entirely out of scope.
