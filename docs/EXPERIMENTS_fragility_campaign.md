# Fragility campaign — explaining the training-set / weight-init sensitivity

Follow-up campaign (2026-07-01/02) to the blog_cliff seed-lottery investigation and the
wide-fleet work. Nine experiments (E1–E9) targeting: *why* is closed-loop flyability a
weight-init lottery invisible to open-loop error, can we detect/select it cheaply, and
can we remove it (data, objective, or physics-side fixes).

Infrastructure notes: probe tooling in the session scratchpad (`probe.py`, `e12_rankfid.py`,
`e7_interp.py`, `e34_flytest.py`, `e8_mix.py`); new repo pieces: `train.py --warm-start`
/ `--ema-tau`, `train_pc.py --arms uncond,cond,mass --save-prefix`, `SkyJEPA(mass_aware=)`,
`examples/dagger_collect.rs`, `scripts/experiments/e_queue*.sh`. All flights are the deterministic
`rotor_fly` 12-race benchmark unless noted.

## E1+E2 — A flight-predicting metric the standard eval can't see  ✅ DONE

Scored all 29 labeled blog_cliff checkpoints against ONE shared ground truth: 8 hover-ish
hummingbird states × 96 deployment-like rotor-force plans (hover nominal, deployed
sigma=0.22·hover & beta=0.85 smoothing, 1/8 aggressive thrust offsets), rolled through the
true sim for 0.6 s.

| metric (group mean±sd) | CRASH (17) | FLY (9) |
|---|---|---|
| ds_pos40 (dataset actions, 2 s) | 2.366±0.015 | 2.358±0.020 (blind ✓) |
| ds_rot40 | 90.3° | 90.4° (blind) |
| plan_pos (perturbed plans, 0.6 s) | 0.579 | 0.587 (STILL blind) |
| **plan_rot (perturbed plans, 0.6 s)** | **47.4°±7.5** | **29.4°±3.1** |
| **inversion-miss** (pred vs real body-z flip) | **12.6%** | **6.8%** |
| rank-corr(pred cost, real cost) | 0.82 | 0.89 (weak separator) |

- 15/17 crashers have plan_rot ≥ 46°; 0/9 fliers above 34°. Exceptions: fullmix_w2_s1,
  sep_i1_d0_w2 (crash but fly-like probe — a second failure mode this probe misses).
- **The exploitable defect is in the ATTITUDE channel under deployment-like actions.**
  Position error is blind even on the perturbed plans. Matches the "steps flipped" crash mode.
- loc_crashS1_p* / loc_flyS1_p* cluster exactly by stage-1 parent → re-confirms stage-1.
- base-data models rank-corr 0.98–0.99 vs full-mix ~0.8: the fuller mixture costs rank
  fidelity for everyone; the crashers additionally mis-integrate attitude.
- Cost: ~30 s one-time ground truth + ~1.5 s per checkpoint, no flying. **This is the
  model-selection criterion train.py was missing** (select/reject on plan_rot ≈ 40°).

## E7 — Crash and fly solutions are genuinely different loss basins  ✅ DONE

Linear weight interpolation crash(i0_d0) ↔ fly(i2_d0), same data seed. val_pred (latent
MSE): 0.0003 → **0.53 at α=0.5** → 0.0000; plan_rot 51 → 134 → 32. A real loss barrier —
"multi-basin" is literal. At the endpoints val_pred is near-identical while plan_rot
differs 51° vs 32° (the blindness again). Flight sanity: α=0.125 (plan_rot 112°) → 12/12
hard-crash; α=0.875 (69°) → 0 crashes but timid (2.1 gates) — probe ordering holds
off-manifold too.

## E5a — Warm-starting from a flyer SURVIVES full-mix retraining  ✅ DONE

Full 8000/8000 retraining on full-mix, warm-started from the flyer blog_fullmix_w2_s2:

| run | flight | probe |
|---|---|---|
| warm-fly, data-seed 1 | **11/12, 0 crashes, 4.8 gates** | plan_rot 27.0°, inv 6.9% |
| warm-fly, data-seed 2 | **11/12, 0 crashes, 4.9 gates** | plan_rot 23.8°, inv 6.2% |
| warm-crash control (from sep_i0_d0) | **0/12, 12 crashes, 30% flipped** | plan_rot 45.0°, inv 12.2% |

Symmetric heritability: continued full-mix training keeps a flyer flying AND a crasher
crashing (fresh data order does not dislodge either basin — consistent with Exp 6's
"data-order can't rescue"). → **Warm-start from a known flyer is a reliable rescue**;
combined with the E1 probe (select the flyer cheaply), the lottery is practically closed.

## E5b — When does the basin become detectable?  ✅ DONE (negative for early culling)

Shorts replicating the full run's own early steps (same seed ⇒ identical init+data order;
lr identical inside the 4000-step warmup, so each short == the full run's own prefix):

| stage-1 step | flyer s2 vs crashers |
|---|---|
| 600 | all identical (plan_rot 102–109°) |
| 2000 | all identical (93–127°; s2=100) |
| 4000 (warmup end) | still no separation; ordering even misleading (crasher s1 lowest at 68°, flyer s2 at 104°) |

**The basin is *determined* at init (Exp 6) but *expressed* only in the LR-decay half of
training (4000→8000).** Early culling is not possible; probe at the END of training
(cheap anyway: full training ≈ 10 min, probe ≈ 2 s).

## E6 — EMA target encoder vs the lottery  ✅ DONE (clean negative)

`--ema-tau 0.99` stage-1 target encoder, seeds 41,0,1,2,3,4 matched to the labeled
default arm (default outcome on these seeds: 1 fly / 1 marginal / 4 crash).

**EMA arm: 0/6 fly — strictly worse.** Even seed 2 (the default arm's flyer) crashes
(18.5% flipped). Probe plan_rot 94–132° on every seed (fly cluster is <34°). Mechanism:
with a lagging target the latent loss converges to val_pred ≈ 0.16 vs the default's
~0.0003 (no collapse, latent_std 0.78) — the tight symmetric fit is *load-bearing*, not
the pathology. Two seeds (s0, s4) do show a changed failure mode (flip 3–6%, endless
respawns instead of tumbling), but nothing flies. The encoder–predictor co-adaptation
hypothesis is NOT a lever: stop-grad/EMA does not close the crash basin.

## E9 — Localize the mixture cliff on the smooth axis  ⏳ running

smooth ∈ {500,1000,1500} × seeds {0,1,2} (rec=800 fixed; smooth=500 s0 ≈ blog_rec800 11/12;
smooth=2000 = full mix: crash-lottery). Question: gradual crash-rate rise or phase transition?

**DONE.** Fly-rate by smooth volume (seeds 0,1,2; sm500 s0 = blog_rec800):

| smooth | flights | crash rate |
|---|---|---|
| 500 | 11/12, 12/12, 11/12 | 0/3 |
| 1000 | 11/12, 10/12, 11/12 | 0/3 |
| 1500 | 11/12, **0/12 (20.9% flipped)**, 11/12 | **1/3** |
| 2000 (=full mix, prior) | mostly crash | ~3/4 |

**Progressive onset, not a sharp cliff**: the crash basin appears between smooth=1000 and
1500 and dominates by 2000. Adding more *smooth clean* SE3 data is what turns the lottery
on — plausibly because the smooth data dilutes the action-diverse MPPI base data, degrading
the off-dataset attitude behavior some basins rely on.

**Probe exception #3**: e9_sm1500_s1 crashes but probes fly-like (plan_rot 32.5°, all six
e9 checkpoints 27–33°). Together with fullmix_w2_s1 and sep_i1_d0 there is a second,
rarer crash mode the attitude probe misses (~3/20 crashers) — selection should combine
the probe with one flight test of the selected model rather than trust the probe alone.

## E3+E4 — Tell the model the drone: params (learned) vs mass (analytic)  ✅ (rerun for clean logs)

Three arms, same wide CTBR data (racing_pc, mass 0.2–2 kg asym arms), same init:
uncond / param-conditioned (n_params=10) / **mass-aware** (E4: hover-normalized actions +
true 1/m in the DKI — the RL policy's de-normalization trick, ~0 learned params).
Judged by closed-loop MPPI circle-tracking on 10 wide drones (the metric Exp D never used).

| arm | held-out open-loop RMSE @1 s | closed-loop median RMSE (10 wide drones) |
|---|---|---|
| uncond | 1.046 m | **17.2 m** (uncontrolled) |
| param-conditioned | 1.019 m (−3%) | **7.7 m** |
| **mass-aware (E4)** | 1.016 m (−3%) | **0.95 m** |

- **Exp D's "param-conditioning is null" is overturned in closed loop**: the same ~3%
  open-loop gain that looked negligible is a ~2× closed-loop improvement — RMSE-blindness
  strikes again (this time in the wide-fleet setting).
- **The analytic mass fix dominates**: hover-normalized actions + true 1/m in the DKI
  (zero learned params) takes the wide fleet from uncontrolled (17 m) to ~1 m tracking —
  ~18× better than uncond, ~8× better than learned conditioning. Most of the "capacity
  wall" was a UNITS problem: the model was asked to learn, per drone, what hover means,
  when the RL baseline was simply *told* (action de-normalization).
- Clean 12-drone rerun: uncond 18.7 m / cond 9.0 m / mass 1.4 m (medians). The mass-aware
  failures concentrate on the heavy long-arm drones (1.87 kg/32 cm → 17.7 m; 1.85 kg → 25.6 m;
  1.34 kg/32 cm → 19.1 m): mass normalization fixes translation, the residual failure is
  rotational (inertia/allocation mismatch) — the natural next step is the same analytic
  trick for the angular channel (scale K by nominal 1/I(m, arm)). Deploy-side requirement:
  an estimate of m (hover-thrust estimation suffices — the same info the RL policy consumes).

## E8 — Decision-aware data (DAgger)  ✅ DONE (mixed result + a windowing bombshell)

`dagger_collect.rs`: race the CRASHING model (fullmix_w2_s0) with the deployed planner
(high-spawn variant so tumble segments fit records), dump visited (state, action) 40-step
records (2502 collected, 2411 respawns). Retrain crash seeds on full-mix(chopped)+DAgger
vs chopped-only control:

| run (all seed-matched to known crashers) | flight | flip% | probe |
|---|---|---|---|
| ctrl seed 0 (chopped full-mix, NO dagger) | **10/12 FLY** | 2.2 | 30.5° |
| dagger seed 0 | **12/12, 0 respawns** (campaign best) | 0.7 | 26.9° |
| dagger seed 1 | timid (0.6 gates, 0 crashes) | 0.3 | 24.0° |
| dagger seed 41 | **crash** (5 hard, 79 respawns) | 13.4 | 27.0° (miss #4) |

Two findings:
1. **The windowing bombshell**: re-chopping the SAME full-mix data into 40-step records
   flipped seed 0 from 0/12-crash (robust across 4 data orders, Exp 6) to 10/12-fly. The
   basin selection depends on incidental dataset mechanics (window inventory/alignment),
   not init alone — "weight-init lottery" is really an "init × data-pipeline-details
   lottery", which explains why it resisted every principled init fix.
2. **DAgger data helps but doesn't close the basin**: on top of chopping it gives the
   campaign's best model (12/12, 0 respawns, and visibly reduced severity everywhere —
   flip 13.4% vs 27–39% for the worst seed) but seed 41 still crashes and seed 1 lands
   timid. On-policy data reduces exploit severity; it does not make training basin-free.

Probe post-mortem across the campaign: the plan_rot probe detects the SEVERE tumble mode
(flip ≥ 25%) with 100% hit-rate, but misses the milder crash modes (flip 13–23%): 4
stealth crashers of ~21 total. Use it to *rank/filter*, then confirm the winner with one
flight test.

## Final synthesis

**What the fragility is.** The full-mix loss landscape has multiple loss-equivalent basins
that differ in how latent dynamics + DKI integrate ATTITUDE under off-dataset
(deployment-like) actions (E1/E2). The basins are separated by a real loss barrier (E7)
yet indistinguishable by any dataset-action metric (E1, E7 endpoints). Which basin a run
lands in is decided by init (prior Exp 6) *and* by incidental data-pipeline mechanics —
re-windowing the same data flipped a robust crasher to a flyer (E8 control) — which is
why principled init recipes couldn't move it. The signature is only expressed late in
training (E5b), and severity rises with the smooth-data share of the mixture (E9:
onset between smooth=1000 and 1500).

**What doesn't fix it.** EMA/stop-grad targets (E6: strictly worse — the tight symmetric
latent fit is load-bearing), early culling (E5b), on-policy DAgger data as a universal fix
(E8: reduces severity, gives the campaign-best model on one seed, but another seed still
crashes).

**What works (the playbook).**
1. **Probe, don't trust val loss**: `scripts/probe_plans.py` (rotation error on
   deployment-like plans) ranks models in ~2 s; catches every severe (flip≥25%) crasher.
2. **Confirm the selected model with ONE flight test** (the probe misses mild crash
   modes, 4/21).
3. **Warm-start descendants from a confirmed flyer** — basins are heritable in both
   directions under continued training (E5a).
4. For wide fleets: **put drone identity into the physics, not the network** — hover-
   normalized actions + true 1/m in the DKI (E4) beats learned param-conditioning 8×
   and the unconditioned model 18× in closed loop, at zero learned params. The learned
   conditioning itself is NOT null in closed loop (E3 overturns Exp D's conclusion — a 3%
   open-loop gain was a 2× closed-loop gain).

**Adopted workflow (decision 2026-07-02).** Levers 1, 2 and 4 are now the default:
- `scripts/train_select.py` — trains N seeds, probe-ranks them
  (`drone_jepa/eval/probe.py`, GT cached at `artifacts/probe_gt.npz`), confirms
  candidates with one rotor_fly (gate: 0 hard crashes, <5% flipped), saves the winner as
  the canonical checkpoint and the warm-start parent for all future retrains.
- `python -m drone_jepa.eval.openloop --probe` — the standard eval now reports the
  deployment-action probe next to the (blind) RMSE table.
- The mass-aware DKI (E4) stays **opt-in** (`train_pc.py --arms mass`, `mass_aware=True`),
  not the default model.

**First production run (2026-07-02).** `scripts/train_select.py` ran on both rotor-force
recipes (4 seeds each):
- full-mix w2 → winner seed2 (12/12, 1.6% flipped) saved as `skyjepa_fullmix_w2.pt`. The
  gate worked where the probe is blind: the stealth-crasher seed1 probed best (26.5°) and
  was caught in flight (12 crashes). Seeds reproduced their original lottery outcomes.
- DAgger mix → winner seed0 (12/12, 0 respawns, 0.7% flipped — campaign best) saved as
  `skyjepa_rf_dagger.pt`. This run exposed and fixed a gate flaw: two TIMID basins
  (0 crashes but 0.1–0.6 gates/race) passed the original crash-only gate; the gate now
  also requires **≥4 gates/race**. Note the DAgger mixture's basin spread: of 4 seeds,
  1 races perfectly and 2 are timid — on-policy data shifts basins from "tumble" toward
  "timid" rather than eliminating the lottery.
- One-command wrapper: `scripts/pipeline.py --stem X --data Y -- <train args>` runs
  select → eval(--probe) → jblob deploy end-to-end.

## New-vehicle validation — the "bigquad" run (2026-07-02)

Full from-scratch test of the recipe on a NEW drone (`drones/bigquad.json`: 1.2 kg,
25 cm arms, TWR 3, tau_m 0.02 — deliberately heavier/slower than the hummingbird; all
harnesses take DRONE_* env overrides now, hummingbird stays the default). No reuse of
any prior model: fresh data (8000 rf-native + 8000 dual-action ctbr trajs, Table-I
randomization recentered on the bigquad), fresh AR baselines, fresh seeds.
Driver: `scripts/experiments/run_bigquad.sh` + `run_bigquad_round2.sh`; recipe: `drone_jepa/train_recipe.py`.

**Rotor-force.** All 4 seeds probed fly-like (14.7–17.9° — best of the project; rf-native
data spawns no crash basin, consistent with E9). Round-1 winner: 8/12 wins, 0 crashes,
4.4 gates, 2.7% flipped. One DAgger round (2503 records collected with the winner,
40-step chop, retrain warm-started from the winner): **both retrains score 12/12 wins,
0 crashes, 0 RESPAWNS, 5.0 gates/race** (0.1–0.6% flipped) → `bigquad_rf_v2.pt`. The
base→DAgger jump (8/12+13 respawns → 12/12+0) replicates the hummingbird pattern:
round-1-on-base-data is NOT the recipe's final quality; the DAgger round is where the
last respawns disappear.

**Benchmark (paper claim).** vs naive 1-step AR baseline: final-horizon position error
**7.9× lower** (hummingbird: ~10×) — the core SkyJEPA claim reproduces on a second
vehicle. vs the strong multi-step baseline: ~6× better at 1 step, tied at the training
horizon, 0.84× at 2 s — same picture as before.

**CTBR + a gate-calibration lesson.** All 4 ctbr seeds failed the demo-tuned jepa_fly
gate (0.3–1.3 gates, 8–12 crashes/race). The fleet reference on a bigquad-pinned fleet
explains most of it: **the perfect-model true-MPPI ceiling on this vehicle is only
1.6 gates/race** (the sluggish rate loop — 5× inertia at the same k_w — caps open-loop
planned racing), so the ≥2.5-gates gate was set above what a perfect model can do. The
JEPA's 1.1 gates is ~70% of ceiling; its excess CRASHES are the real gap (ceiling: 0).
Notable: JEPA+RL-proposal flies the same fleet at 4.9 gates / 0 crashes. Consequences:
(i) `train_recipe` gate thresholds are now CLI-configurable and documented as
vehicle-relative — calibrate against a true-MPPI reference before judging; (ii) a ctbr
DAgger round (needs a ctbr collector variant) is the natural next fix for the crashes.

**Unified model + demo refresh (follow-up).** One JEPA now serves both web demos:
`bigquad_unified.pt` (= uni_d2: warm-retrained from the DAgger winner on the union mix
base+DAgger+SE3, data-seed 2) races 12/12 / 0 respawns / 5.0 gates AND follows a fresh
SE3 test path at 0.21 m @1 s (the race-only v2: 0.36 m). Dual-role selection table in
`artifacts/logs/bigquad_unified_select.log`. Demo changes: CTBR JEPA drone removed from
the race (below the vehicle's planning ceiling anyway); RL drone now ROTOR-controlled
(fresh PPO on raw per-rotor forces, 10/12 / 0 crashes / 4.6 gates natively — slightly
behind the unified JEPA-rotor's 12/12 on this vehicle); forecast demo got an aerobatic
reference (~4.3 m/s ≈ training-p95 pace, real altitude swings) + a live speed knob
(0.4-1.6x, phase-preserving) that lets you push the model past its training envelope
and watch forecast error respond.

**Tooling hardened by this run**: dataset/stem name-collision guard, winner-blob export
in train_recipe, generalized `scripts/dagger_mix.py`, DRONE_* env in rotor_fly /
jepa_fly / gen_dataset_rf / dagger_collect.

**The recurring meta-lesson** (now confirmed in five independent settings: Exp 1, Exp F,
E1/E2, E3, E7): open-loop error on dataset actions does not measure what a planner needs.
Every evaluation of a world model intended for control should include (i) error under the
*deployment action distribution*, split by state component, and (ii) a closed-loop test.
