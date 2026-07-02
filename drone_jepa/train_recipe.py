"""The SkyJEPA training recipe — train, select, confirm.

This file is the distilled, runnable result of the fragility campaign
(docs/EXPERIMENTS_fragility_campaign.md). It exists because of one hard-won
fact: for a tiny world model driving a sampling-based planner,

    **whether a trained model can FLY is not a property the training loss can
    see.** Two checkpoints with statistically identical validation error can
    differ between 12/12 flawless races and crashing on every single one.

The training loop below is therefore deliberately boring — it is the paper's
two-stage procedure with no tricks — and the value is in what surrounds it:
train several seeds, rank them with a metric that *does* predict flight, and
confirm the winner with one real closed-loop test.

The recipe in five lines
------------------------
1.  Train N seeds of the same configuration (~10 min each at this scale).
2.  Rank them with the deployment-action probe (rotation error on MPPI-like
    candidate plans — `drone_jepa.eval.probe`). Reject the crash signature.
3.  Confirm candidates best-first with ONE closed-loop race; gate on
    0 hard crashes, <5% inverted steps, >=4 gates/race.
4.  Ship the confirmed winner; warm-start every future retrain from it
    (basins are heritable under continued training — you exit the lottery).
5.  Optionally: collect one DAgger round with the deployed planner and
    repeat 1-4 on the enriched mixture (produced our best model).

What we tried that does NOT work (so you don't have to):
  - Selecting on validation loss / open-loop RMSE (provably blind).
  - EMA / stop-grad target encoders (strictly worse: the symmetric latent
    loss is load-bearing, not a bug — collapse is prevented by SIGReg).
  - Orthogonal/He/gate-bias init schemes (the fly-or-crash basin is set at
    init but is not an init-statistics artifact).
  - Early stopping/culling on any metric (basins only express themselves in
    the LR-decay half of training).
  - Scaling the model (capacity amplifies the basin you land in, good or bad).

Usage
-----
    python -m drone_jepa.train_recipe \
        --data artifacts/my_dataset.pt --stem my_model \
        [--drone drones/bigquad.json]      # non-default vehicle (see below)
        [--n-seeds 4] [--device mps] [--width-mult 1]
        [--stage1-steps 8000] [--stage2-steps 8000]
        [--warm-start artifacts/known_flyer.pt]   # skip the lottery entirely
        [--no-fly]                          # skip flight gate (non rotor-force)

The drone spec JSON (single source of truth for sim, probe and race harness)
holds: mass, ixx, iyy, izz, arm, k_eta, k_m, tau_m. Omit --drone for the
default 0.5 kg hummingbird.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data_gen.dataset import load_dataset
from .model.jepa import SkyJEPA
from .model.losses import latent_loss, physical_loss

ROOT = Path(__file__).resolve().parents[1]
RUST = ROOT / "web-demo" / "racer"

# Flight gates (deterministic closed-loop race benchmarks). The criteria
# reject the three observed bad-basin phenotypes: tumblers (crashes/flipped),
# stealth-crashers (crashes with low flip), and timid hoverers (low gates).
# rotor_force -> examples/rotor_fly (12 races, win/crash/flip accounting).
GATE_RF_MAX_CRASHES = 0
GATE_RF_MAX_FLIPPED = 5.0    # % of steps past 90 deg roll/pitch
GATE_RF_MIN_GATES = 4.0      # gates/race out of 5
RF_SUMMARY = re.compile(
    r"== (?P<wins>\d+)/(?P<trials>\d+) WON, (?P<crashes>\d+) hard-crashed, "
    r"(?P<respawns>\d+) respawns, (?P<gates>[0-9.]+) gates/race, "
    r"(?P<flipped>[0-9.]+)% steps flipped ==")
# ctbr -> examples/jepa_fly (10 races, respawn-on-crash, gates/crashes per race).
GATE_CTBR_MAX_CRASHES = 0.5  # crashes/race
GATE_CTBR_MIN_GATES = 2.5    # gates/race
CTBR_SUMMARY = re.compile(
    r"== over \d+ races: \d+ gates, \d+ crashes "
    r"\((?P<gates>[0-9.]+) gates/race, (?P<crashes>[0-9.]+) crashes/race\) ==")


# --------------------------------------------------------------------------
# Optimizer schedule — the paper's: Adam(wd=1e-5), grad-clip 0.5, linear
# warmup to 5e-3 then cosine to 1e-4. NOTE the warmup span (4000 steps) is
# deliberately long relative to an 8000-step budget: the fly/crash basin is
# decided during the decay phase, and shortening warmup was never needed.
# --------------------------------------------------------------------------
def lr_at(step: int, total: int, warmup: int = 4000,
          lr_max: float = 5e-3, lr_min: float = 1e-4) -> float:
    if step < warmup:
        return lr_max * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(t, 1.0)))


def train_stage1(model: SkyJEPA, loader, steps: int, device,
                 lambda_sig: float = 0.02) -> None:
    """Stage 1 — latent dynamics (encoders + predictor).

    Loss = multi-step latent consistency + lambda_sig * SIGReg.

    Two intentional non-choices, both verified the hard way:
      * The consistency term is SYMMETRIC — gradients flow into the target
        latents too. No stop-grad, no EMA target encoder. Collapse is
        prevented by SIGReg alone; adding an EMA target degraded the latent
        fit ~500x and produced zero flyable models across seeds.
      * Nothing about this loop selects for a flyable model. Do not add
        val-loss early stopping — it measures the wrong thing.
    """
    params = (list(model.state_encoder.parameters())
              + list(model.action_encoder.parameters())
              + list(model.predictor.parameters()))
    opt = torch.optim.Adam(params, lr=0.0, weight_decay=1e-5)
    model.train()
    step = 0
    while step < steps:
        for X, A in loader:
            if step >= steps:
                break
            X, A = X.to(device), A.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, steps)
            out = model.latent_forward(X, A)
            loss, comps = latent_loss(out, lambda_sig=lambda_sig)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            if step % max(1, steps // 10) == 0:
                # latent std is the collapse probe: healthy runs sit ~0.7-0.8;
                # a slide toward 0 means SIGReg lost (raise lambda_sig).
                std = out.s_all.reshape(-1, out.s_all.shape[-1]).std(0).mean()
                print(f"  [s1 {step:5d}/{steps}] pred={comps['pred']:.4f} "
                      f"sigreg={comps['sigreg']:.4f} latent_std={std:.2f}", flush=True)
            step += 1


def train_stage2(model: SkyJEPA, loader, steps: int, device) -> None:
    """Stage 2 — prober + DKI decode, on FROZEN stage-1.

    Supervised L2 on the physical state, in normalized units so position,
    velocity, rotation and body rates contribute comparably.

    The prober reads (latent, DKI running state, action) — NOT latent-only.
    This deviates from the strictest reading of the paper and it is the one
    deviation you must keep: a latent-only prober cannot see the DKI's own
    open-loop drift and the double integration diverges (~10 m at 2 s).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.prober.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.prober.parameters(), lr=0.0, weight_decay=1e-5)
    H, T = model.H, model.T
    model.train()
    step = 0
    while step < steps:
        for X, A in loader:
            if step >= steps:
                break
            X, A = X.to(device), A.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, steps)
            with torch.no_grad():                       # stage 1 stays frozen
                out = model.latent_forward(X, A)
            pred = model.physical_rollout(X, A, out.s_now, out.s_pred)
            loss, comps = physical_loss(pred, X[:, H:H + T], std=model.state_std)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.prober.parameters(), 0.5)
            opt.step()
            if step % max(1, steps // 10) == 0:
                print(f"  [s2 {step:5d}/{steps}] pos={comps['pos']:.3f} "
                      f"rot={comps['rot']:.3f}", flush=True)
            step += 1


def train_one_seed(args, seed: int, out: Path) -> None:
    """One complete training run. Reproducible: the seed fixes weight init
    AND data order (that determinism is what makes the selection loop
    meaningful — rerunning a seed reproduces its basin)."""
    torch.manual_seed(seed)
    ds = load_dataset(args.data, history=10, horizon=20, stride=args.stride)
    loader = DataLoader(ds["train"], batch_size=args.batch, shuffle=True,
                        drop_last=True)
    model = SkyJEPA(history=10, horizon=20, width_mult=args.width_mult,
                    action_mode=args.action_mode, pos_mode="relative")
    model = model.to(args.device)
    model.fit_normalization(ds["raw"]["states"], ds["raw"]["actions"])
    if args.warm_start:
        # Basin heritability: continued training from a confirmed flyer stays
        # a flyer (and from a crasher stays a crasher) — warm-starting is how
        # you stop playing the seed lottery after the first confirmed model.
        sd = torch.load(args.warm_start, weights_only=False)["model"]
        model.load_state_dict(sd)
        print(f"  warm-started from {args.warm_start}")
    n = sum(p.numel() for p in model.parameters())
    print(f"  seed {seed}: {n} params, {len(ds['train'])} train windows", flush=True)
    train_stage1(model, loader, args.stage1_steps, args.device)
    train_stage2(model, loader, args.stage2_steps, args.device)
    cfg = {"history": 10, "horizon": 20, "width_mult": args.width_mult,
           "prober_hidden": model.prober_hidden, "action_mode": args.action_mode,
           "probabilistic": False, "learn_thrust": False, "pos_mode": "relative"}
    torch.save({"model": model.state_dict(), "config": cfg}, out)


# --------------------------------------------------------------------------
# Selection: probe, then confirm in closed loop.
# --------------------------------------------------------------------------
def drone_env(spec: dict | None) -> dict:
    """Env vars understood by the Rust sim/race binaries (defaults=hummingbird)."""
    env = dict(os.environ)
    if spec:
        for k, v in spec.items():
            env[f"DRONE_{k.upper()}"] = str(v)
    return env


def race(stem: str, spec: dict | None, mode: str) -> dict | None:
    """Deterministic closed-loop race benchmark of an exported model."""
    subprocess.run([sys.executable, "scripts/export_jepa.py",
                    f"artifacts/{stem}.pt", stem], cwd=ROOT, capture_output=True)
    subprocess.run([sys.executable, "scripts/export_jepa_blob.py", stem],
                   cwd=ROOT, capture_output=True)
    env = drone_env(spec)
    if mode == "ctbr":
        env["JBLOB"] = f"assets/{stem}.jblob"
        binary, summary = "jepa_fly", CTBR_SUMMARY
    else:
        env["ROTOR_BLOB"] = f"assets/{stem}.jblob"
        binary, summary = "rotor_fly", RF_SUMMARY
    out = subprocess.run([str(RUST / f"target/release/examples/{binary}")],
                         cwd=RUST, env=env, capture_output=True, text=True)
    m = summary.search(out.stdout)
    return {k: float(v) for k, v in m.groupdict().items()} if m else None


def passes_gate(f: dict | None, mode: str) -> bool:
    if not f:
        return False
    if mode == "ctbr":
        return (f["crashes"] <= GATE_CTBR_MAX_CRASHES
                and f["gates"] >= GATE_CTBR_MIN_GATES)
    return (f["crashes"] <= GATE_RF_MAX_CRASHES
            and f["flipped"] < GATE_RF_MAX_FLIPPED
            and f["gates"] >= GATE_RF_MIN_GATES)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--drone", default=None,
                    help="drone spec JSON (mass, ixx, iyy, izz, arm, k_eta, "
                         "k_m, tau_m); default = hummingbird")
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--width-mult", type=float, default=1.0)
    ap.add_argument("--action-mode", choices=["rotor_force", "ctbr"],
                    default="rotor_force")
    ap.add_argument("--stage1-steps", type=int, default=8000)
    ap.add_argument("--stage2-steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--warm-start", default=None)
    ap.add_argument("--no-fly", action="store_true")
    # Gate thresholds are HARNESS- AND VEHICLE-RELATIVE: calibrate them against
    # a perfect-model reference on the same course (fleet_fly's true-MPPI row).
    # E.g. the 1.2 kg TWR-3 bigquad's true-MPPI ceiling is ~1.6 gates/race —
    # demanding 4 gates there would reject every model including a perfect one.
    ap.add_argument("--gate-min-gates", type=float, default=None,
                    help="override gates/race threshold (default: 4.0 rf, 2.5 ctbr)")
    ap.add_argument("--gate-max-crashes", type=float, default=None,
                    help="override crash threshold (default: 0 rf, 0.5/race ctbr)")
    args = ap.parse_args()
    global GATE_RF_MIN_GATES, GATE_CTBR_MIN_GATES, GATE_RF_MAX_CRASHES, GATE_CTBR_MAX_CRASHES
    if args.gate_min_gates is not None:
        GATE_RF_MIN_GATES = GATE_CTBR_MIN_GATES = args.gate_min_gates
    if args.gate_max_crashes is not None:
        GATE_RF_MAX_CRASHES = GATE_CTBR_MAX_CRASHES = args.gate_max_crashes

    spec = json.loads(Path(args.drone).read_text()) if args.drone else None

    # the winner is copied to artifacts/<stem>.pt — refuse a dataset with the
    # same path (a collision would overwrite the dataset with the checkpoint)
    if Path(args.data).resolve() == (ROOT / f"artifacts/{args.stem}.pt").resolve():
        raise SystemExit(f"--data and --stem collide on artifacts/{args.stem}.pt; "
                         "rename one (e.g. suffix the dataset with _data)")

    # ---- 1. train N seeds (idempotent: existing checkpoints are kept) ----
    names = []
    for s in range(args.n_seeds):
        name = f"{args.stem}_seed{s}"
        names.append(name)
        out = ROOT / f"artifacts/{name}.pt"
        if out.exists():
            print(f"[train] {name} exists — skipping")
            continue
        print(f"[train] {name}", flush=True)
        train_one_seed(args, s, out)

    # ---- 2. probe-rank (cheap, ~2 s/model after a one-time GT build) ----
    # The probe measures ROTATION error on deployment-like candidate plans
    # against the true sim — the one offline metric that tracks flyability.
    # Its ground truth is drone-specific, so a new vehicle gets its own cache.
    from .eval.probe import FLY_THRESHOLD_DEG, get_gt, probe_checkpoint
    drone_name = Path(args.drone).stem if args.drone else None
    gt = get_gt(drone=spec, action_mode=args.action_mode, drone_name=drone_name)
    ranked = []
    for name in names:
        r = probe_checkpoint(f"artifacts/{name}.pt", gt)
        ranked.append((r["plan_rot"], name))
        print(f"[probe] {name}: plan_rot={r['plan_rot']:.1f} deg "
              f"inv={r['inv']*100:.1f}% rank={r['rank']:+.2f}", flush=True)
    ranked.sort()

    # ---- 3. confirm best-first with one real race each ------------------
    # The probe catches the dominant (attitude-integration) bad basin but
    # misses two rarer phenotypes; the closed-loop gate is not optional.
    winner = None
    for prot, name in ranked:
        # The absolute crash-signature threshold is calibrated for rotor-force
        # models only; CTBR probe scores live on a different scale (the inner
        # rate loop reshapes attitude errors), so there the probe RANKS
        # candidates and the closed-loop gate alone decides.
        if args.action_mode == "rotor_force" and prot > FLY_THRESHOLD_DEG:
            print(f"[gate] {name}: crash signature ({prot:.1f} deg) — rejected")
            continue
        if args.no_fly:
            winner = name
            break
        f = race(name, spec, args.action_mode)
        print(f"[race] {name}: {f}", flush=True)
        if passes_gate(f, args.action_mode):
            winner = name
            break
        print(f"[gate] {name}: failed the flight gate — trying next")

    if winner is None:
        raise SystemExit(
            "No seed passed both gates. Train more seeds, or warm-start from "
            "a confirmed flyer of a related recipe. Do NOT ship the best "
            "val-loss model — that metric cannot see the failure.")
    shutil.copy(ROOT / f"artifacts/{winner}.pt", ROOT / f"artifacts/{args.stem}.pt")
    # export the deployable blob under the canonical stem too (downstream
    # harnesses — fleet_fly, dagger_collect, race sims — look for it)
    subprocess.run([sys.executable, "scripts/export_jepa.py",
                    f"artifacts/{args.stem}.pt", args.stem], cwd=ROOT, capture_output=True)
    subprocess.run([sys.executable, "scripts/export_jepa_blob.py", args.stem],
                   cwd=ROOT, capture_output=True)
    print(f"\nWINNER {winner} -> artifacts/{args.stem}.pt "
          f"(+ assets/{args.stem}.jblob)")
    print("Warm-start future retrains from it:\n"
          f"  python -m drone_jepa.train_recipe --warm-start artifacts/{args.stem}.pt ...")


if __name__ == "__main__":
    main()
