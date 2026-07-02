"""Deployment-action plan probe — the flight-predicting metric.

Rolls MPPI-like candidate plans (hover nominal, deployed sigma/EMA smoothing,
a slice of aggressive thrust offsets) through the TRUE sim once (cached), then
scores any checkpoint's predictions on them, per state component. Findings
(docs/EXPERIMENTS_fragility_campaign.md):

  - plan_rot (rotation geodesic error @0.6 s) separates crash from fly models
    (~47 deg vs ~29 deg) where ALL dataset-action metrics are blind;
  - position error is blind even here — the defect is the attitude channel;
  - catches every severe tumbler (>=25% steps flipped); misses milder crash
    modes — rank/filter with the probe, then CONFIRM with one closed-loop race;
  - only meaningful on FULLY-TRAINED checkpoints (basins express late).

Supports both action modes and arbitrary vehicles: pass a drone-spec dict
(mass, ixx, iyy, izz, arm, k_eta, k_m, tau_m, k_w) — defaults to the 0.5 kg
hummingbird — and action_mode "rotor_force" (4 rotor forces) or "ctbr"
([thrust, wx, wy, wz] through the inner rate loop). Ground truth is cached
per (drone, mode).

CLI:  python -m drone_jepa.eval.probe [--drone spec.json] [--mode ctbr] stems...
"""
from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import torch

from ..data_gen.sim import _state_to_vec, random_initial_state, rotor_force_limit
from ..model.jepa import SkyJEPA

N_STATES, S, T, H = 8, 96, 12, 10
DT, BETA = 0.05, 0.85
SIG_FRAC = 0.22            # thrust sigma = 0.22 * hover (deployed planner value)
SIG_RATE = 1.5             # body-rate sigma [rad/s] (ctbr)
FLY_THRESHOLD_DEG = 40.0   # plan_rot above this = crash-basin signature


def quad_params_from_spec(spec: dict | None) -> dict:
    """RotorPy param dict for a drone-spec (None = stock hummingbird)."""
    from rotorpy.vehicles.hummingbird_params import quad_params as BASE
    p = copy.deepcopy(BASE)
    if not spec:
        p["k_w"] = 16.0  # match the Rust harness's inner-rate-loop gain
        return p
    d = spec.get("arm", 0.17) / np.sqrt(2.0)
    p.update(
        mass=spec["mass"], Ixx=spec["ixx"], Iyy=spec["iyy"], Izz=spec["izz"],
        k_eta=spec["k_eta"], k_m=spec["k_m"], tau_m=spec["tau_m"],
        k_w=spec.get("k_w", 16.0),
        rotor_pos={"r1": np.array([d, d, 0.0]), "r2": np.array([d, -d, 0.0]),
                   "r3": np.array([-d, -d, 0.0]), "r4": np.array([-d, d, 0.0])})
    return p


def _true_rollout(params, st0, plan, mode):
    from rotorpy.vehicles.multirotor import Multirotor
    abstraction = "cmd_ctbr" if mode == "ctbr" else "cmd_motor_thrusts"
    veh = Multirotor(params, control_abstraction=abstraction, aero=True)
    st = copy.deepcopy(st0)
    traj = []
    for a in plan:
        if mode == "ctbr":
            st = veh.step(st, {"cmd_thrust": float(a[0]),
                               "cmd_w": np.asarray(a[1:], float)}, DT)
        else:
            st = veh.step(st, {"cmd_motor_thrusts": np.asarray(a, float)}, DT)
        x = _state_to_vec(st)
        if not np.isfinite(x).all():
            break
        traj.append(x)
    while len(traj) < len(plan):
        traj.append(traj[-1] if traj else np.zeros(18))
    return np.array(traj)


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def rot_angle_deg(Rp, Rt):
    A = Rp.reshape(*Rp.shape[:-1], 3, 3)
    B = Rt.reshape(*Rt.shape[:-1], 3, 3)
    tr = np.einsum("...ij,...ij->...", A, B)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def _sample_plans(rng, hover, lo, hi):
    """(S,T,4) candidate plans: hover nominal + EMA-smoothed noise, with a
    slice of aggressive thrust offsets (the climb/dive regime planners probe)."""
    sig = np.where(hover > 0, SIG_FRAC * hover, SIG_RATE)
    plans = np.zeros((S, T, 4))
    for s in range(S):
        off = np.zeros(4)
        if s < S // 8:
            off = rng.uniform(-0.35, 0.5) * hover     # thrust channels only
        raw = rng.normal(0.0, 1.0, (T, 4)) * sig
        sm, acc = np.zeros_like(raw), raw[0]
        for k in range(T):
            acc = BETA * acc + (1 - BETA) * raw[k]
            sm[k] = acc
        sm *= sig.mean() / max(sm.std(), 1e-9)
        plans[s] = hover + off + sm
    return np.clip(plans, lo, hi)


def gt_cache_path(drone_name: str | None, mode: str) -> str:
    tag = f"{drone_name or 'hummingbird'}_{mode}"
    return f"artifacts/probe_gt_{tag}.npz"


def get_gt(cache: str | None = None, drone: dict | None = None,
           action_mode: str = "rotor_force", drone_name: str | None = None):
    """(x0s, plans, real, targets, hover(4,)) — built once, cached."""
    cache = cache or gt_cache_path(drone_name, action_mode)
    if os.path.exists(cache):
        z = np.load(cache)
        return z["x0"], z["plans"], z["real"], z["tgt"], z["hover"]
    params = quad_params_from_spec(drone)
    f_max = rotor_force_limit(params)
    mg = params["mass"] * 9.81
    if action_mode == "ctbr":
        hover = np.array([mg, 0.0, 0.0, 0.0])
        lo = np.array([0.0, -12.0, -12.0, -12.0])
        hi = np.array([4 * f_max, 12.0, 12.0, 12.0])
    else:
        hover = np.full(4, mg / 4.0)
        lo, hi = np.zeros(4), np.full(4, f_max)
    rng = np.random.default_rng(7)
    x0s, plans_l, real_l, tgt_l = [], [], [], []
    for _ in range(N_STATES):
        init = random_initial_state(rng, center=np.array([0.0, 0.0, 1.5]))
        x0 = _state_to_vec(init)
        plans = _sample_plans(rng, hover, lo, hi)
        real = np.stack([_true_rollout(params, init, plans[s], action_mode)
                         for s in range(S)])
        x0s.append(x0); plans_l.append(plans); real_l.append(real)
        tgt_l.append(x0[:3] + np.array([2.0, 0.0, 3.0]))
    x0s, plans_l, real_l, tgt_l = map(np.array, (x0s, plans_l, real_l, tgt_l))
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    np.savez(cache, x0=x0s, plans=plans_l, real=real_l, tgt=tgt_l, hover=hover)
    return x0s, plans_l, real_l, tgt_l, hover


@torch.no_grad()
def probe_model(model, gt) -> dict:
    """rank / plan_rot / inv / plan_pos for one model (mode must match gt)."""
    x0s, plans_l, real_l, tgt_l, hover = gt
    hover_t = torch.tensor(np.asarray(hover, dtype=np.float32))
    sps, rot_e, inv, pos_e = [], [], [], []
    for x0, plans, real, tgt in zip(x0s, plans_l, real_l, tgt_l):
        hist = torch.tensor(np.tile(x0, (H, 1)), dtype=torch.float32)
        cand = torch.tensor(plans, dtype=torch.float32)
        win = torch.zeros(S, H + T, 4)
        win[:, :H - 1] = hover_t
        win[:, H - 1:H - 1 + T] = cand
        win[:, H - 1 + T:] = cand[:, -1:]
        pred = model.predict(hist.unsqueeze(0).expand(S, -1, -1), win, horizon=T).numpy()
        pc = ((pred[..., :3] - tgt) ** 2).sum(-1).mean(-1)
        rc = ((real[..., :3] - tgt) ** 2).sum(-1).mean(-1)
        sps.append(_spearman(pc, rc))
        rot_e.append(rot_angle_deg(pred[..., 6:15], real[..., 6:15])[:, -1].mean())
        inv.append(float(((real[..., 14] < 0) != (pred[..., 14] < 0)).mean()))
        pos_e.append(np.linalg.norm(pred[..., :3] - real[..., :3], axis=-1)[:, -1].mean())
    return dict(rank=float(np.mean(sps)), plan_rot=float(np.mean(rot_e)),
                inv=float(np.mean(inv)), plan_pos=float(np.mean(pos_e)))


def probe_checkpoint(path: str, gt) -> dict:
    model, _ = SkyJEPA.from_checkpoint(path, device="cpu")
    model.eval()
    return probe_model(model, gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drone", default=None, help="drone spec JSON path")
    ap.add_argument("--mode", default="rotor_force", choices=["rotor_force", "ctbr"])
    ap.add_argument("stems", nargs="+")
    args = ap.parse_args()
    spec = json.loads(open(args.drone).read()) if args.drone else None
    name = os.path.splitext(os.path.basename(args.drone))[0] if args.drone else None
    gt = get_gt(drone=spec, action_mode=args.mode, drone_name=name)
    for stem in args.stems:
        path = stem if stem.endswith(".pt") else f"artifacts/{stem}.pt"
        try:
            r = probe_checkpoint(path, gt)
        except FileNotFoundError:
            print(f"{stem:<26} MISSING")
            continue
        flag = "CRASH-SIGNATURE" if r["plan_rot"] > FLY_THRESHOLD_DEG else "fly-like"
        print(f"{stem:<26} rank={r['rank']:+.2f} plan_rot={r['plan_rot']:6.1f} "
              f"inv={r['inv']*100:5.1f}% plan_pos={r['plan_pos']:.2f}  [{flag}]",
              flush=True)


if __name__ == "__main__":
    main()
