"""Param-conditioning experiment: does feeding the JEPA the drone's physical params
(mass/inertia/...) let a ~10K world model stay accurate across the WIDE drone fleet?

Trains conditioned (n_params=10) vs unconditioned (n_params=0) on the SAME wide data
+ per-trajectory params sidecar, and reports open-loop physical rollout error on
HELD-OUT drones (the honest generalization test). If conditioned << unconditioned,
the model was failing to infer the drone from history; conditioning supplies it.

  .venv/bin/python -m drone_jepa.train_pc --bin artifacts/racing_pc.bin --device mps
"""
from __future__ import annotations

import argparse
import struct

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .model.jepa import SkyJEPA
from .model.losses import latent_loss, physical_loss


def load_bin(path):
    buf = open(path, "rb").read()
    n_traj, steps, n_act = struct.unpack_from("<III", buf, 0)
    off = 12
    ns = n_traj * steps * 18
    states = np.frombuffer(buf, "<f4", ns, off).reshape(n_traj, steps, 18); off += ns * 4
    na = n_traj * steps * n_act
    actions = np.frombuffer(buf, "<f4", na, off).reshape(n_traj, steps, n_act)
    return states, actions, n_act


def load_params(path):
    b = open(path, "rb").read()
    n_traj, NP = struct.unpack_from("<II", b, 0)
    p = np.frombuffer(b, "<f4", n_traj * NP, 8).reshape(n_traj, NP)
    return p


class PWindows(Dataset):
    """(X, A, P) windows. P is the trajectory's param vector (constant over the window)."""
    def __init__(self, states, actions, params, trajs, H, T, stride):
        self.s, self.a, self.p = states, actions, params
        self.L = H + T
        self.idx = []
        S = states.shape[1]
        for n in trajs:
            for st in range(0, S - self.L + 1, stride):
                self.idx.append((n, st))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        n, st = self.idx[i]
        X = torch.from_numpy(self.s[n, st:st + self.L].copy())
        A = torch.from_numpy(self.a[n, st:st + self.L].copy())
        P = torch.from_numpy(self.p[n].copy())
        return X, A, P


def lr_at(step, total, warmup=2000, lr_max=5e-3, lr_min=1e-4):
    import math
    if step < warmup:
        return lr_max * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


@torch.no_grad()
def openloop_pos_rmse(model, loader, device, horizons=(5, 10, 20)):
    """Mean position RMSE [m] at given horizon steps over held-out drones."""
    model.eval()
    H = model.H
    sums = {h: 0.0 for h in horizons}
    n = 0
    for X, A, P in loader:
        X, A, P = X.to(device), A.to(device), P.to(device)
        Pm = P if model.n_params else None
        Mm = P[:, 0] if model.mass_aware else None
        out = model.latent_forward(X, A, Pm, M=Mm)
        pred = model.physical_rollout(X, A, out.s_now, out.s_pred, P=Pm, M=Mm)  # (B,T,18)
        tgt = X[:, H:H + pred.shape[1]]
        err = torch.linalg.norm(pred[..., :3] - tgt[..., :3], dim=-1)  # (B,T)
        for h in horizons:
            sums[h] += err[:, h - 1].sum().item()
        n += X.shape[0]
    return {h: sums[h] / n for h in horizons}


def train_one(n_params, train_dl, val_dl, raw_states, raw_actions, raw_params,
              device, s1_steps, s2_steps, lambda_sig, mass_aware=False):
    model = SkyJEPA(history=10, horizon=20, action_mode="ctbr", pos_mode="relative",
                    prober_hidden=40, n_params=n_params,
                    mass_aware=mass_aware).to(device)
    fit_actions = raw_actions
    if mass_aware:
        # z-score buffers must match the hover-normalized convention (E4)
        fit_actions = raw_actions.copy()
        fit_actions[..., 0] /= raw_params[:, 0:1] * 9.81
    model.fit_normalization(torch.from_numpy(raw_states), torch.from_numpy(fit_actions))
    if n_params:
        model.fit_param_normalization(torch.from_numpy(raw_params))
    tag = "mass" if mass_aware else (f"cond({n_params})" if n_params else "uncond")
    np_tot = sum(p.numel() for p in model.parameters())
    print(f"\n===== TRAIN {tag}  ({np_tot} params) =====", flush=True)

    # Stage 1: encoders + predictor (+ param_enc, which feeds the predictor)
    s1 = (list(model.state_encoder.parameters()) + list(model.action_encoder.parameters())
          + list(model.predictor.parameters()))
    if n_params:
        s1 += list(model.param_enc.parameters())
    opt = torch.optim.Adam(s1, lr=0.0, weight_decay=1e-5)
    model.train(); step = 0
    while step < s1_steps:
        for X, A, P in train_dl:
            if step >= s1_steps: break
            X, A, P = X.to(device), A.to(device), P.to(device)
            set_lr(opt, lr_at(step, s1_steps))
            out = model.latent_forward(X, A, P if n_params else None,
                                       M=P[:, 0] if mass_aware else None)
            loss, _ = latent_loss(out, lambda_sig=lambda_sig)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(s1, 0.5); opt.step()
            if step % max(1, s1_steps // 16) == 0:
                print(f"  [{tag} s1 {step:5d}/{s1_steps}] loss={loss.item():.4f}", flush=True)
            step += 1
    print(f"  [{tag}] stage1 done (loss {loss.item():.4f})", flush=True)

    # Stage 2: prober (+ param_enc, which also feeds the prober)
    for p in model.parameters(): p.requires_grad_(False)
    s2 = list(model.prober.parameters())
    for p in model.prober.parameters(): p.requires_grad_(True)
    if n_params:
        for p in model.param_enc.parameters(): p.requires_grad_(True)
        s2 += list(model.param_enc.parameters())
    opt = torch.optim.Adam(s2, lr=0.0, weight_decay=1e-5)
    model.train(); step = 0; H, T = model.H, model.T
    while step < s2_steps:
        for X, A, P in train_dl:
            if step >= s2_steps: break
            X, A, P = X.to(device), A.to(device), P.to(device)
            set_lr(opt, lr_at(step, s2_steps))
            Pm = P if n_params else None
            Mm = P[:, 0] if mass_aware else None
            with torch.no_grad():
                out = model.latent_forward(X, A, Pm, M=Mm)
            pred = model.physical_rollout(X, A, out.s_now, out.s_pred, P=Pm, M=Mm)
            loss, comps = physical_loss(pred, X[:, H:H + T], std=model.state_std)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(s2, 0.5); opt.step()
            if step % max(1, s2_steps // 16) == 0:
                print(f"  [{tag} s2 {step:5d}/{s2_steps}] loss={loss.item():.4f} "
                      f"pos={comps['pos']:.4f}", flush=True)
            step += 1
    print(f"  [{tag}] stage2 done (loss {loss.item():.4f})", flush=True)

    rmse = openloop_pos_rmse(model, val_dl, device)
    print(f"  [{tag}] HELD-OUT open-loop pos RMSE: "
          + "  ".join(f"{h*0.05:.2f}s={v:.3f}m" for h, v in rmse.items()), flush=True)
    cfg = {"history": 10, "horizon": 20, "width_mult": 1, "prober_hidden": 40,
           "action_mode": "ctbr", "probabilistic": False, "learn_thrust": False,
           "pos_mode": "relative", "n_params": n_params, "mass_aware": mass_aware}
    return model, rmse, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="artifacts/racing_pc.bin")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--s1", type=int, default=8000)
    ap.add_argument("--s2", type=int, default=8000)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--lambda-sig", type=float, default=0.02)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--save-prefix", default=None,
                    help="save each arm's checkpoint to <prefix>_<arm>.pt")
    ap.add_argument("--arms", default="uncond,cond",
                    help="comma list of arms to train: uncond, cond, mass")
    args = ap.parse_args()

    states, actions8, _ = load_bin(args.bin)
    actions = actions8[..., 4:8]  # ctbr [thrust, wx, wy, wz]
    params = load_params(args.bin + ".params")
    n_traj = states.shape[0]
    print(f"{args.bin}: {n_traj} traj, params {params.shape}, "
          f"mass range [{params[:,0].min():.2f},{params[:,0].max():.2f}]")

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n_traj, generator=g).tolist()
    n_val = int(n_traj * args.val_frac)
    val_trajs, train_trajs = perm[:n_val], perm[n_val:]  # HELD-OUT drones
    print(f"train drones {len(train_trajs)}  held-out drones {len(val_trajs)}")

    mk = lambda trajs, stride: PWindows(states, actions, params, trajs, 10, 20, stride)
    train_dl = DataLoader(mk(train_trajs, args.stride), batch_size=256, shuffle=True, drop_last=True)
    val_dl = DataLoader(mk(val_trajs, 20), batch_size=512)
    print(f"train windows {len(train_dl.dataset)}  val windows {len(val_dl.dataset)}", flush=True)

    arm_spec = {"uncond": (0, False), "cond": (10, False), "mass": (0, True)}
    res = {}
    for arm in args.arms.split(","):
        npar, maware = arm_spec[arm.strip()]
        torch.manual_seed(0)  # same init across arms
        model, res[arm], cfg = train_one(npar, train_dl, val_dl, states, actions,
                                         params, args.device, args.s1, args.s2,
                                         args.lambda_sig, mass_aware=maware)
        if args.save_prefix:
            path = f"{args.save_prefix}_{arm.strip()}.pt"
            torch.save({"model": model.state_dict(), "config": cfg}, path)
            print(f"  saved {path}", flush=True)
    print("\n===== RESULT (held-out drones, open-loop pos RMSE) =====")
    for h in (5, 10, 20):
        line = f"  {h*0.05:.2f}s: " + "  ".join(
            f"{arm}={v[h]:.3f}m" for arm, v in res.items())
        print(line)


if __name__ == "__main__":
    main()
