"""E7: linear weight interpolation between a CRASH and a FLY checkpoint
(same data-seed d0, init-seeds 0 vs 2), full-mix w2 rotor-force.

Per alpha: val latent-pred MSE + val pos error (train-loss-family metrics),
the E1/E2 probe (rank-corr, plan rotation error, inversion-miss), checkpoint
save + jblob export for rotor_fly.

.venv/bin/python scratchpad/e7_interp.py   (from repo root)
"""
import copy, json, subprocess, sys, time
import numpy as np
import torch
from rotorpy.vehicles.hummingbird_params import quad_params as BASE
from rotorpy.vehicles.multirotor import Multirotor

sys.path.insert(0, ".")
from drone_jepa.data_gen.sim import _state_to_vec, random_initial_state, rotor_force_limit  # noqa
from drone_jepa.model.jepa import SkyJEPA  # noqa
from drone_jepa.eval.openloop import _test_trajectories, _long_windows  # noqa

SCRATCH = "artifacts"
A_CKPT = "artifacts/blog_sep_i0_d0_w2.pt"   # CRASH (0/12, 39.1% flipped)
B_CKPT = "artifacts/blog_sep_i2_d0_w2.pt"   # FLY   (11/12, 4.9 gates)
ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
N_STATES, S, T = 8, 96, 12
DT, BETA, SIG_FRAC = 0.05, 0.85, 0.22
H = 10


def true_rollout(params, st0, plan):
    veh = Multirotor(params, control_abstraction="cmd_motor_thrusts", aero=True)
    st = copy.deepcopy(st0)
    traj = []
    for a in plan:
        st = veh.step(st, {"cmd_motor_thrusts": np.asarray(a, float)}, DT)
        x = _state_to_vec(st)
        if not np.isfinite(x).all():
            break
        traj.append(x)
    while len(traj) < len(plan):
        traj.append(traj[-1] if traj else np.zeros(18))
    return np.array(traj)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def rot_angle_deg(Rp, Rt):
    A = Rp.reshape(*Rp.shape[:-1], 3, 3)
    B = Rt.reshape(*Rt.shape[:-1], 3, 3)
    tr = np.einsum("...ij,...ij->...", A, B)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def sample_plans(rng, hover, f_max):
    sig = SIG_FRAC * hover
    plans = np.zeros((S, T, 4))
    for s in range(S):
        off = rng.uniform(-0.35, 0.5) * hover if s < S // 8 else 0.0
        raw = rng.normal(0.0, sig, (T, 4))
        sm, acc = np.zeros_like(raw), raw[0]
        for k in range(T):
            acc = BETA * acc + (1 - BETA) * raw[k]
            sm[k] = acc
        sm *= sig / max(sm.std(), 1e-9)
        plans[s] = hover + off + sm
    return np.clip(plans, 0.0, f_max)


def make_gt():
    params = copy.deepcopy(BASE)
    f_max = rotor_force_limit(params)
    hover = params["mass"] * 9.81 / 4.0
    rng = np.random.default_rng(7)
    gt = []
    for _ in range(N_STATES):
        init = random_initial_state(rng, center=np.array([0.0, 0.0, 1.5]))
        x0 = _state_to_vec(init)
        plans = sample_plans(rng, hover, f_max)
        real = np.stack([true_rollout(params, init, plans[s]) for s in range(S)])
        tgt = x0[:3] + np.array([2.0, 0.0, 3.0])
        gt.append((x0, plans, real, tgt))
    return gt, hover


def probe(model, gt, hover):
    sps, rot_e, inv = [], [], []
    for x0, plans, real, tgt in gt:
        hist = torch.tensor(np.tile(x0, (H, 1)), dtype=torch.float32)
        cand = torch.tensor(plans, dtype=torch.float32)
        win = torch.zeros(S, H + T, 4)
        win[:, :H - 1] = float(hover)
        win[:, H - 1:H - 1 + T] = cand
        win[:, H - 1 + T:] = cand[:, -1:]
        with torch.no_grad():
            pred = model.predict(hist.unsqueeze(0).expand(S, -1, -1), win, horizon=T).numpy()
        pc = ((pred[..., :3] - tgt) ** 2).sum(-1).mean(-1)
        rc = ((real[..., :3] - tgt) ** 2).sum(-1).mean(-1)
        sps.append(spearman(pc, rc))
        rot_e.append(rot_angle_deg(pred[..., 6:15], real[..., 6:15])[:, -1].mean())
        inv.append(float(((real[..., 14] < 0) != (pred[..., 14] < 0)).mean()))
    return float(np.mean(sps)), float(np.mean(rot_e)), float(np.mean(inv))


def main():
    t0 = time.time()
    ckA = torch.load(A_CKPT, map_location="cpu", weights_only=False)
    ckB = torch.load(B_CKPT, map_location="cpu", weights_only=False)
    sdA, sdB, cfg = ckA["model"], ckB["model"], ckA["config"]

    gt, hover = make_gt()
    print(f"[gt] done t={time.time()-t0:.0f}s", flush=True)

    # val windows from the full-mix dataset (by-trajectory held-out split)
    st_te, ac_te = _test_trajectories("artifacts/skyjepa_rotor_mix_recovery/dataset.pt")
    Xw, Aw = _long_windows(st_te, ac_te, H, 40, stride=20)
    idx = torch.randperm(Xw.shape[0], generator=torch.Generator().manual_seed(0))[:512]
    Xw, Aw = Xw[idx], Aw[idx]

    rows = {}
    for i, a in enumerate(ALPHAS):
        sd = {k: (1 - a) * sdA[k].float() + a * sdB[k].float() for k in sdA}
        stem = f"e7_a{int(a*1000):04d}"
        torch.save({"model": sd, "config": cfg}, f"artifacts/{stem}.pt")
        model, _ = SkyJEPA.from_checkpoint(f"artifacts/{stem}.pt", device="cpu")
        model.eval()

        with torch.no_grad():
            out = model.latent_forward(Xw[:, :H + 20], Aw[:, :H + 20])
            val_pred = torch.nn.functional.mse_loss(out.s_pred, out.s_target).item()
            lat_std = out.s_target.std().item()
            predw = model.predict(Xw[:, :H], Aw, horizon=40)
        pos20 = torch.linalg.norm(predw[:, 19, :3] - Xw[:, H + 19, :3], dim=-1).mean().item()
        pos40 = torch.linalg.norm(predw[:, 39, :3] - Xw[:, H + 39, :3], dim=-1).mean().item()
        rot40 = rot_angle_deg(predw[:, 39, 6:15].numpy(), Xw[:, H + 39, 6:15].numpy()).mean()

        rank, prot, inv = probe(model, gt, hover)
        rows[stem] = dict(alpha=a, val_pred=val_pred, lat_std=lat_std, pos20=pos20,
                          pos40=pos40, rot40=float(rot40), rank=rank, plan_rot=prot, inv=inv)
        subprocess.run([".venv/bin/python", "scripts/export_jepa.py", f"artifacts/{stem}.pt", stem],
                       capture_output=True)
        subprocess.run([".venv/bin/python", "scripts/export_jepa_blob.py", stem],
                       capture_output=True)
        print(f"a={a:.3f} val_pred={val_pred:.4f} lat_std={lat_std:.2f} pos20={pos20:.3f} "
              f"pos40={pos40:.3f} rot40={rot40:.1f} rank={rank:+.2f} plan_rot={prot:.1f} "
              f"inv={inv*100:.1f}%  t={time.time()-t0:.0f}s", flush=True)

    json.dump(rows, open(f"{SCRATCH}/e7_results.json", "w"), indent=1)
    print("DONE — now run rotor_fly per blob", flush=True)


if __name__ == "__main__":
    main()
