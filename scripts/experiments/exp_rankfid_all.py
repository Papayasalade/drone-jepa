"""E1+E2: rank-fidelity + per-component error under deployment-like actions,
for ALL labeled seed-lottery checkpoints (fly/crash known from rotor_fly).

Shared ground truth: at n_states hover-ish states of the fixed hummingbird,
sample S candidate rotor-force plans the way the deployed RotorMppiController
does (hover nominal, sigma=0.22*hover_force, temporal smoothing beta=0.85,
plus a few aggressive thrust-offset plans), roll the TRUE RotorPy sim.

Per checkpoint:
  E1: spearman(predicted cost, realized cost), chosen plan's realized pctile
  E2(ii): per-component open-loop error ON THESE PLANS (pos / rot-angle / omega),
          plus inversion-miss (model fails to predict body-z flip) and the same
          restricted to the model's top-10 preferred plans
  E2(i):  per-component error on DATASET actions (shared rf_mppi test split),
          k=12 and k=40 — the branch Exp 1 proved blind.

Run:  .venv/bin/python <this file>   (from repo root)
"""
import copy, json, sys, time
import numpy as np
import torch
from rotorpy.vehicles.hummingbird_params import quad_params as BASE
from rotorpy.vehicles.multirotor import Multirotor

sys.path.insert(0, ".")
from drone_jepa.data_gen.sim import _state_to_vec, random_initial_state, rotor_force_limit  # noqa
from drone_jepa.model.jepa import SkyJEPA  # noqa
from drone_jepa.state import POS  # noqa
from drone_jepa.eval.openloop import _test_trajectories, _long_windows  # noqa

OUT_JSON = "artifacts/e12_results.json"

# (checkpoint stem, outcome label, gates/race, %flipped)  -- from blog_cliff RESULTS*
CKPTS = [
    ("blog_fullmix_w2_s0", "CRASH", 0.0, 27.4), ("blog_fullmix_w2_s1", "CRASH", 0.0, 22.5),
    ("blog_fullmix_w2_s2", "FLY", 5.0, 1.6),    ("blog_fullmix_w1_s0", "MARGINAL", 2.6, 9.5),
    ("blog_sep_i0_d0_w2", "CRASH", 0.0, 39.1),  ("blog_sep_i0_d1_w2", "CRASH", 0.0, 28.5),
    ("blog_sep_i0_d2_w2", "CRASH", 0.0, 30.5),  ("blog_sep_i0_d3_w2", "CRASH", 0.0, 35.6),
    ("blog_sep_i1_d0_w2", "CRASH", 0.0, 17.8),  ("blog_sep_i2_d0_w2", "FLY", 4.9, 1.5),
    ("blog_sep_i3_d0_w2", "TIMID", 0.6, 0.9),
    ("blog_def_s3_w2", "MARGINAL", 2.1, 7.2),   ("blog_def_s4_w2", "CRASH", 0.0, 3.7),
    ("blog_ortho_s0_w2", "CRASH", 0.0, 18.4),   ("blog_ortho_s1_w2", "CRASH", 0.0, 19.5),
    ("blog_litinit_s0_w2", "CRASH", 0.0, 13.5), ("blog_litinit_s1_w2", "MARGINAL", 0.2, 5.0),
    ("blog_litinit_s2_w2", "CRASH", 0.0, 16.4), ("blog_litinit_s3_w2", "FLY", 4.7, 3.5),
    ("blog_loc_crashS1_p1", "CRASH", 0.0, 30.0), ("blog_loc_crashS1_p2", "CRASH", 0.0, 28.2),
    ("blog_loc_crashS1_p3", "CRASH", 0.0, 35.7), ("blog_loc_flyS1_p1", "FLY", 4.7, 1.3),
    ("blog_loc_flyS1_p2", "FLY", 4.9, 1.5),      ("blog_loc_flyS1_p3", "FLY", 4.8, 1.7),
    # controls trained on other data (base / small-mix / rec sweep)
    ("blog_base_w2_s0", "FLY.base", 4.7, 0.9),  ("blog_base_w2_s1", "FLY.base", 4.4, 1.0),
    ("blog_base_w2_s2", "FLY.base", 4.5, 0.8),
    ("blog_small_w2_s1", "FLY.small", 4.6, 2.3), ("blog_small_w2_s2", "FLY.small", 4.9, 2.3),
    ("blog_rec800_w2", "FLY.rec", 4.8, 1.4),
]

N_STATES, S, T = 8, 96, 12
DT = 0.05
BETA = 0.85          # deployed temporal smoothing
SIG_FRAC = 0.22      # deployed sigma_force = 0.22 * hover_force


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
    """geodesic angle between rotation blocks. Rp,Rt: (...,9)"""
    A = Rp.reshape(*Rp.shape[:-1], 3, 3)
    B = Rt.reshape(*Rt.shape[:-1], 3, 3)
    tr = np.einsum("...ij,...ij->...", A, B)      # trace(A^T B)
    c = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def sample_plans(rng, hover, f_max):
    """(S,T,4) rotor-force plans like the deployed sampler."""
    sig = SIG_FRAC * hover
    plans = np.zeros((S, T, 4))
    for s in range(S):
        if s < S // 8:            # aggressive thrust offsets (climb/dive regime)
            off = rng.uniform(-0.35, 0.5) * hover
        else:
            off = 0.0
        raw = rng.normal(0.0, sig, (T, 4))
        sm = np.zeros_like(raw)
        acc = raw[0]
        for k in range(T):
            acc = BETA * acc + (1 - BETA) * raw[k]
            sm[k] = acc
        sm *= sig / max(sm.std(), 1e-9)          # keep deployed variance after EMA
        plans[s] = hover + off + sm
    return np.clip(plans, 0.0, f_max)


def main():
    t0 = time.time()
    params = copy.deepcopy(BASE)
    f_max = rotor_force_limit(params)
    hover = params["mass"] * 9.81 / 4.0
    rng = np.random.default_rng(7)

    # ---------- shared ground truth ----------
    print(f"[gt] rolling {N_STATES}x{S} true-sim plans (f_max={f_max:.2f} hover={hover:.3f})", flush=True)
    states0, plans_all, real_all, targets = [], [], [], []
    for i in range(N_STATES):
        init = random_initial_state(rng, center=np.array([0.0, 0.0, 1.5]))
        x0 = _state_to_vec(init)
        plans = sample_plans(rng, hover, f_max)
        real = np.stack([true_rollout(params, init, plans[s]) for s in range(S)])  # (S,T,18)
        tgt = x0[:3] + np.array([2.0, 0.0, 3.0])   # gate-like: forward + climb
        states0.append((init, x0)); plans_all.append(plans); real_all.append(real); targets.append(tgt)
        print(f"[gt] state {i} done  t={time.time()-t0:.0f}s", flush=True)

    # dataset-action branch: shared rf_mppi test split windows
    H = 10
    K_LONG = 40
    st_te, ac_te = _test_trajectories("artifacts/dataset_rf_mppi.pt")
    Xw, Aw = _long_windows(st_te, ac_te, H, K_LONG, stride=20)
    idx = torch.randperm(Xw.shape[0], generator=torch.Generator().manual_seed(0))[:512]
    Xw, Aw = Xw[idx], Aw[idx]
    print(f"[gt] dataset windows {Xw.shape}  t={time.time()-t0:.0f}s", flush=True)

    results = {}
    for stem, label, gates, flip in CKPTS:
        try:
            model, _ = SkyJEPA.from_checkpoint(f"artifacts/{stem}.pt", device="cpu")
        except FileNotFoundError:
            print(f"[skip] {stem} missing", flush=True)
            continue
        model.eval()
        r = dict(label=label, gates=gates, flip=flip)

        # ---- E1 + E2(ii): plan branch ----
        sps, pctls, pos_e, rot_e, om_e, inv_miss, top_pos_e, top_rot_e = [], [], [], [], [], [], [], []
        for (init, x0), plans, real, tgt in zip(states0, plans_all, real_all, targets):
            hist = torch.tensor(np.tile(x0, (H, 1)), dtype=torch.float32)
            ahist = torch.tensor(np.tile(hover * np.ones(4), (H, 1)), dtype=torch.float32)
            cand = torch.tensor(plans, dtype=torch.float32)
            win = torch.zeros(S, H + T, 4)
            win[:, :H - 1] = ahist[-(H - 1):].unsqueeze(0).expand(S, -1, -1)
            win[:, H - 1:H - 1 + T] = cand
            win[:, H - 1 + T:] = cand[:, -1:]
            sh = hist.unsqueeze(0).expand(S, -1, -1)
            with torch.no_grad():
                pred = model.predict(sh, win, horizon=T).numpy()   # (S,T,18)

            pred_cost = ((pred[..., :3] - tgt) ** 2).sum(-1).mean(-1)
            real_cost = ((real[..., :3] - tgt) ** 2).sum(-1).mean(-1)
            sps.append(spearman(pred_cost, real_cost))
            chosen = int(np.argmin(pred_cost))
            pctls.append(float((real_cost < real_cost[chosen]).mean()))

            pe = np.linalg.norm(pred[..., :3] - real[..., :3], axis=-1)[:, -1]     # (S,) @T
            re = rot_angle_deg(pred[..., 6:15], real[..., 6:15])[:, -1]
            oe = np.linalg.norm(pred[..., 15:18] - real[..., 15:18], axis=-1)[:, -1]
            pos_e.append(pe.mean()); rot_e.append(re.mean()); om_e.append(oe.mean())
            # inversion-miss: true body-z flips (e33<0) but model says fine (or vice versa)
            e33_true = real[..., 14] < 0.0    # R[2,2] = index 6+8
            e33_pred = pred[..., 14] < 0.0
            inv_miss.append(float((e33_true != e33_pred).mean()))
            top = np.argsort(pred_cost)[:10]  # the plans the planner prefers
            top_pos_e.append(pe[top].mean()); top_rot_e.append(re[top].mean())

        r.update(rank_corr=float(np.mean(sps)), chosen_pctl=float(np.mean(pctls)),
                 plan_pos=float(np.mean(pos_e)), plan_rot=float(np.mean(rot_e)),
                 plan_om=float(np.mean(om_e)), inv_miss=float(np.mean(inv_miss)),
                 top_pos=float(np.mean(top_pos_e)), top_rot=float(np.mean(top_rot_e)))

        # ---- E2(i): dataset-action branch ----
        with torch.no_grad():
            predw = model.predict(Xw[:, :H], Aw, horizon=K_LONG)
        truew = Xw[:, H:H + K_LONG]
        dp = torch.linalg.norm(predw[..., :3] - truew[..., :3], dim=-1)
        dr = rot_angle_deg(predw[..., 6:15].numpy(), truew[..., 6:15].numpy())
        do = torch.linalg.norm(predw[..., 15:18] - truew[..., 15:18], dim=-1)
        for k, tag in [(11, "12"), (39, "40")]:
            r[f"ds_pos{tag}"] = float(dp[:, k].mean()); r[f"ds_rot{tag}"] = float(dr[:, k].mean())
            r[f"ds_om{tag}"] = float(do[:, k].mean())
        results[stem] = r
        print(f"[{stem:<24}] {label:<9} rank={r['rank_corr']:+.2f} pctl={r['chosen_pctl']*100:3.0f}% "
              f"plan_pos={r['plan_pos']:.2f} plan_rot={r['plan_rot']:5.1f} inv={r['inv_miss']*100:4.1f}% "
              f"ds_pos40={r['ds_pos40']:.2f} ds_rot40={r['ds_rot40']:5.1f}  t={time.time()-t0:.0f}s", flush=True)

    json.dump(results, open(OUT_JSON, "w"), indent=1)
    print("WROTE", OUT_JSON, flush=True)

    # quick separation summary
    def group(lbl):
        return [v for v in results.values() if v["label"].startswith(lbl)]
    for metric in ["rank_corr", "chosen_pctl", "plan_pos", "plan_rot", "plan_om",
                   "inv_miss", "top_rot", "ds_pos40", "ds_rot40", "ds_om40"]:
        c = [g[metric] for g in group("CRASH")]
        f = [g[metric] for g in results.values() if g["label"].startswith("FLY")]
        print(f"{metric:<12} CRASH {np.mean(c):8.3f}±{np.std(c):.3f}   FLY {np.mean(f):8.3f}±{np.std(f):.3f}", flush=True)


if __name__ == "__main__":
    main()
