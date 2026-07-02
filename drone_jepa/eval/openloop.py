"""Open-loop evaluation: compounding error of SkyJEPA vs the direct baseline.

Reproduces the qualitative claim of Fig. 6 / Table III: predicting in latent
space and decoding with the physics prober (SkyJEPA) grows long-horizon error
more slowly than direct autoregressive next-state prediction. The decisive test
is the *compounding* regime — we evaluate at a horizon LONGER than the T=20 the
models were trained on (both the GRU predictor and the DKI unroll arbitrarily),
where free-form autoregression diverges but the physics-structured DKI stays
bounded. Exact RMSE won't match the paper (different sim); the trend is the target.
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..model.baseline import DirectBaseline
from ..model.jepa import SkyJEPA
from ..state import OMEGA, POS, ROT, VEL


def _test_trajectories(path, test_frac=0.1, all_traj=False):
    """Reproduce load_dataset's by-trajectory split and return test tensors.

    all_traj=True returns every trajectory (used for a dedicated OOD file that
    the models never trained on, so there is no train/test split to honor)."""
    data = torch.load(path, weights_only=False)
    states, actions = data["states"], data["actions"]
    if all_traj:
        return states, actions
    N = states.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    n_test = int(N * test_frac)
    test_i = perm[:n_test]
    return states[test_i], actions[test_i]


def _long_windows(states, actions, H, L, stride):
    """Slice (B, H+L) windows across the test trajectories."""
    N, S = states.shape[0], states.shape[1]
    Xs, As = [], []
    for n in range(N):
        for s in range(0, S - (H + L) + 1, stride):
            Xs.append(states[n, s:s + H + L])
            As.append(actions[n, s:s + H + L])
    return torch.stack(Xs), torch.stack(As)


@torch.no_grad()
def evaluate(jepa_ckpt, base_ckpt, data, eval_horizon=40, stride=20,
             device="cpu", batch=256, all_traj=False):
    device = torch.device(device)
    jepa, cfg = SkyJEPA.from_checkpoint(jepa_ckpt, device=device)
    H, Ttrain = cfg["history"], cfg["horizon"]
    base = DirectBaseline(history=H, horizon=Ttrain).to(device)
    base.load_state_dict(torch.load(base_ckpt, weights_only=False)["model"]); base.eval()

    states, actions = _test_trajectories(data, all_traj=all_traj)
    X, A = _long_windows(states, actions, H, eval_horizon, stride)
    print(f"eval windows: {X.shape[0]}  horizon={eval_horizon} (trained T={Ttrain})")

    acc = {"jepa": None, "base": None}
    n = 0
    for i in range(0, X.shape[0], batch):
        Xb, Ab = X[i:i + batch].to(device), A[i:i + batch].to(device)
        target = Xb[:, H:H + eval_horizon]
        preds = {
            "jepa": jepa.predict(Xb[:, :H], Ab, horizon=eval_horizon),
            "base": base.rollout(Xb, Ab, horizon=eval_horizon),
        }
        for k, p in preds.items():
            sq = (p - target).pow(2).sum(dim=0).cpu().numpy()  # (L,18)
            acc[k] = sq if acc[k] is None else acc[k] + sq
        n += Xb.shape[0]

    def finalize(s):
        def rmse(sl):
            return np.sqrt(s[:, sl].sum(1) / (n * (sl.stop - sl.start)))
        return {"pos": rmse(POS), "vel": rmse(VEL), "rot": rmse(ROT),
                "omega": rmse(OMEGA), "full": np.sqrt(s.sum(1) / (n * 18))}

    return {k: finalize(v) for k, v in acc.items()}, eval_horizon, Ttrain


def report(res, L, Ttrain, out_png="artifacts/compounding.png"):
    h = np.arange(1, L + 1)
    print("\n=== Open-loop position RMSE [m] vs horizon step ===")
    print(f"{'k':>4} {'jepa':>10} {'baseline':>10}   (train horizon T={Ttrain})")
    marks = sorted(set([1, Ttrain, Ttrain + 1, L // 2, L]))
    for k in marks:
        if k <= L:
            print(f"{k:>4} {res['jepa']['pos'][k-1]:>10.4f} {res['base']['pos'][k-1]:>10.4f}")

    def ratio(r):
        return r["pos"][-1] / max(r["pos"][0], 1e-9)
    cj, cb = ratio(res["jepa"]), ratio(res["base"])
    print(f"\nRMSE(L)/RMSE(1):  jepa={cj:.2f}  baseline={cb:.2f}  "
          "(confounded: jepa's 1-step error is much smaller)")
    # The faithful 'compounding' measure is error growth in the EXTRAPOLATION
    # regime (past the training horizon), which is not confounded by 1-step error.
    gj = gb = None
    if L > Ttrain:
        gj = res["jepa"]["pos"][-1] / max(res["jepa"]["pos"][Ttrain-1], 1e-9)
        gb = res["base"]["pos"][-1] / max(res["base"]["pos"][Ttrain-1], 1e-9)
        print(f"growth past train horizon  RMSE(L)/RMSE(T):  jepa={gj:.2f}  baseline={gb:.2f}")
    final_ratio = res["base"]["pos"][-1] / max(res["jepa"]["pos"][-1], 1e-9)
    print(f"final-horizon position RMSE (baseline/jepa): {final_ratio:.2f}x  "
          f"(>1 = jepa better)")
    # verdict: jepa wins if it compounds slower past the train horizon AND/OR
    # has lower absolute long-horizon error.
    slower = (gj is not None and gj < gb)
    lower = final_ratio >= 1.0
    verdict = "PASS" if (slower or lower) else "FAIL"
    print(f"CLAIM (jepa compounds slower / lower long-horizon error): {verdict}"
          + (f"  [slower={slower}, lower_abs={lower}]"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for r, lab, c in [(res["jepa"], "SkyJEPA (latent+DKI)", "C0"),
                      (res["base"], "direct autoregressive", "C3")]:
        ax[0].plot(h, r["pos"], c, label=lab, marker="o", ms=2.5)
        ax[1].plot(h, r["full"], c, label=lab, marker="o", ms=2.5)
    for a, title, yl in [(ax[0], "position RMSE vs horizon", "RMSE [m]"),
                         (ax[1], "full-state RMSE vs horizon", "RMSE")]:
        a.axvline(Ttrain, color="gray", ls=":", lw=1, label=f"train horizon T={Ttrain}")
        a.set(title=title, xlabel="horizon step k", ylabel=yl)
        a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa", default="artifacts/skyjepa.pt")
    ap.add_argument("--baseline", default="artifacts/baseline.pt")
    ap.add_argument("--data", default="artifacts/dataset.pt")
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--all-traj", action="store_true",
                    help="use every trajectory (for a dedicated OOD file)")
    ap.add_argument("--out-png", default="artifacts/compounding.png")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--probe", action="store_true",
                    help="also run the deployment-action plan probe (the metric that "
                         "actually predicts closed-loop flight; dataset-action RMSE is "
                         "blind to it — see docs/EXPERIMENTS_fragility_campaign.md)")
    args = ap.parse_args()
    res, L, Ttrain = evaluate(args.jepa, args.baseline, args.data,
                              eval_horizon=args.horizon, device=args.device,
                              all_traj=args.all_traj)
    report(res, L, Ttrain, out_png=args.out_png)
    if args.probe:
        from .probe import FLY_THRESHOLD_DEG, probe_checkpoint
        r = probe_checkpoint(args.jepa)
        flag = ("CRASH-SIGNATURE" if r["plan_rot"] > FLY_THRESHOLD_DEG else "fly-like")
        print("\n=== Deployment-action plan probe (rotor-force; flight-predicting) ===")
        print(f"plan_rot={r['plan_rot']:.1f} deg  inv_miss={r['inv']*100:.1f}%  "
              f"rank_corr={r['rank']:+.2f}  plan_pos={r['plan_pos']:.2f} m  ->  [{flag}]")
        print("(threshold ~40 deg; probe catches severe tumble basins, confirm the "
              "selected model with one rotor_fly)")


if __name__ == "__main__":
    main()
