"""Test the two uncertainty-aware-planning options (a) and (b).

(a) Fully-independent ENSEMBLE: does the disagreement penalty tame the iCEM
    divergence that the cheap (shared-stage-1) ensemble could not?
(b) PROBABILISTIC model: is the predicted variance calibrated to the actual
    error, and does a variance penalty help iCEM?
"""
import numpy as np, torch
from drone_jepa.model.jepa import SkyJEPA
from drone_jepa.model.ensemble import EnsembleJEPA
from drone_jepa.eval.closed_loop import fly
from drone_jepa.eval.openloop import _test_trajectories, _long_windows
from drone_jepa.data_gen.sim import sample_params, random_initial_state
from drone_jepa.state import POS

SEEDS = [1, 3, 5]
import functools
print = functools.partial(__builtins__.print if hasattr(__builtins__, "print")
                          else __import__("builtins").print, flush=True)


def domains():
    out = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        out.append((sample_params(rng),
                    random_initial_state(rng, center=np.array([0., 0., 1.5]))))
    return out


def sweep(model, lams, label):
    doms = domains()
    print(f"\n{label} — iCEM tracking RMSE vs penalty λ:")
    print("  λ        " + "  ".join(f"seed{s}" for s in SEEDS) + "   mean")
    for lam in lams:
        rs = [fly(model, "ctbr", p, i, ki=0.1, sampler="icem", unc_lambda=lam)
              for p, i in doms]
        print(f"  {lam:<7.2f}  " + "  ".join(f"{r:5.2f}" for r in rs) + f"  {np.mean(rs):.2f}")


def main():
    # ---------- (a) fully-independent ensemble ----------
    paths = ["artifacts/skyjepa_ctbr.pt", "artifacts/skyjepa_ctbr_indep11.pt",
             "artifacts/skyjepa_ctbr_indep12.pt", "artifacts/skyjepa_ctbr_indep13.pt"]
    ens = EnsembleJEPA(paths)
    print(f"(a) fully-independent ensemble of {len(ens.members)} CTBR JEPAs")
    sweep(ens, [0.0, 10.0, 40.0], "(a) ensemble disagreement penalty")

    # ---------- (b) probabilistic model ----------
    pm, _ = SkyJEPA.from_checkpoint("artifacts/skyjepa_ctbr_prob.pt")
    # calibration: predicted sigma vs actual error, binned, on held-out data
    st, ac = _test_trajectories("artifacts/dataset_ctbr.pt")
    X, A = _long_windows(st, ac, 10, 20, stride=20); X, A = X[:1500], A[:1500]
    mean, unc = pm.predict_with_unc(X[:, :10], A, horizon=20)        # (B,20),(B,20)
    err = (mean - X[:, 10:30]).pow(2).sum(-1).sqrt()                 # (B,20) norm err
    u, e = unc.flatten().numpy(), err.flatten().numpy()
    print("\n(b) probabilistic calibration — predicted σ vs actual error:")
    qs = np.quantile(u, [0, .25, .5, .75, 1.0])
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (u >= lo) & (u <= hi)
        print(f"  σ∈[{lo:.2f},{hi:.2f}]: predicted σ={u[m].mean():.3f}  actual err={e[m].mean():.3f} m")
    print(f"  corr(predicted σ, actual error) = {np.corrcoef(u, e)[0,1]:.3f}")
    sweep(pm, [0.0, 5.0, 20.0], "(b) probabilistic variance penalty")


if __name__ == "__main__":
    main()
