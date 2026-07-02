"""E3+E4 fly test: uncond vs param-conditioned vs mass-aware JEPA (all trained on
the same wide CTBR data) flown closed-loop (MPPI, circle tracking) on wide drones
sampled like the Rust gen_dataset sampler (mass 0.2-2 kg, asym arms, TWR-derived
k_eta, k_w 6-18). Reports tracking RMSE per drone per arm.

  .venv/bin/python e34_flytest.py [n_drones]   (from repo root)
"""
import copy, json, sys, time
import numpy as np
import torch
from rotorpy.vehicles.hummingbird_params import quad_params as BASE

sys.path.insert(0, ".")
from drone_jepa.model.jepa import SkyJEPA  # noqa
from drone_jepa.eval.closed_loop import fly  # noqa

SCRATCH = "artifacts"
HUM_KETA, HUM_KM = 5.57e-6, 1.36e-7
HUM_I = (3.65e-3, 3.68e-3, 7.03e-3)


def sample_drone(rng, sym=False):
    """Replicates web-demo/racer/examples/gen_dataset.rs sample_drone."""
    arms = np.full(4, rng.uniform(0.05, 0.40)) if sym else rng.uniform(0.05, 0.40, 4)
    d = arms / np.sqrt(2.0)
    rotor_pos = {
        "r1": np.array([d[0], d[0], 0.0]), "r2": np.array([d[1], -d[1], 0.0]),
        "r3": np.array([-d[2], -d[2], 0.0]), "r4": np.array([-d[3], d[3], 0.0]),
    }
    avg_arm = arms.mean()
    mass = rng.uniform(0.2, 2.0)
    rpm_max = 1500.0
    twr = rng.uniform(2.0, 4.0)
    k_eta = twr * mass * 9.81 / (4 * rpm_max ** 2)
    k_m = k_eta * (HUM_KM / HUM_KETA) * rng.uniform(0.7, 1.3)
    i_scale = (mass / 0.5) * (avg_arm / 0.17) ** 2 * rng.uniform(0.7, 1.3)
    p = copy.deepcopy(BASE)
    p.update(mass=mass, Ixx=HUM_I[0] * i_scale, Iyy=HUM_I[1] * i_scale,
             Izz=HUM_I[2] * i_scale, rotor_pos=rotor_pos,
             c_Dx=rng.uniform(0.02, 0.30), c_Dy=rng.uniform(0.02, 0.30),
             c_Dz=rng.uniform(0.05, 0.40), k_eta=k_eta, k_m=k_m,
             tau_m=rng.uniform(0.01, 0.04), k_w=rng.uniform(6.0, 18.0),
             rotor_speed_max=rpm_max)
    pvec = np.array([mass, p["Ixx"], p["Iyy"], p["Izz"], k_eta, k_m,
                     p["tau_m"], p["k_w"], p["c_Dz"], avg_arm], dtype=np.float32)
    return p, pvec


def hover_init(p):
    hov = np.sqrt(p["mass"] * 9.81 / (4 * p["k_eta"]))
    return {"x": np.array([0.0, 0.0, 1.5]), "v": np.zeros(3),
            "q": np.array([0.0, 0.0, 0.0, 1.0]), "w": np.zeros(3),
            "wind": np.zeros(3), "rotor_speeds": np.full(4, hov)}


class Wrap(torch.nn.Module):
    def __init__(self, model, P=None, M=None):
        super().__init__()
        self.m = model
        self.P, self.M = P, M
        self.H, self.T = model.H, model.T
        self.action_mode = model.action_mode

    def predict(self, sh, win, horizon=None):
        B = sh.shape[0]
        P = self.P.expand(B, -1) if self.P is not None else None
        M = self.M.expand(B) if self.M is not None else None
        return self.m.predict(sh, win, horizon, P=P, M=M)


def main():
    n_drones = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    arms = {}
    for arm in ("uncond", "cond", "mass"):
        m, _ = SkyJEPA.from_checkpoint(f"artifacts/e34_{arm}.pt", device="cpu")
        m.eval()
        arms[arm] = m

    rng = np.random.default_rng(4)
    rows = []
    t0 = time.time()
    for i in range(n_drones):
        p, pvec = sample_drone(rng)
        init = hover_init(p)
        row = {"drone_mass": float(pvec[0]), "arm_len": float(pvec[9])}
        for arm, model in arms.items():
            if arm == "cond":
                w = Wrap(model, P=torch.tensor(pvec).unsqueeze(0))
            elif arm == "mass":
                w = Wrap(model, M=torch.tensor([pvec[0]]))
            else:
                w = Wrap(model)
            rmse = fly(w, "ctbr", p, init, seconds=8.0, samples=384, smooth=0.8, ki=0.1)
            row[arm] = rmse
        rows.append(row)
        print(f"drone {i}: mass={row['drone_mass']:.2f}kg arm={row['arm_len']*100:.0f}cm  "
              + "  ".join(f"{a}={row[a]:.2f}m" for a in arms)
              + f"  t={time.time()-t0:.0f}s", flush=True)

    json.dump(rows, open(f"{SCRATCH}/e34_fly.json", "w"), indent=1)
    print("\nmedian RMSE: " + "  ".join(
        f"{a}={np.median([r[a] for r in rows]):.2f}m" for a in arms))
    print("diverged(>90m): " + "  ".join(
        f"{a}={sum(r[a] > 90 for r in rows)}" for a in arms))


if __name__ == "__main__":
    main()
