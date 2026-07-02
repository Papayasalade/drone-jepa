"""Closed-loop demo: fly RotorPy with model-based MPPI tracking a reference.

Drops the trained SkyJEPA world model into the deployment MPPI loop and flies a
RotorPy hummingbird (held-out random domain) to track a moving position
reference. Reports position tracking RMSE and saves a plot. This exercises the
full deployment path: history -> encode -> latent rollout -> prober+DKI ->
cost -> importance-weighted action -> execute first action.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rotorpy.vehicles.multirotor import Multirotor

from drone_jepa.control import MPPIController
from drone_jepa.data_gen.sim import (_state_to_vec, random_initial_state,
                                     rotor_force_limit, sample_params)
from drone_jepa.model.jepa import SkyJEPA
from drone_jepa.state import POS


def ref_state(t, T, dt, center, radius=1.2, freq=0.15):
    """Full target states for steps t+1..t+T: a slow circle, level, near-rest."""
    out = np.zeros((T, 18))
    out[:, 6:15] = np.eye(3).reshape(9)  # level attitude
    for k in range(T):
        tk = t + (k + 1) * dt
        w = 2 * np.pi * freq
        out[k, POS] = center + np.array([radius * np.cos(w * tk) - radius,
                                         radius * np.sin(w * tk), 0.0])
        out[k, 3:6] = np.array([-radius * w * np.sin(w * tk),
                                radius * w * np.cos(w * tk), 0.0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/skyjepa.pt")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    model, cfg = SkyJEPA.from_checkpoint(args.ckpt)
    H, T = cfg["history"], cfg["horizon"]

    rng = np.random.default_rng(args.seed)
    params = sample_params(rng)
    f_max = rotor_force_limit(params)
    veh = Multirotor(params, control_abstraction="cmd_motor_thrusts", aero=True)
    dt = 0.05
    center = np.array([0.0, 0.0, 1.5])

    a_hover = params["mass"] * 9.81 / 4
    mppi = MPPIController(model, horizon=min(15, T), samples=args.samples,
                         sigma=0.6, f_max=f_max, temperature=1.0)
    # warm-start the nominal at hover thrust (not mid-range), so MPPI explores a
    # narrow band around the data distribution rather than diverging upward.
    mppi.nominal = torch.full((mppi.T, 4), a_hover)

    st = random_initial_state(rng, center=center)
    x0 = _state_to_vec(st)
    hist = torch.tensor(np.tile(x0, (H, 1)), dtype=torch.float32)        # (H,18)
    ahist = torch.full((H, 4), a_hover, dtype=torch.float32)             # (H,4)

    n = int(args.seconds / dt)
    log_pos, log_ref = [], []
    for i in range(n):
        ref = torch.tensor(ref_state(i * dt, T, dt, center), dtype=torch.float32)
        a = mppi.step(hist, ahist, ref)
        a_np = a.numpy().clip(0, f_max)
        st = veh.step(st, {"cmd_motor_thrusts": a_np}, dt)
        x = _state_to_vec(st)
        hist = torch.cat([hist[1:], torch.tensor(x, dtype=torch.float32)[None]], 0)
        ahist = torch.cat([ahist[1:], a[None]], 0)
        log_pos.append(x[POS]); log_ref.append(ref[0, POS].numpy())
        if not np.isfinite(x).all() or np.abs(x[POS] - center).max() > 30:
            print(f"diverged at step {i}"); break

    log_pos = np.array(log_pos); log_ref = np.array(log_ref)
    m = min(len(log_pos), len(log_ref))
    rmse = np.sqrt(((log_pos[:m] - log_ref[:m]) ** 2).sum(1).mean())
    print(f"domain mass={params['mass']:.2f} f_max/rotor={f_max:.1f}N")
    print(f"closed-loop position tracking RMSE: {rmse:.3f} m over {m} steps")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(log_ref[:, 0], log_ref[:, 1], "k--", label="reference")
    ax[0].plot(log_pos[:, 0], log_pos[:, 1], "C0", label="MPPI flown")
    ax[0].set(title="xy track", xlabel="x [m]", ylabel="y [m]"); ax[0].legend(); ax[0].axis("equal")
    t = np.arange(m) * dt
    ax[1].plot(t, log_ref[:m, 2], "k--", label="ref z"); ax[1].plot(t, log_pos[:m, 2], "C0", label="z")
    ax[1].set(title="altitude", xlabel="t [s]", ylabel="z [m]"); ax[1].legend()
    fig.tight_layout(); fig.savefig("artifacts/mppi_track.png", dpi=120)
    print("saved artifacts/mppi_track.png")


if __name__ == "__main__":
    main()
