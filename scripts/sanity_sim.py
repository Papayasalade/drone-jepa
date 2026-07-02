"""Milestone 1: simulate a few trajectories and sanity-plot them."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from drone_jepa.data_gen import simulate_trajectory
from drone_jepa.data_gen.sim import rotor_force_limit
from drone_jepa.state import OMEGA, POS, VEL

rng = np.random.default_rng(0)
os.makedirs("artifacts", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
n_traj = 4
for j in range(n_traj):
    res = simulate_trajectory(rng, t_final=10.0, dt=0.05)
    valid = res.is_valid()
    t = np.arange(res.states.shape[0]) * 0.05
    pos = res.states[:, POS]
    vel = res.states[:, VEL]
    R = res.states[:, 6:15].reshape(-1, 3, 3)
    # orthonormality check: ||R^T R - I||
    err = np.linalg.norm(np.einsum("nij,nik->njk", R, R) - np.eye(3), axis=(1, 2))
    print(f"traj {j}: mass={res.params['mass']:.2f} k_eta={res.params['k_eta']:.2e} "
          f"f_max={rotor_force_limit(res.params):.2f}N  "
          f"pos_range={pos.max(0)-pos.min(0)}  max|v|={np.abs(vel).max():.2f}  "
          f"max R-orthonorm err={err.max():.2e}  "
          f"force[min,max]=[{res.actions.min():.2f},{res.actions.max():.2f}]  valid={valid}")
    axes[0, 0].plot(t, pos[:, 0], alpha=0.7)
    axes[0, 1].plot(t, pos[:, 2], alpha=0.7)
    axes[1, 0].plot(t, np.linalg.norm(vel, axis=1), alpha=0.7)
    axes[1, 1].plot(t, res.actions, alpha=0.4)

axes[0, 0].set(title="x position", xlabel="t [s]", ylabel="m")
axes[0, 1].set(title="z position", xlabel="t [s]", ylabel="m")
axes[1, 0].set(title="speed |v|", xlabel="t [s]", ylabel="m/s")
axes[1, 1].set(title="rotor forces", xlabel="t [s]", ylabel="N")
fig.tight_layout()
fig.savefig("artifacts/sanity_trajectories.png", dpi=110)
print("saved artifacts/sanity_trajectories.png")
