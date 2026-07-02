"""Finer metric than fleet_fly (which saturates): held-out open-loop position RMSE
for each capacity-sweep model, on unseen wide drones. Tells us whether more params
actually buy better PREDICTION on the fleet, even when all models still fail to fly.

  .venv/bin/python -m drone_jepa.eval_openloop_widths
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model.jepa import SkyJEPA
from .train_pc import PWindows, load_bin, load_params, openloop_pos_rmse


def main():
    states, actions8, _ = load_bin("artifacts/racing_pc.bin")
    actions = actions8[..., 4:8]
    params = load_params("artifacts/racing_pc.bin.params")
    n_traj = states.shape[0]
    # SAME held-out split as train_pc (seed 0, 12%) → genuinely unseen drones
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n_traj, generator=g).tolist()
    val_trajs = perm[:int(n_traj * 0.12)]
    val_dl = DataLoader(PWindows(states, actions, params, val_trajs, 10, 20, 20),
                        batch_size=512)
    print(f"held-out drones: {len(val_trajs)}  windows: {len(val_dl.dataset)}\n")

    models = [
        ("9.5K  (v2clean)", "artifacts/skyjepa_ctbr_v2clean.pt"),
        ("14.2K (w1.25)",   "artifacts/exp_w125.pt"),
        ("19.8K (w1.5)",    "artifacts/exp_w15.pt"),
        ("33.8K (w2)",      "artifacts/exp_w2.pt"),
    ]
    print(f"{'model':<22} {'params':>8}   open-loop pos RMSE (held-out)")
    print(f"{'':<22} {'':>8}   0.25s     0.50s     1.00s")
    for name, path in models:
        try:
            model, _ = SkyJEPA.from_checkpoint(path, device="cpu")
        except FileNotFoundError:
            print(f"{name:<22} (missing {path})")
            continue
        npar = sum(p.numel() for p in model.parameters())
        rmse = openloop_pos_rmse(model, val_dl, "cpu", horizons=(5, 10, 20))
        print(f"{name:<22} {npar:>8}   {rmse[5]:.3f}m    {rmse[10]:.3f}m    {rmse[20]:.3f}m",
              flush=True)


if __name__ == "__main__":
    main()
