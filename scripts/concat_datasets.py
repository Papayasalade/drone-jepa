"""Concatenate multiple trajectory datasets (same steps) into one, re-indexing
domains so train/val split stays per-trajectory. Used to mix the gentle SE3 and
aggressive MPPI sets into one varied dataset.

    .venv/bin/python scripts/concat_datasets.py OUT.pt IN1.pt IN2.pt ...
"""

from __future__ import annotations

import sys

import numpy as np
import torch


def main() -> None:
    out = sys.argv[1]
    ins = sys.argv[2:]
    states, actions = [], []
    n = 0
    for p in ins:
        d = torch.load(p, weights_only=False)
        states.append(d["states"])
        actions.append(d["actions"])
        n += d["states"].shape[0]
        print(f"  + {p}: {tuple(d['states'].shape)}")
    states = torch.cat(states, 0)
    actions = torch.cat(actions, 0)
    data = {
        "states": states,
        "actions": actions,
        "domain": torch.arange(states.shape[0], dtype=torch.long),
        "dt": 0.05,
    }
    torch.save(data, out)
    sp = np.linalg.norm(states.numpy()[..., 3:6], axis=-1)
    print(f"saved {out}: states {tuple(states.shape)}  |v| p50={np.percentile(sp,50):.1f} "
          f"p95={np.percentile(sp,95):.1f} max={sp.max():.1f}")


if __name__ == "__main__":
    main()
