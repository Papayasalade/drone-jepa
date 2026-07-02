"""Convert a Rust dataset binary into the .pt format drone_jepa.train expects.

Supports the dual-action SE3 format (header: n_traj, steps, n_act=8) which packs
BOTH action labels for identical flights: cols 0:4 = rotor forces, cols 4:8 =
CTBR [thrust, wx, wy, wz]. Writes two datasets sharing the same states:
  <stem>_rotor.pt  (action = 4 rotor forces)
  <stem>_ctbr.pt   (action = [thrust, body rates])

Older single-action binaries (header: n_traj, steps, n_act) write one <stem>.pt.

    .venv/bin/python scripts/convert_dataset.py artifacts/se3_dual.bin artifacts/dataset_se3
"""

from __future__ import annotations

import struct
import sys

import numpy as np
import torch


def save(states: np.ndarray, actions: np.ndarray, path: str) -> None:
    n = states.shape[0]
    data = {
        "states": torch.from_numpy(states.copy()),
        "actions": torch.from_numpy(actions.copy()),
        "domain": torch.arange(n, dtype=torch.long),
        "dt": 0.05,
    }
    torch.save(data, path)
    sp = np.linalg.norm(states[..., 3:6], axis=-1)
    print(f"  saved {path}: states {states.shape}, actions {actions.shape}  |v| p95={np.percentile(sp,95):.1f}")


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "artifacts/se3_dual.bin"
    stem = sys.argv[2] if len(sys.argv) > 2 else "artifacts/dataset_se3"

    with open(src, "rb") as f:
        buf = f.read()
    n_traj, steps, n_act = struct.unpack_from("<III", buf, 0)
    off = 12
    n_s = n_traj * steps * 18
    states = np.frombuffer(buf, dtype="<f4", count=n_s, offset=off).reshape(n_traj, steps, 18)
    off += n_s * 4
    n_a = n_traj * steps * n_act
    actions = np.frombuffer(buf, dtype="<f4", count=n_a, offset=off).reshape(n_traj, steps, n_act)

    finite = np.isfinite(states).all() and np.isfinite(actions).all()
    print(f"{src}: {n_traj} traj x {steps} steps, n_act={n_act}, finite={finite}")
    if n_act == 8:
        save(states, actions[..., 0:4], f"{stem}_rotor.pt")
        save(states, actions[..., 4:8], f"{stem}_ctbr.pt")
    else:
        save(states, actions, f"{stem}.pt")


if __name__ == "__main__":
    main()
