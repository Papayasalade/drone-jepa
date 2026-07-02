"""Profile a dataset's coverage so we can judge whether it's varied enough:
distributions of speed, tilt, body-rate, and action, plus the workspace extent.
Pass multiple files to compare (e.g. old gentle vs new racing).

    .venv/bin/python scripts/dataset_coverage.py artifacts/dataset_ctbr.pt artifacts/dataset_racing_ctbr.pt
"""

from __future__ import annotations

import sys

import numpy as np
import torch


def pct(a):
    a = np.asarray(a).ravel()
    return f"p50={np.percentile(a,50):6.2f} p95={np.percentile(a,95):6.2f} p99={np.percentile(a,99):6.2f} max={a.max():6.2f}"


def profile(path: str) -> None:
    d = torch.load(path, weights_only=False)
    s = d["states"].numpy()      # (N, T, 18)
    a = d["actions"].numpy()     # (N, T, 4)
    n = s.shape[0] * s.shape[1]

    speed = np.linalg.norm(s[..., 3:6], axis=-1)
    # tilt = angle between body-z and world-z = acos(R[2][2]); R[2][2] is state idx 14
    tilt = np.degrees(np.arccos(np.clip(s[..., 14], -1.0, 1.0)))
    bodyrate = np.linalg.norm(s[..., 15:18], axis=-1)
    pos = s[..., 0:3]
    accel = np.linalg.norm(np.diff(s[..., 3:6], axis=1), axis=-1) / 0.05  # |dv/dt|

    print(f"\n=== {path}  ({s.shape[0]} traj x {s.shape[1]} steps = {n} samples) ===")
    print(f"  speed     [m/s]   {pct(speed)}")
    print(f"  tilt      [deg]   {pct(tilt)}")
    print(f"  body-rate [rad/s] {pct(bodyrate)}")
    print(f"  |accel|   [m/s^2] {pct(accel)}")
    print(f"  action[0] (thrust/f0) {pct(a[..., 0])}")
    print(f"  action[1:] mag        {pct(np.linalg.norm(a[..., 1:], axis=-1))}")
    print(f"  workspace x/y/z extent: "
          f"x[{pos[...,0].min():.1f},{pos[...,0].max():.1f}] "
          f"y[{pos[...,1].min():.1f},{pos[...,1].max():.1f}] "
          f"z[{pos[...,2].min():.1f},{pos[...,2].max():.1f}]")


def main():
    paths = sys.argv[1:] or ["artifacts/dataset_racing_ctbr.pt"]
    for p in paths:
        try:
            profile(p)
        except FileNotFoundError:
            print(f"(skip {p}: not found)")


if __name__ == "__main__":
    main()
