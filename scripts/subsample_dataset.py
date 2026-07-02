"""Select a deterministic subset of trajectories from a .pt trajectory dataset.

    python scripts/subsample_dataset.py IN.pt OUT.pt 500 123
"""

from __future__ import annotations

import sys

import torch


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit("usage: subsample_dataset.py IN.pt OUT.pt N SEED")
    src, out, n_s, seed_s = sys.argv[1:]
    n = int(n_s)
    seed = int(seed_s)
    data = torch.load(src, weights_only=False)
    states = data["states"]
    actions = data["actions"]
    if n > states.shape[0]:
        raise SystemExit(f"requested {n} trajectories but {src} only has {states.shape[0]}")
    g = torch.Generator().manual_seed(seed)
    ix = torch.randperm(states.shape[0], generator=g)[:n]
    subset = {
        "states": states[ix].contiguous(),
        "actions": actions[ix].contiguous(),
        "domain": torch.arange(n, dtype=torch.long),
        "dt": data.get("dt", 0.05),
    }
    torch.save(subset, out)
    print(f"saved {out}: {tuple(subset['states'].shape)} from {src}")


if __name__ == "__main__":
    main()
