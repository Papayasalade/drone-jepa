"""Collect a domain-randomized dataset of closed-loop trajectories.

Logs (state, action) at 20 Hz over many trajectories spread across distinct
randomized domains (Table I), saved as a single .pt file:
  states  : float32 (N_traj, n_steps, 18)
  actions : float32 (N_traj, n_steps, 4)
  domain  : int64   (N_traj,)            which domain each traj belongs to

Defaults are small for a quick laptop run; pass --domains/--per-domain to scale
toward the paper's 500 domains x 20000 trajectories.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .sim import DRRanges, OOD_RANGES, sample_params, simulate_trajectory


def collect(n_domains: int, per_domain: int, t_final: float, dt: float,
            seed: int, max_retries: int = 5, ranges: DRRanges = None,
            action_noise_max: float = 0.0, action_mode: str = "rotor_force"):
    rng = np.random.default_rng(seed)
    ranges = ranges or DRRanges()
    states, actions, domains = [], [], []
    n_invalid = 0
    t0 = time.time()
    total = n_domains * per_domain
    for d in range(n_domains):
        params = sample_params(rng, ranges)  # one domain = one param set
        for _ in range(per_domain):
            # per-trajectory exploration noise level (half clean, half perturbed)
            noise = rng.uniform(0.0, action_noise_max) if action_noise_max > 0 else 0.0
            for attempt in range(max_retries):
                res = simulate_trajectory(rng, params=params, t_final=t_final,
                                          dt=dt, ranges=ranges, action_noise=noise,
                                          action_mode=action_mode)
                if res.is_valid():
                    break
                noise *= 0.5  # back off noise if it destabilized this domain
                n_invalid += 1
            states.append(res.states.astype(np.float32))
            actions.append(res.actions.astype(np.float32))
            domains.append(d)
        done = (d + 1) * per_domain
        if d % max(1, n_domains // 20) == 0 or done == total:
            print(f"  {done}/{total} trajs  ({n_invalid} invalid retried)  "
                  f"{time.time()-t0:.1f}s", flush=True)
    return {
        "states": torch.from_numpy(np.stack(states)),
        "actions": torch.from_numpy(np.stack(actions)),
        "domain": torch.tensor(domains, dtype=torch.long),
        "dt": dt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", type=int, default=100)
    ap.add_argument("--per-domain", type=int, default=20)
    ap.add_argument("--t-final", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ranges", choices=["train", "ood"], default="train")
    ap.add_argument("--action-mode", choices=["rotor_force", "ctbr"],
                    default="rotor_force")
    ap.add_argument("--action-noise-max", type=float, default=0.0,
                    help="max per-traj action exploration noise std [N] (broadens "
                         "state/action coverage for closed-loop control)")
    ap.add_argument("--out", type=str, default="artifacts/dataset.pt")
    args = ap.parse_args()

    ranges = OOD_RANGES if args.ranges == "ood" else DRRanges()
    print(f"collecting {args.domains} domains x {args.per_domain} trajs "
          f"({args.t_final}s @ {1/args.dt:.0f}Hz, ranges={args.ranges}, "
          f"action_noise_max={args.action_noise_max})...")
    data = collect(args.domains, args.per_domain, args.t_final, args.dt,
                   args.seed, ranges=ranges, action_noise_max=args.action_noise_max,
                   action_mode=args.action_mode)
    torch.save(data, args.out)
    print(f"saved {args.out}: states {tuple(data['states'].shape)}, "
          f"actions {tuple(data['actions'].shape)}")


if __name__ == "__main__":
    main()
