"""Evaluate a trained RL policy (reactive OR recurrent) on the WIDE randomized fleet,
deterministically (action = policy mean). Reports mean/median episode return + length
over many episodes on fresh drones. This is the apples-to-apples fleet metric for the
recurrent-vs-reactive question (the recurrent LSTM policy can't use the stateless Rust
fleet_fly, so we roll it out here with proper per-env hidden-state management).

  .venv/bin/python -m drone_jepa.rl.eval_fleet artifacts/skyrl_ctbr_v2.pt artifacts/skyrl_lstm.pt
"""
from __future__ import annotations

import sys

import numpy as np
import torch

import pufferlib.models
from .policy_net import Policy
from .puffer_env import DroneRaceEnv


def load_policy(path, env, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    h = cfg["hidden"]
    rec = cfg.get("recurrent", False)
    if rec:
        inner = Policy(env, hidden_size=h, encoder_layers=1)
        pol = pufferlib.models.LSTMWrapper(env, inner, input_size=h, hidden_size=h)
    else:
        pol = Policy(env, hidden_size=h)
    pol.load_state_dict(ck["model"])
    pol.to(device).eval()
    return pol, rec


@torch.no_grad()
def evaluate(path, n_envs=512, steps=4000, seed=987, device="cpu"):
    env = DroneRaceEnv(num_envs=n_envs, seed=seed)
    pol, rec = load_policy(path, env, device)
    obs, _ = env.reset()
    state = {"lstm_h": None, "lstm_c": None} if rec else None
    rets, lens = [], []
    for _ in range(steps):
        ot = torch.from_numpy(np.asarray(obs)).float().to(device)
        if rec:
            logits, _ = pol.forward_eval(ot, state)
        else:
            logits, _ = pol(ot)
        a = logits.mean.clamp(-1.0, 1.0).cpu().numpy()
        obs, rew, term, trunc, infos = env.step(a)
        for info in infos:
            rets.append(info["episode_return"])
            lens.append(info["episode_length"])
        if rec and state["lstm_h"] is not None:
            d = np.nonzero(np.asarray(term, bool))[0]
            if len(d):
                idx = torch.from_numpy(d).to(device)
                state["lstm_h"][idx] = 0.0
                state["lstm_c"][idx] = 0.0
    env.close()
    kind = "recurrent" if rec else "reactive"
    return dict(path=path, kind=kind, episodes=len(rets),
                mean_ret=float(np.mean(rets)), med_ret=float(np.median(rets)),
                mean_len=float(np.mean(lens)))


def main():
    paths = sys.argv[1:] or ["artifacts/skyrl_ctbr_v2.pt"]
    print(f"{'policy':<34} {'kind':<10} {'eps':>5} {'mean_ret':>9} {'med_ret':>9} {'mean_len':>9}")
    for p in paths:
        r = evaluate(p)
        print(f"{r['path']:<34} {r['kind']:<10} {r['episodes']:>5} "
              f"{r['mean_ret']:>9.2f} {r['med_ret']:>9.2f} {r['mean_len']:>9.1f}", flush=True)


if __name__ == "__main__":
    main()
