"""PPO-train the ~9.5K-param racing policy on the vectorized rotor-rs env with
PufferLib. Saves the actor weights for export to a WASM RlPolicy drone.

  python -m drone_jepa.rl.train_puffer --steps 30_000_000 --envs 4096
"""
from __future__ import annotations

import argparse

import torch

import pufferlib.pufferl as pufferl
import pufferlib.models
from .policy_net import Policy
from .puffer_env import DroneRaceEnv


def make_config(steps, device, use_rnn=False):
    # full [train] config (PuffeRL indexes these directly)
    return dict(
        env="drone_race", name="drone_race", project="drone_jepa",
        use_rnn=use_rnn,
        seed=42, torch_deterministic=True, cpu_offload=False, device=device,
        # continuous-control PPO settings (PufferLib's defaults are tuned for
        # discrete arcade envs — lr=0.015/muon diverges here).
        optimizer="adam", anneal_lr=True, precision="float32",
        total_timesteps=int(steps), learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        update_epochs=4, clip_coef=0.2, vf_coef=0.5, vf_clip_coef=0.2,
        max_grad_norm=0.5, ent_coef=0.0, adam_beta1=0.9, adam_beta2=0.999, adam_eps=1e-8,
        data_dir="artifacts/rl", checkpoint_interval=1_000_000,
        batch_size="auto", minibatch_size=8192, max_minibatch_size=32768, bptt_horizon=32,
        compile=False, compile_mode="default", compile_fullgraph=False,
        vtrace_rho_clip=1.0, vtrace_c_clip=1.0, prio_alpha=0.8, prio_beta0=0.2,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, default=30_000_000)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="artifacts/skyrl_ctbr.pt")
    ap.add_argument("--recurrent", action="store_true",
                    help="LSTM policy (memory) instead of the reactive MLP — tests "
                         "whether feedback+memory does in-context drone system-ID")
    ap.add_argument("--hidden", type=int, default=85)
    args = ap.parse_args()

    env = DroneRaceEnv(num_envs=args.envs, seed=0)
    if args.recurrent:
        # ~9.5K actor: single-layer encoder (h=32) + LSTM(32,32) carries the memory.
        inner = Policy(env, hidden_size=args.hidden, encoder_layers=1)
        policy = pufferlib.models.LSTMWrapper(env, inner, input_size=args.hidden,
                                              hidden_size=args.hidden).to(args.device)
        actor = (sum(p.numel() for p in inner.encoder.parameters())
                 + sum(p.numel() for p in policy.lstm.parameters())
                 + sum(p.numel() for p in inner.decoder_mean.parameters()))
        total = sum(p.numel() for p in policy.parameters())
        print(f"policy: RECURRENT actor(enc+lstm+mean)={actor} total={total} "
              f"h={args.hidden} | {args.envs} envs", flush=True)
    else:
        policy = Policy(env, hidden_size=args.hidden).to(args.device)
        actor, total = policy.param_counts()
        print(f"policy: reactive actor(deployed)={actor} total={total}  | {args.envs} envs")

    config = make_config(args.steps, args.device, use_rnn=args.recurrent)
    trainer = pufferl.PuffeRL(config, env, policy, logger=pufferl.NoLogger(config))

    def _scalar(x):  # recurrent (use_rnn) returns list-valued stats; reactive scalar
        if isinstance(x, (list, tuple)):
            return sum(float(v) for v in x) / len(x) if x else float("nan")
        return float(x)

    it = 0
    while trainer.global_step < config["total_timesteps"]:
        stats = trainer.evaluate()
        trainer.train()
        it += 1
        if it % 10 == 0 and isinstance(stats, dict) and "episode_return" in stats:
            r = _scalar(stats["episode_return"])
            ln = _scalar(stats.get("episode_length", 0.0))
            print(f"[iter {it}] step={trainer.global_step:>10,}  ep_return={r:7.2f}  ep_len={ln:5.0f}",
                  flush=True)
    for _ in range(8):
        trainer.evaluate()

    torch.save({"model": policy.state_dict(),
                "config": {"hidden": args.hidden, "obs_dim": env.rust.obs_dim,
                           "act_dim": env.rust.act_dim, "action_mode": "ctbr",
                           "recurrent": args.recurrent}}, args.out)
    print(f"saved {args.out}  (global_step={trainer.global_step})")


if __name__ == "__main__":
    main()
