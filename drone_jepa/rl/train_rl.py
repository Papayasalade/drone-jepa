"""Train a PPO policy for drone tracking (the RL baseline).

Uses Stable-Baselines3 PPO over parallel RotorPy tracking envs. SB3 is the
reliable choice for a prototype; for serious RL throughput you'd reach for
PufferLib / GPU-parallel sims (the "RL is sample-hungry, make the sim fast"
lesson). Saves the policy + a VecNormalize obs-stats file for deployment.
"""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from .env import DroneTrackingEnv


def make_env():
    return DroneTrackingEnv(episode_s=8.0, randomize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", default="artifacts/rl_policy")
    args = ap.parse_args()

    venv = SubprocVecEnv([make_env for _ in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", venv, verbose=1, n_steps=1024, batch_size=4096,
                gae_lambda=0.95, gamma=0.99, ent_coef=0.0, learning_rate=3e-4,
                n_epochs=10, policy_kwargs=dict(net_arch=[128, 128]))
    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(args.out)
    venv.save(args.out + "_vecnorm.pkl")
    print(f"saved {args.out}.zip and {args.out}_vecnorm.pkl")


if __name__ == "__main__":
    main()
