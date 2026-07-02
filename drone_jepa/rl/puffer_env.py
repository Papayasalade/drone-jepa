"""PufferLib native-env wrapper around the vectorized rotor-rs racing env.

Presents the N batched drones as N PufferLib "agents" sharing flat buffers, so
PufferLib's PPO (pufferl) trains directly on the fast Rust env — no per-step Python
overhead. Action = CTBR in [-1,1]^4; obs is the 21-dim translation-invariant vector.
"""
from __future__ import annotations

import gymnasium
import numpy as np
import pufferlib

from .rust_env import RustVecEnv


class DroneRaceEnv(pufferlib.PufferEnv):
    def __init__(self, num_envs: int = 4096, seed: int = 0, buf=None):
        self.rust = RustVecEnv(num_envs, seed)
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.rust.obs_dim,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(self.rust.act_dim,), dtype=np.float32)
        self.num_agents = num_envs
        # PufferLib's vector wrapper normally sets this; we pass the env straight to
        # PuffeRL, so define it ourselves (only read by the LSTM/recurrent path).
        self.agents_per_batch = num_envs
        self._ret = np.zeros(num_envs, np.float64)  # running episode return per env
        self._len = np.zeros(num_envs, np.int64)
        super().__init__(buf)

    def reset(self, seed=None):
        self.observations[:] = self.rust.reset()
        self._ret[:] = 0.0
        self._len[:] = 0
        return self.observations, []

    def step(self, actions):
        a = np.asarray(actions, dtype=np.float32).reshape(self.num_agents, self.rust.act_dim)
        obs, rew, done = self.rust.step(np.clip(a, -1.0, 1.0))
        self._ret += rew
        self._len += 1
        d = done.astype(bool)
        infos = []
        for i in np.nonzero(d)[0]:
            infos.append({"episode_return": float(self._ret[i]), "episode_length": int(self._len[i])})
        self._ret[d] = 0.0
        self._len[d] = 0
        self.observations[:] = obs
        self.rewards[:] = rew
        self.terminals[:] = d
        self.truncations[:] = False  # timeouts folded into `done`
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def close(self):
        self.rust.close()


if __name__ == "__main__":
    env = DroneRaceEnv(num_envs=64, seed=0)
    obs, _ = env.reset()
    print(f"PufferEnv OK: num_agents={env.num_agents} obs{tuple(env.observations.shape)} "
          f"act{tuple(env.single_action_space.shape)}")
    acts = np.random.default_rng(0).standard_normal((64, env.rust.act_dim), dtype=np.float32) * 0.3
    o, r, t, tr, info = env.step(acts)
    print(f"step OK: obs{tuple(o.shape)} reward[mean={r.mean():+.3f}] done={int(t.sum())}/64")
