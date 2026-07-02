"""ctypes binding to the vectorized rotor-rs racing env (web-demo/rl-env).

Presents the batched env as plain numpy arrays so it can be wrapped for PufferLib
/ CleanRL PPO. One `step` advances all N drones in a single Rust call (auto-reset
on episode end), which is the whole point — orders of magnitude faster than the
pure-Python RotorPy env.
"""
from __future__ import annotations

import ctypes
import platform
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_EXT = {"Darwin": "dylib", "Linux": "so", "Windows": "dll"}[platform.system()]
_LIB = _ROOT / "web-demo" / "rl-env" / "target" / "release" / f"librl_env.{_EXT}"


class RustVecEnv:
    """Vectorized gate-racing env. obs (N, obs_dim) f32; actions (N, act_dim) in [-1,1]."""

    def __init__(self, num_envs: int, seed: int = 0):
        if not _LIB.exists():
            raise FileNotFoundError(
                f"{_LIB} not found — build it: "
                f"cd web-demo/rl-env && cargo build --release")
        self.lib = ctypes.CDLL(str(_LIB))
        self.lib.rlenv_obs_dim.restype = ctypes.c_size_t
        self.lib.rlenv_act_dim.restype = ctypes.c_size_t
        self.lib.rlenv_create.restype = ctypes.c_void_p
        self.lib.rlenv_create.argtypes = [ctypes.c_size_t, ctypes.c_uint64]
        P = ctypes.POINTER(ctypes.c_float)
        self.lib.rlenv_reset.argtypes = [ctypes.c_void_p, P]
        self.lib.rlenv_step.argtypes = [ctypes.c_void_p, P, P, P, P]
        self.lib.rlenv_free.argtypes = [ctypes.c_void_p]

        self.num_envs = num_envs
        self.obs_dim = self.lib.rlenv_obs_dim()
        self.act_dim = self.lib.rlenv_act_dim()
        self.handle = self.lib.rlenv_create(num_envs, seed)
        # persistent contiguous buffers (no per-step alloc)
        self.obs = np.zeros((num_envs, self.obs_dim), dtype=np.float32)
        self.rew = np.zeros(num_envs, dtype=np.float32)
        self.done = np.zeros(num_envs, dtype=np.float32)

    def _p(self, a):
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def reset(self):
        self.lib.rlenv_reset(self.handle, self._p(self.obs))
        return self.obs

    def step(self, actions: np.ndarray):
        a = np.ascontiguousarray(actions, dtype=np.float32)
        self.lib.rlenv_step(self.handle, self._p(a), self._p(self.obs), self._p(self.rew), self._p(self.done))
        return self.obs, self.rew, self.done

    def close(self):
        if self.handle:
            self.lib.rlenv_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


if __name__ == "__main__":
    import time
    N, STEPS = 4096, 500
    env = RustVecEnv(N, seed=1)
    obs = env.reset()
    print(f"obs {obs.shape} act_dim {env.act_dim}  ({N} envs)")
    rng = np.random.default_rng(0)
    # warmup
    for _ in range(10):
        env.step(rng.standard_normal((N, env.act_dim), dtype=np.float32) * 0.3)
    t0 = time.perf_counter()
    total_r = 0.0
    for _ in range(STEPS):
        a = (rng.standard_normal((N, env.act_dim), dtype=np.float32) * 0.3).clip(-1, 1)
        obs, rew, done = env.step(a)
        total_r += rew.mean()
    dt = time.perf_counter() - t0
    sps = N * STEPS / dt
    print(f"{N*STEPS:,} steps in {dt:.2f}s  =>  {sps/1e6:.2f}M steps/sec  "
          f"(mean reward/step {total_r/STEPS:+.3f})")
