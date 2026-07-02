"""Deployment wrapper for the trained PPO policy.

Rebuilds the training observation (velocity + rotation matrix + body rates +
reference-error preview), applies the saved VecNormalize stats, and runs the
policy — so the RL controller can be raced in the evaluator next to the MPC+JEPA
controllers. Unlike those, it needs no MPPI: one network forward pass per step.
"""

from __future__ import annotations

import numpy as np

from .env import PREVIEW
from ..state import POS, VEL, ROT, OMEGA


class RLController:
    def __init__(self, policy_path: str, vecnorm_path: str, f_max: float = 1.0):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
        self.model = PPO.load(policy_path, device="cpu")
        vn = VecNormalize.load(vecnorm_path, venv=None) if False else None
        # load only the obs running-mean/var (no env needed)
        import pickle
        with open(vecnorm_path, "rb") as f:
            vec = pickle.load(f)
        self.obs_mean = vec.obs_rms.mean.astype(np.float32)
        self.obs_var = vec.obs_rms.var.astype(np.float32)
        self.clip = vec.clip_obs
        self.f_max = f_max

    def _obs(self, x_vec, ref_points):
        """x_vec: (18,) physical state; ref_points: (PREVIEW,3) future world refs."""
        p = x_vec[POS]
        prev = np.concatenate([(rp - p) / 3.0 for rp in ref_points[:PREVIEW]])
        return np.concatenate([x_vec[VEL] / 5.0, x_vec[ROT], x_vec[OMEGA] / 5.0,
                               prev]).astype(np.float32)

    def act(self, x_vec, ref_points):
        obs = self._obs(x_vec, ref_points)
        norm = np.clip((obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8),
                       -self.clip, self.clip)
        action, _ = self.model.predict(norm, deterministic=True)
        return (np.asarray(action, float) + 1.0) * 0.5 * self.f_max  # -> rotor forces
