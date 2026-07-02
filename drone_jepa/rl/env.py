"""Gymnasium tracking environment around the RotorPy dynamics.

Same task as the MPPI race: a domain-randomized quadrotor tracks a circular
reference by commanding the 4 rotor forces at 20 Hz. This lets a PPO policy be
trained end-to-end and then raced head-to-head against the MPC+JEPA controllers
in the evaluator (apples-to-apples: same dynamics, same action space, same task).

Observation (translation-invariant — only the error to the reference matters):
  velocity(3) + rotation matrix(9) + body rates(3) + reference-error preview
  (relative position to the next PREVIEW reference points, 3 each).
Action: 4 rotor forces, mapped from [-1, 1] -> [0, f_max].
Reward: peaked tracking bonus - control effort; episode ends on crash.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rotorpy.vehicles.multirotor import Multirotor
from scipy.spatial.transform import Rotation

from ..data_gen.sim import sample_params, rotor_force_limit, DRRanges

PREVIEW = 3          # number of future reference points in the observation
DT = 0.05
CENTER = np.array([0.0, 0.0, 1.5])


def circle_ref(t, phase, radius, freq, center=CENTER):
    w = 2 * np.pi * freq
    p = center + np.array([radius * np.cos(w * t + phase) - radius * np.cos(phase),
                           radius * np.sin(w * t + phase) - radius * np.sin(phase),
                           0.0])
    v = np.array([-radius * w * np.sin(w * t + phase),
                  radius * w * np.cos(w * t + phase), 0.0])
    return p, v


class DroneTrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_s: float = 8.0, randomize: bool = True):
        super().__init__()
        self.n_steps = int(episode_s / DT)
        self.randomize = randomize
        obs_dim = 3 + 9 + 3 + 3 * PREVIEW
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self._rng = np.random.default_rng()

    # ---- helpers ----
    def _obs(self):
        p = self.state["x"]
        R = Rotation.from_quat(self.state["q"]).as_matrix().reshape(9)
        prev = []
        for k in range(1, PREVIEW + 1):
            pr, _ = circle_ref((self.i + k) * DT, *self.ref)
            prev.append((pr - p) / 3.0)               # scaled relative error
        return np.concatenate([self.state["v"] / 5.0, R, self.state["w"] / 5.0,
                               *prev]).astype(np.float32)

    # ---- gym API ----
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        self.params = sample_params(rng) if self.randomize else sample_params(np.random.default_rng(0))
        self.f_max = rotor_force_limit(self.params)
        self.vehicle = Multirotor(self.params, control_abstraction="cmd_motor_thrusts", aero=True)
        # random circle reference
        self.ref = (rng.uniform(0, 2 * np.pi), rng.uniform(0.8, 1.6),
                    rng.uniform(0.1, 0.2))
        p0, v0 = circle_ref(0.0, *self.ref)
        att = Rotation.from_euler("xyz", rng.uniform(-0.1, 0.1, 3))
        self.state = {"x": p0 + rng.uniform(-0.2, 0.2, 3), "v": v0 + rng.uniform(-0.2, 0.2, 3),
                      "q": att.as_quat(), "w": rng.uniform(-0.1, 0.1, 3),
                      "wind": np.zeros(3), "rotor_speeds": np.full(4, 500.0)}
        self.i = 0
        return self._obs(), {}

    def step(self, action):
        force = (np.asarray(action, float) + 1.0) * 0.5 * self.f_max   # [-1,1]->[0,fmax]
        self.state = self.vehicle.step(self.state, {"cmd_motor_thrusts": force}, DT)
        self.i += 1
        p = self.state["x"]
        p_ref, v_ref = circle_ref(self.i * DT, *self.ref)
        perr = np.linalg.norm(p - p_ref)
        verr = np.linalg.norm(self.state["v"] - v_ref)
        finite = np.isfinite(p).all()
        crashed = (not finite) or perr > 6.0 or p[2] < 0.2
        reward = (np.exp(-(perr ** 2) / 0.5) + 0.2 * np.exp(-(verr ** 2) / 4.0)
                  - 5e-4 * float(np.sum(force ** 2)))
        if crashed:
            reward -= 5.0
        terminated = bool(crashed)
        truncated = self.i >= self.n_steps
        obs = self._obs() if finite else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(reward), terminated, truncated, {}
