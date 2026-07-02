"""Random smooth reference trajectories for closed-loop data collection.

The paper builds full-state references with a GP sampler + differential
flatness. We use the flatness-friendly equivalent: each position axis is a sum
of sinusoids, so position and all derivatives the SE(3) controller needs
(x, x_dot, x_ddot, x_dddot, x_ddddot) are available in closed form. Random
amplitudes / frequencies / phases give the diverse, persistently-exciting
maneuvers needed to identify the dynamics.

Returns RotorPy-style flat_output dicts so it plugs straight into SE3Control.
"""

from __future__ import annotations

import numpy as np


class RandomFourierTrajectory:
    def __init__(self, rng: np.random.Generator, center=(0.0, 0.0, 1.5),
                 n_modes: int = 3, amp=(0.3, 1.0), freq=(0.2, 1.0),
                 yaw_amp: float = 0.4):
        self.center = np.asarray(center, dtype=float)
        # per-axis (3) x n_modes parameters
        self.A = rng.uniform(*amp, size=(3, n_modes))
        self.w = rng.uniform(*freq, size=(3, n_modes))
        self.phi = rng.uniform(0, 2 * np.pi, size=(3, n_modes))
        # taper z so the drone does not dive into the ground
        self.A[2] *= 0.4
        self.yaw_A = rng.uniform(-yaw_amp, yaw_amp)
        self.yaw_w = rng.uniform(0.2, 0.8)
        self.yaw_phi = rng.uniform(0, 2 * np.pi)

    def update(self, t: float) -> dict:
        A, w, phi, c = self.A, self.w, self.phi, self.center
        s = np.sin(w * t + phi)
        co = np.cos(w * t + phi)
        x = c + (A * s).sum(1)
        x_dot = (A * w * co).sum(1)
        x_ddot = (-A * w**2 * s).sum(1)
        x_dddot = (-A * w**3 * co).sum(1)
        x_ddddot = (A * w**4 * s).sum(1)
        yaw = self.yaw_A * np.sin(self.yaw_w * t + self.yaw_phi)
        yaw_dot = self.yaw_A * self.yaw_w * np.cos(self.yaw_w * t + self.yaw_phi)
        return {
            "x": x, "x_dot": x_dot, "x_ddot": x_ddot,
            "x_dddot": x_dddot, "x_ddddot": x_ddddot,
            "yaw": yaw, "yaw_dot": yaw_dot,
        }
