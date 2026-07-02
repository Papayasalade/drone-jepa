"""A simple classical cascaded PID controller (no model, no learning).

Baseline for the evaluator race. Outer loop: PID on position -> desired
acceleration -> desired thrust vector. Inner mapping: collective thrust +
body-rate command from the attitude error (proportional), flown through
RotorPy's cmd_ctbr inner rate loop. Uses a NOMINAL mass (it does not know the
domain-randomized true mass) — the integral term absorbs that mismatch, exactly
like a real flight controller. This is the "fake PID" contestant.
"""

from __future__ import annotations

import numpy as np

from ..state import POS, VEL, ROT


class PIDController:
    def __init__(self, m_nominal: float = 0.5, g: float = 9.81, dt: float = 0.05,
                 kp_pos=(6.0, 6.0, 9.0), kd_pos=(4.0, 4.0, 5.0), ki_pos=1.5,
                 kp_att: float = 9.0, rate_max: float = 12.0, i_max: float = 1.0):
        self.m = m_nominal; self.g = g; self.dt = dt
        self.kp_pos = np.asarray(kp_pos); self.kd_pos = np.asarray(kd_pos)
        self.ki_pos = ki_pos; self.kp_att = kp_att
        self.rate_max = rate_max; self.i_max = i_max
        self.integral = np.zeros(3)

    def reset(self):
        self.integral = np.zeros(3)

    def act(self, x_vec, p_ref, v_ref):
        """x_vec: (18,) state; p_ref,v_ref: (3,). Returns [thrust, wx, wy, wz]."""
        p, v = x_vec[POS], x_vec[VEL]
        R = x_vec[ROT].reshape(3, 3)
        e_p, e_v = p_ref - p, v_ref - v
        self.integral = np.clip(self.integral + e_p * self.dt, -self.i_max, self.i_max)
        a_cmd = self.kp_pos * e_p + self.kd_pos * e_v + self.ki_pos * self.integral
        f_world = self.m * (a_cmd + np.array([0.0, 0.0, self.g]))  # desired force
        b3 = R[:, 2]                                   # current body-z in world
        thrust = max(0.0, float(f_world @ b3))         # project onto body-z
        b3_des = f_world / (np.linalg.norm(f_world) + 1e-6)
        err_world = np.cross(b3, b3_des)               # axis to rotate b3 -> b3_des
        w_cmd = self.kp_att * (R.T @ err_world)        # body-frame rate command
        w_cmd = np.clip(w_cmd, -self.rate_max, self.rate_max)
        return np.concatenate([[thrust], w_cmd])
