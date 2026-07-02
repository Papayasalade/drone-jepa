"""Differentiable Kinematic Integrator (DKI) — zero learned parameters.

Given the current physical state, the action (4 rotor forces) and the prober's
residuals, advance the rigid-body state one step. The nominal model is just
thrust + gravity; everything the nominal model gets wrong (true mass, drag,
motor lag, inertia, rotor allocation) is absorbed by the learned residuals
  - dvdot : residual linear acceleration   (R^3, world frame)
  - K     : residual angular-accel map      (R^{3x4}); omega_dot = K @ a
Attitude is advanced on SO(3) with the matrix exponential (Eqs. 15-16).

State packing follows drone_jepa.state (pos, vel, R-flat row-major, omega-body).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..state import GRAVITY, OMEGA, POS, ROT, VEL
from .so3 import exp_so3, project_so3


class DKI(nn.Module):
    def __init__(self, dt: float = 0.05, m_nominal: float = 1.0,
                 g: float = GRAVITY, reproject: bool = False,
                 action_mode: str = "rotor_force"):
        super().__init__()
        self.dt = dt
        self.m_nominal = m_nominal
        self.g = g
        self.reproject = reproject
        self.action_mode = action_mode  # "rotor_force" or "ctbr"

    def step(self, x: torch.Tensor, a: torch.Tensor,
             dvdot: torch.Tensor, ang: torch.Tensor,
             thrust_scale: torch.Tensor | None = None) -> torch.Tensor:
        """One semi-implicit Euler step on SE(3) x velocities.

        x:     (B, 18) current physical state
        a:     (B, 4)  action — rotor forces, or [thrust, wx, wy, wz] for CTBR
        dvdot: (B, 3)  residual linear accel (world frame)
        ang:   rotor_force: (B,3,4) residual angular-accel map K (omega_dot=K a)
               ctbr:        (B,3)   angular acceleration omega_dot directly
        returns (B, 18) next physical state
        """
        B = x.shape[0]
        dt = self.dt
        p = x[:, POS]
        v = x[:, VEL]
        R = x[:, ROT].reshape(B, 3, 3)
        omega = x[:, OMEGA]

        # --- translational: nominal thrust+gravity + residual ---
        if self.action_mode == "ctbr":
            thrust = a[:, :1]                         # collective thrust command
        else:
            thrust = a.sum(dim=-1, keepdim=True)      # sum of rotor forces
        body_z_world = R[..., :, 2]                   # third column = body z in world
        g_world = torch.zeros_like(v)
        g_world[:, 2] = -self.g
        # (Fix A) learned thrust-scale replaces the fixed nominal inverse-mass, so
        # the exploited thrust->accel mapping varies across ensemble members.
        inv_mass = (thrust_scale if thrust_scale is not None else 1.0 / self.m_nominal)
        vdot = g_world + (thrust * inv_mass) * body_z_world + dvdot

        # --- rotational ---
        if self.action_mode == "ctbr":
            # prober predicts omega_dot directly (captures the inner rate loop
            # driving omega toward the commanded body rates a[:,1:4]).
            omegadot = ang                            # (B,3)
        else:
            omegadot = (ang @ a.unsqueeze(-1)).squeeze(-1)  # K @ a  (B,3)

        v_next = v + vdot * dt
        p_next = p + v_next * dt
        omega_next = omega + omegadot * dt
        R_next = R @ exp_so3(omega * dt)              # body-frame omega
        # exp_so3 keeps R on SO(3) to ~1e-15 over T~20 steps, so reprojection is
        # off by default (SVD of an exact rotation has repeated singular values
        # (1,1,1) and can fail to converge in LAPACK).
        if self.reproject:
            R_next = project_so3(R_next)

        out = torch.empty_like(x)
        out[:, POS] = p_next
        out[:, VEL] = v_next
        out[:, ROT] = R_next.reshape(B, 9)
        out[:, OMEGA] = omega_next
        return out
