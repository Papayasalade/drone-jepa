"""Physics-inspired prober psi — a 3-layer MLP that feeds the DKI.

Maps a (frozen) state latent s_bar AND the current action to the DKI residuals:
  dvdot in R^3        residual linear acceleration
  K     in R^{3x4}    residual angular-acceleration allocation (omega_dot = K a)

Inputs (spec-gap fills, see NOTES.md): the prober reads three things —
  - the latent s        : identifies the *domain* (mass/drag) from history, which
                          a single state cannot reveal;
  - the running state x  : the DKI's *own* integrated state, so the residual can
                          correct open-loop drift (without it the prober is blind
                          to drift and the double-integration explodes) and can
                          express state-dependent effects like drag(velocity);
  - the action a         : the residual is thrust-dependent
                          ((1/m_nom-1/m_true)*T*R e3), so dvdot must scale with it.

Spec gap: the paper gives depth=3 but not the hidden width; we use 40. The final
layer starts small so the prober begins near the nominal thrust+gravity model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..state import ACTION_DIM, STATE_DIM
from .encoders import STATE_LATENT_DIM

DVDOT_DIM = 3
K_DIM = 3 * 4


class Prober(nn.Module):
    def __init__(self, state_latent: int = STATE_LATENT_DIM, hidden: int = 40,
                 action_mode: str = "rotor_force", probabilistic: bool = False,
                 learn_thrust: bool = False, param_dim: int = 0,
                 inputs: str = "full") -> None:
        super().__init__()
        self.action_mode = action_mode
        self.probabilistic = probabilistic
        self.learn_thrust = learn_thrust   # (Fix A) learn effective thrust-scale
        self.param_dim = param_dim         # drone-param embedding width (0 = off)
        # What the prober is allowed to see (the decisive design choice, NOTES.md):
        #   "latent"        — the paper-literal psi(latent): action-blind AND blind
        #                     to the DKI's own open-loop drift.
        #   "latent_action" — can model thrust-scaled residuals, still drift-blind.
        #   "full"          — (latent, running state, action): a closed-loop
        #                     corrector; the only variant that decodes stably here.
        assert inputs in ("full", "latent_action", "latent")
        self.inputs = inputs
        # rotor_force: dvdot(3) + K(3x4); ctbr: dvdot(3) + omega_dot(3) [+ thrust_logscale(1)]
        self.ang_dim = K_DIM if action_mode == "rotor_force" else 3
        out_dim = DVDOT_DIM + self.ang_dim + (1 if learn_thrust else 0)
        prober_in = state_latent + param_dim
        if inputs == "full":
            prober_in += STATE_DIM + ACTION_DIM
        elif inputs == "latent_action":
            prober_in += ACTION_DIM
        self.net = nn.Sequential(
            nn.Linear(prober_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        # start near the nominal model: residuals ~ 0
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        if probabilistic:
            # (b) variance head: per-state-dim log-variance of the next state
            # (in normalized units), trained with a Gaussian NLL -> aleatoric
            # uncertainty calibrated to the model's actual prediction error.
            self.var_head = nn.Sequential(
                nn.Linear(prober_in, hidden), nn.GELU(),
                nn.Linear(hidden, STATE_DIM))
            nn.init.zeros_(self.var_head[-1].bias)  # start at logvar=0 -> var=1

    def forward(self, s: torch.Tensor, x_norm: torch.Tensor, a_norm: torch.Tensor,
                p_lat: torch.Tensor | None = None):
        """-> (dvdot, ang, logvar, thrust_scale).

        ang is K (...,3,4) for rotor_force, or omega_dot (...,3) for ctbr.
        logvar (...,18) if probabilistic else None.
        thrust_scale (...,1) multiplicative thrust correction (≈1) if learn_thrust
        else None — (Fix A) makes the exploited thrust→accel term member-varying.
        p_lat (...,P): drone-param embedding (param_dim>0) — tells the physical
        decode the drone's mass/inertia directly instead of inferring from history.
        """
        if self.inputs == "full":
            parts = [s, x_norm, a_norm]
        elif self.inputs == "latent_action":
            parts = [s, a_norm]
        else:  # "latent" — the paper-literal reading
            parts = [s]
        if self.param_dim > 0 and p_lat is not None:
            parts.append(p_lat)
        feat = torch.cat(parts, dim=-1)
        out = self.net(feat)
        dvdot = out[..., :DVDOT_DIM]
        rest = out[..., DVDOT_DIM:]
        thrust_scale = None
        if self.learn_thrust:
            thrust_scale = torch.exp(rest[..., -1:].clamp(-2.0, 2.0))  # ≈1 at init
            rest = rest[..., :-1]
        ang = rest.reshape(*out.shape[:-1], 3, 4) if self.action_mode == "rotor_force" else rest
        logvar = self.var_head(feat).clamp(-8.0, 4.0) if self.probabilistic else None
        return dvdot, ang, logvar, thrust_scale
