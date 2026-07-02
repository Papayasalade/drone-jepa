"""Latent dynamics predictor — a single-layer GRU, hidden dim 24.

s_bar_{t+1} = Pred(s_t, z_t), unrolled recursively over the horizon.

Spec note (see NOTES.md): the paper gives "single-layer GRU, hidden 24" and the
inferred latent dims 16 (state) + 8 (action) = 24. We read that literally: the
per-step GRU *input* is the concatenation [s_t (16) ; z_t (8)] = 24, the hidden
state is 24, and a linear readout maps hidden -> next state latent (16). The
predicted latent is fed back in as s_t for the next step, so the rollout is
genuinely recurrent (error can compound — which is exactly what we measure).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import ACTION_LATENT_DIM, STATE_LATENT_DIM

HIDDEN_DIM = 24


class Predictor(nn.Module):
    def __init__(self, state_latent: int = STATE_LATENT_DIM,
                 action_latent: int = ACTION_LATENT_DIM,
                 hidden: int | None = None, param_dim: int = 0) -> None:
        super().__init__()
        self.state_latent = state_latent
        self.param_dim = param_dim
        hidden = hidden if hidden is not None else state_latent + action_latent
        # param_dim>0: condition the latent rollout on a drone-param embedding so the
        # predicted latent dynamics are drone-aware (not the "average" drone).
        self.cell = nn.GRUCell(state_latent + action_latent + param_dim, hidden)
        self.readout = nn.Linear(hidden, state_latent)
        self.h0 = nn.Linear(state_latent, hidden)

    def forward(self, s0: torch.Tensor, z_seq: torch.Tensor,
                p_lat: torch.Tensor | None = None) -> torch.Tensor:
        """Unroll the predictor.

        s0:    (B, 16)         current state latent (last history step).
        z_seq: (B, T, 8)       per-step action latents for the future horizon.
        p_lat: (B, P) or None  drone-param embedding (when param_dim>0).
        returns (B, T, 16)     predicted future state latents s_bar_{t+1..t+T}.
        """
        B, T, _ = z_seq.shape
        s = s0
        h = torch.tanh(self.h0(s0))
        out = []
        for k in range(T):
            parts = [s, z_seq[:, k]]
            if self.param_dim > 0 and p_lat is not None:
                parts.append(p_lat)
            h = self.cell(torch.cat(parts, dim=-1), h)
            s = self.readout(h)
            out.append(s)
        return torch.stack(out, dim=1)
