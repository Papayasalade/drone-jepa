"""Direct autoregressive next-state baseline (the thing SkyJEPA beats).

Matched-capacity control for the central claim: predicting the next *physical
state* autoregressively compounds error over long horizons, whereas SkyJEPA
predicts the next *latent* and decodes once per step. Same encoders and same GRU
size as SkyJEPA, but the GRU readout is the physical-state delta (18-dim), fed
back at each step. No SO(3) structure, no DKI — which is precisely why error
accumulates faster.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..state import STATE_DIM
from .encoders import ActionEncoder, StateEncoder
from .predictor import HIDDEN_DIM


class DirectBaseline(nn.Module):
    def __init__(self, history: int = 10, horizon: int = 20):
        super().__init__()
        self.H = history
        self.T = horizon
        self.state_encoder = StateEncoder()      # history -> initial context
        self.action_encoder = ActionEncoder()
        self.emb = nn.Linear(STATE_DIM, 16)      # per-step physical-state embed
        self.cell = nn.GRUCell(16 + 8, HIDDEN_DIM)
        self.h0 = nn.Linear(16, HIDDEN_DIM)
        self.readout = nn.Linear(HIDDEN_DIM, STATE_DIM)
        nn.init.zeros_(self.readout.bias)
        nn.init.normal_(self.readout.weight, std=1e-3)  # start near identity dynamics
        self.register_buffer("state_mean", torch.zeros(STATE_DIM))
        self.register_buffer("state_std", torch.ones(STATE_DIM))
        self.register_buffer("action_mean", torch.zeros(4))
        self.register_buffer("action_std", torch.ones(4))

    @torch.no_grad()
    def fit_normalization(self, states, actions):
        s = states.reshape(-1, STATE_DIM)
        a = actions.reshape(-1, 4)
        self.state_mean.copy_(s.mean(0)); self.state_std.copy_(s.std(0).clamp_min(1e-4))
        self.action_mean.copy_(a.mean(0)); self.action_std.copy_(a.std(0).clamp_min(1e-4))

    def rollout(self, X: torch.Tensor, A: torch.Tensor,
                horizon: int | None = None) -> torch.Tensor:
        """Predict physical states for t+1..t+L autoregressively. (B,L,18).

        horizon L defaults to the training T; may be larger to probe compounding
        (the GRU unrolls arbitrarily, feeding its own predictions back).
        """
        H = self.H
        T = horizon if horizon is not None else self.T
        Xn = (X - self.state_mean) / self.state_std
        An = (A - self.action_mean) / self.action_std
        s0 = self.state_encoder(Xn)[:, H - 1]      # (B,16) history context
        Z = self.action_encoder(An)                # (B,L,8)
        h = torch.tanh(self.h0(s0))
        xn = Xn[:, H - 1]                          # normalized current state
        preds = []
        for k in range(T):
            inp = torch.cat([self.emb(xn), Z[:, H - 1 + k]], dim=-1)
            h = self.cell(inp, h)
            xn = xn + self.readout(h)              # residual step in normalized space
            preds.append(xn)
        pred_n = torch.stack(preds, dim=1)         # (B,T,18) normalized
        return pred_n * self.state_std + self.state_mean

    @torch.no_grad()
    def predict(self, state_hist: torch.Tensor, action_window: torch.Tensor,
                horizon: int | None = None) -> torch.Tensor:
        """SkyJEPA-compatible interface so MPPI / the UI can drive the baseline.

        Only the first H states of state_hist are used (it rolls out
        autoregressively), so passing the H-step history is sufficient.
        """
        return self.rollout(state_hist, action_window, horizon)
