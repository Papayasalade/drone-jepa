"""State and action encoders (the two TCNs).

StateEncoder  : (B, H, 18) state history  -> (B, H, 16) per-step state latents s
ActionEncoder : (B, L,  4) action history -> (B, L,  8) per-step action latents z

Both are causal, so the latent at step k summarizes the window ending at k. The
predictor starts its rollout from s at the last history step and consumes the
per-step action latents z for each future step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..state import ACTION_DIM, STATE_DIM
from .tcn import TCN

STATE_LATENT_DIM = 16   # base widths (width_mult=1, the paper's shapes)
ACTION_LATENT_DIM = 8


class StateEncoder(nn.Module):
    def __init__(self, channels=(8, 8, 16)) -> None:
        super().__init__()
        self.tcn = TCN(STATE_DIM, channels=list(channels))
        self.out_dim = self.tcn.out_ch

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        """(B, H, 18) -> (B, H, latent). Use [..., -1, :] for the current latent."""
        return self.tcn(x_hist)


class ActionEncoder(nn.Module):
    def __init__(self, channels=(4, 4, 8)) -> None:
        super().__init__()
        self.tcn = TCN(ACTION_DIM, channels=list(channels))
        self.out_dim = self.tcn.out_ch

    def forward(self, a_seq: torch.Tensor) -> torch.Tensor:
        """(B, L, 4) -> (B, L, latent). Per-step action latents."""
        return self.tcn(a_seq)
