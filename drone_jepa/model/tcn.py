"""Causal Temporal Convolutional Network (TCN) — the encoder backbone.

SkyJEPA uses two small TCNs to ingest fixed-length history windows:
  - state encoder  Enc_theta : channels [8, 8, 16]  -> state latent  s_t (R^16)
  - action encoder Enc_phi   : channels [4, 4,  8]  -> action latent z_t (R^8)

Spec gaps the paper leaves open (see NOTES.md): kernel size and dilations.
We use kernel_size=3 with exponentially increasing dilations (1, 2, 4, ...),
the standard TCN recipe, giving a receptive field that comfortably covers the
H=10 history window with three layers.

Convolutions are *causal*: output at time t depends only on inputs <= t, so the
same module can be applied to a whole sequence and read out per-step latents
(needed for the per-step action latents the predictor consumes during rollout).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1D conv with left padding only, so output[t] sees input[<= t]."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, L)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Causal conv -> GELU -> LayerNorm (over channels), with residual."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(out_ch)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        y = self.act(self.conv(x))
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)  # LN over channels
        return y + self.res(x)


class TCN(nn.Module):
    """Stack of causal blocks. Returns the full per-step sequence (B, L, C_out)."""

    def __init__(self, in_ch: int, channels: Sequence[int], kernel_size: int = 3):
        super().__init__()
        blocks = []
        c_prev = in_ch
        for i, c in enumerate(channels):
            blocks.append(TCNBlock(c_prev, c, kernel_size, dilation=2**i))
            c_prev = c
        self.blocks = nn.Sequential(*blocks)
        self.out_ch = c_prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, in_ch)  ->  (B, L, out_ch).  Per-step causal latents."""
        x = x.transpose(1, 2)  # (B, in_ch, L)
        x = self.blocks(x)
        return x.transpose(1, 2)  # (B, L, out_ch)
