"""Ensemble of SkyJEPA models — for epistemic uncertainty (disagreement).

The members share the same encoders + predictor (stage 1) and differ only in the
prober (stage 2, different seeds), so we encode + roll the latents ONCE and run
each member's prober/DKI on the shared latents — K decodes, not K full models.

`predict_with_unc` returns the mean predicted trajectory AND the cross-member
disagreement (std of predicted position per step). Where the members agree the
dynamics are confident; where they disagree we are off-distribution — exactly
the signal uncertainty-aware planning penalizes to stop model exploitation.
"""

from __future__ import annotations

import torch

from .jepa import SkyJEPA
from ..state import POS


class EnsembleJEPA:
    def __init__(self, paths, device="cpu"):
        self.members = []
        for p in paths:
            m, cfg = SkyJEPA.from_checkpoint(p, device=device)
            self.members.append(m)
        self.cfg = cfg
        self.H = self.members[0].H
        self.T = self.members[0].T
        self.action_mode = self.members[0].action_mode

    def to(self, device):
        return self

    def eval(self):
        return self

    @torch.no_grad()
    def predict_with_unc(self, state_hist, action_window, horizon=None):
        """-> (mean (B,L,18), disagreement (B,L)). Runs each member's FULL
        predict (its own encoder/predictor/prober) — required for fully
        independent members, whose encoders disagree off-distribution."""
        preds = torch.stack([m.predict(state_hist, action_window, horizon)
                             for m in self.members])           # (K, B, T, 18)
        mean = preds.mean(0)
        disagree = preds[..., POS].std(0).norm(dim=-1)         # (B, T) positional spread
        return mean, disagree

    @torch.no_grad()
    def predict(self, state_hist, action_window, horizon=None):
        return self.predict_with_unc(state_hist, action_window, horizon)[0]
