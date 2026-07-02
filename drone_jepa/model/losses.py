"""Loss functions for the two training stages.

Stage 1 (Eq. 9):  L_pred = multi-step latent consistency + lambda_sig * SIGReg.
Stage 2 (Eq. 17): supervised, component-weighted L2 on the physical state.

Per the paper, collapse in Stage 1 is prevented by SIGReg, not by a stop-grad /
EMA target encoder, so the encoded-true target keeps its gradient by default.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..state import OMEGA, POS, ROT, VEL
from .jepa import LatentOut
from .sigreg import sigreg

LAMBDA_SIG = 0.02


def latent_loss(out: LatentOut, lambda_sig: float = LAMBDA_SIG,
                stop_grad_target: bool = False):
    """Returns (total, dict of components). Stage-1 objective."""
    target = out.s_target.detach() if stop_grad_target else out.s_target
    pred_term = F.mse_loss(out.s_pred, target)
    sig_term = sigreg(out.s_all.reshape(-1, out.s_all.shape[-1]))
    total = pred_term + lambda_sig * sig_term
    return total, {"pred": pred_term.detach(), "sigreg": sig_term.detach()}


def physical_loss(pred: torch.Tensor, target: torch.Tensor,
                  std: torch.Tensor | None = None):
    """pred, target: (B, T, 18). Stage-2 objective.

    The training loss is MSE in *normalized* state space (residual divided by
    per-dim std), so position, velocity, rotation and angular velocity — which
    live on very different numeric scales — all contribute comparably without
    hand-tuned weights. The reported components are physical-unit RMSEs.
    """
    diff = pred - target
    norm_diff = diff / std if std is not None else diff
    total = norm_diff.pow(2).mean()

    def rmse(sl):
        return diff[..., sl].pow(2).mean().sqrt().detach()
    comps = {"pos": rmse(POS), "vel": rmse(VEL),
             "rot": rmse(ROT), "omega": rmse(OMEGA)}
    return total, comps


def nll_physical_loss(pred, logvar, target, std):
    """Probabilistic Stage-2 objective: Gaussian negative log-likelihood in
    normalized state space. Fits the mean AND the variance (calibrated to the
    actual prediction error). pred/logvar/target: (B,T,18)."""
    norm_diff = (pred - target) / std
    var = logvar.exp()
    nll = 0.5 * (norm_diff.pow(2) / var + logvar)
    total = nll.mean()
    diff = pred - target

    def rmse(sl):
        return diff[..., sl].pow(2).mean().sqrt().detach()
    comps = {"pos": rmse(POS), "vel": rmse(VEL), "rot": rmse(ROT),
             "omega": rmse(OMEGA),
             "sigma": var.sqrt().mean().detach()}  # avg predicted std (norm units)
    return total, comps
