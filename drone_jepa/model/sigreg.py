"""SIGReg — Sketched Isotropic Gaussian Regularization (anti-collapse).

This is what stops the latent dynamics from collapsing in SkyJEPA — NOT an EMA /
momentum target encoder. The idea (LeJEPA, ref [49]): an embedding cloud is
isotropic-Gaussian iff every 1-D projection of it is standard normal. So we
"sketch" the cloud onto many random directions and penalize each projection's
deviation from N(0, 1) with a characteristic-function goodness-of-fit statistic
(the Epps-Pulley family).

Spec detail (NOTES.md): the paper cites 17 spline knots. We implement the
equivalent discretized characteristic-function objective by evaluating the
empirical CF at K=17 fixed frequency knots — differentiable and collapse-proof:
a degenerate (collapsed) cloud has the wrong CF and is penalized.

The CF of N(0,1) is real: phi(t) = exp(-t^2 / 2). The empirical CF of samples y
is mean(cos(t y)) + i mean(sin(t y)). We penalize the squared complex deviation,
summed over knots and averaged over random projections.
"""

from __future__ import annotations

import torch


def sigreg(
    embeddings: torch.Tensor,
    n_projections: int = 64,
    n_knots: int = 17,
    t_max: float = 5.0,
    max_samples: int = 8192,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """embeddings: (N, D) -> scalar penalty (0 iff projections ~ N(0,1))."""
    N, D = embeddings.shape
    if N > max_samples:  # subsample rows to bound the cost of the CF estimate
        idx = torch.randperm(N, device=embeddings.device)[:max_samples]
        embeddings = embeddings[idx]
    e = embeddings - embeddings.mean(dim=0, keepdim=True)

    # random unit directions (resampled each call = the "sketch")
    V = torch.randn(D, n_projections, device=e.device, dtype=e.dtype,
                    generator=generator)
    V = V / V.norm(dim=0, keepdim=True).clamp_min(1e-8)
    proj = e @ V  # (N, P); each column is a 1-D projection of the cloud

    # frequency knots in (0, t_max]; CF is symmetric so positive t suffice
    t = torch.linspace(t_max / n_knots, t_max, n_knots,
                       device=e.device, dtype=e.dtype)  # (K,)
    ty = proj.unsqueeze(-1) * t  # (N, P, K)
    cos_mean = ty.cos().mean(dim=0)  # (P, K)
    sin_mean = ty.sin().mean(dim=0)  # (P, K)

    target = torch.exp(-0.5 * t**2)  # (K,) real CF of N(0,1)
    dev = (cos_mean - target) ** 2 + sin_mean**2  # (P, K)
    return dev.mean()
