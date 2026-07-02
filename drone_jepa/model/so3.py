"""Minimal differentiable SO(3) utilities (batched, torch).

Only what the DKI needs: hat/skew, the exponential map (Rodrigues), and a
re-orthonormalization helper so accumulated integration error does not drift
rotation matrices off the manifold.

Convention: rotation matrices map body-frame vectors into the world frame.
Angular velocity omega is expressed in the BODY frame, so attitude advances as
    R_{t+1} = R_t @ exp(hat(omega) * dt).
"""

from __future__ import annotations

import torch


def hat(v: torch.Tensor) -> torch.Tensor:
    """Map an axis vector (..., 3) to a skew-symmetric matrix (..., 3, 3)."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = torch.zeros_like(x)
    row0 = torch.stack([o, -z, y], dim=-1)
    row1 = torch.stack([z, o, -x], dim=-1)
    row2 = torch.stack([-y, x, o], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def exp_so3(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Rodrigues' formula: exponential map of an axis-angle vector (..., 3).

    Returns a rotation matrix (..., 3, 3). Numerically stable near theta=0 via
    a small-angle guard on the sin/cos coefficients.
    """
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True)  # (..., 1)
    theta = torch.clamp(theta, min=eps)
    k = omega / theta  # unit axis
    K = hat(k)  # (..., 3, 3)
    theta = theta[..., None]  # (..., 1, 1)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(K.shape)
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def project_so3(R: torch.Tensor) -> torch.Tensor:
    """Project a near-rotation matrix back onto SO(3) via SVD.

    R_proj = U @ diag(1, 1, det(U V^T)) @ V^T, which guards against reflections
    (det = -1) so the result is a proper rotation.
    """
    U, _, Vh = torch.linalg.svd(R)
    det = torch.linalg.det(U @ Vh)  # (...,)
    d = torch.ones(R.shape[:-1], dtype=R.dtype, device=R.device)  # (..., 3)
    d[..., -1] = det
    return (U * d.unsqueeze(-2)) @ Vh
