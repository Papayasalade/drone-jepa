"""Unit checks for the SO(3) utilities and the DKI nominal model."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from drone_jepa.model.dki import DKI
from drone_jepa.model.so3 import exp_so3, hat, project_so3
from drone_jepa.state import GRAVITY, OMEGA, POS, ROT, VEL


def test_exp_so3_matches_scipy():
    torch.manual_seed(0)
    v = torch.randn(50, 3)
    R = exp_so3(v).numpy()
    R_ref = Rotation.from_rotvec(v.numpy()).as_matrix()
    err = np.abs(R - R_ref).max()
    assert err < 1e-5, err
    print(f"exp_so3 vs scipy rotvec: max err {err:.2e}  OK")


def test_exp_so3_orthonormal():
    v = torch.randn(100, 3) * 3.0
    R = exp_so3(v)
    I = torch.eye(3).expand(100, 3, 3)
    err = (R.transpose(-1, -2) @ R - I).abs().max().item()
    det = torch.linalg.det(R)
    assert err < 1e-5 and (det - 1).abs().max() < 1e-5
    print(f"exp_so3 orthonormality err {err:.2e}, det~1 OK")


def test_dki_hover():
    """Zero residuals, nominal mass, thrust=weight -> stays at rest."""
    m = 0.5
    dki = DKI(dt=0.05, m_nominal=m)
    x = torch.zeros(1, 18)
    x[:, ROT] = torch.eye(3).reshape(9)  # level attitude
    x[0, POS] = torch.tensor([0.0, 0.0, 2.0])
    a = torch.full((1, 4), m * GRAVITY / 4)  # total thrust = weight
    dvdot = torch.zeros(1, 3)
    K = torch.zeros(1, 3, 4)
    for _ in range(20):
        x = dki.step(x, a, dvdot, K)
    assert torch.allclose(x[0, POS], torch.tensor([0.0, 0.0, 2.0]), atol=1e-5)
    assert torch.allclose(x[0, VEL], torch.zeros(3), atol=1e-5)
    print(f"DKI hover: final pos {x[0, POS].tolist()}  vel {x[0, VEL].tolist()}  OK")


def test_dki_freefall():
    """Zero thrust -> falls under gravity: v_z = -g t."""
    dki = DKI(dt=0.05, m_nominal=0.5)
    x = torch.zeros(1, 18)
    x[:, ROT] = torch.eye(3).reshape(9)
    a = torch.zeros(1, 4)
    dvdot = torch.zeros(1, 3); K = torch.zeros(1, 3, 4)
    n = 10
    for _ in range(n):
        x = dki.step(x, a, dvdot, K)
    expected_vz = -GRAVITY * n * 0.05
    assert abs(x[0, VEL][2].item() - expected_vz) < 1e-4
    print(f"DKI freefall: v_z={x[0,VEL][2]:.4f} expected {expected_vz:.4f}  OK")


def test_dki_spin():
    """Constant angular accel about z via K -> attitude rotates, stays SO(3)."""
    dki = DKI(dt=0.05, m_nominal=0.5)
    x = torch.zeros(1, 18); x[:, ROT] = torch.eye(3).reshape(9)
    a = torch.ones(1, 4)
    K = torch.zeros(1, 3, 4); K[0, 2, :] = 0.1  # yaw accel
    dvdot = torch.zeros(1, 3)
    for _ in range(20):
        x = dki.step(x, a, dvdot, K)
    R = x[0, ROT].reshape(3, 3)
    orth = (R.T @ R - torch.eye(3)).abs().max().item()
    assert orth < 1e-6, orth
    print(f"DKI spin: omega_z={x[0,OMEGA][2]:.3f}  R-orthonorm err {orth:.2e}  OK")


if __name__ == "__main__":
    test_exp_so3_matches_scipy()
    test_exp_so3_orthonormal()
    test_dki_hover()
    test_dki_freefall()
    test_dki_spin()
    print("\nall physics checks passed")
