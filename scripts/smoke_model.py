"""Smoke test: build the model, run both stages on random data, count params."""

import torch

from drone_jepa.model import SkyJEPA
from drone_jepa.model.losses import latent_loss, physical_loss
from drone_jepa.state import ACTION_DIM, STATE_DIM

torch.manual_seed(0)

H, T, B = 10, 20, 8
model = SkyJEPA(history=H, horizon=T)
L = H + T

# random-ish state window with valid rotation matrices
X = torch.randn(B, L, STATE_DIM)
R = torch.eye(3).expand(B, L, 3, 3).reshape(B, L, 9)
X[..., 6:15] = R
A = torch.rand(B, L, ACTION_DIM) * 3.0  # rotor forces

# --- Stage 1 ---
out = model.latent_forward(X, A)
l1, c1 = latent_loss(out)
print("latent shapes:", out.s_pred.shape, out.s_target.shape, out.s_all.shape)
print("stage1 loss:", float(l1.detach()), {k: float(v) for k, v in c1.items()})
l1.backward()
print("stage1 backward ok")

# --- Stage 2 ---
out = model.latent_forward(X, A)
pred = model.physical_rollout(X, A, out.s_now.detach(), out.s_pred.detach())
target = X[:, H:H + T]
l2, c2 = physical_loss(pred, target)
print("physical pred shape:", pred.shape)
print("stage2 loss:", float(l2), {k: float(v) for k, v in c2.items()})
l2.backward()
print("stage2 backward ok")

# --- param counts ---
def n(m):
    return sum(p.numel() for p in m.parameters())

print("\nparam counts:")
print("  state_encoder ", n(model.state_encoder))
print("  action_encoder", n(model.action_encoder))
print("  predictor     ", n(model.predictor))
print("  prober        ", n(model.prober))
print("  TOTAL         ", n(model))
print("  (encoders+pred+prober ~ paper's 9K target)")
