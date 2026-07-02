"""Quick open-loop accuracy probe: how far off is the model's free-running
prediction (no MPPI in the loop), per horizon step? Loads a checkpoint + a
dataset, builds (history, future-actions, future-states) windows from held-out
trajectories, runs model.predict, and reports position RMSE vs ground truth at
each future step. This separates "the model can't predict" from "MPPI misuses it".

  python scripts/openloop_probe.py artifacts/skyjepa_ctbr_rel.pt artifacts/racing_hi_ctbr.pt
"""
import sys

import numpy as np
import torch

from drone_jepa.model.jepa import SkyJEPA

ckpt = sys.argv[1] if len(sys.argv) > 1 else "artifacts/skyjepa_ctbr_rel.pt"
data = sys.argv[2] if len(sys.argv) > 2 else "artifacts/racing_hi_ctbr.pt"
HORIZON = int(sys.argv[3]) if len(sys.argv) > 3 else 40

model, cfg = SkyJEPA.from_checkpoint(ckpt, device="cpu")
H, T = model.H, model.T
print(f"model: pos_mode={getattr(model,'pos_mode','?')} H={H} T={T} (probing horizon {HORIZON})")

d = torch.load(data, weights_only=False)
states, actions = d["states"], d["actions"]  # (N, L, 18), (N, L, 4)
N = states.shape[0]
test = states[int(N * 0.9):], actions[int(N * 0.9):]  # held-out 10%
S, A = test
ntraj, L, _ = S.shape

# sample windows: need H history + HORIZON future
rng = np.random.default_rng(0)
n_win = 2000
errs = np.zeros(HORIZON)
speeds = []
cnt = 0
with torch.no_grad():
    for _ in range(n_win):
        ti = rng.integers(0, ntraj)
        if L < H + HORIZON:
            continue
        k0 = rng.integers(0, L - H - HORIZON)
        sh = S[ti, k0:k0 + H].unsqueeze(0)              # (1,H,18)
        aw = A[ti, k0 + H - 1:k0 + H - 1 + HORIZON].unsqueeze(0)  # (1,HORIZON,4) future actions
        # predict expects action_window indexed H-1.. ; prepend H-1 dummy past actions
        past = A[ti, k0:k0 + H - 1].unsqueeze(0)        # (1,H-1,4)
        full_aw = torch.cat([past, aw], dim=1)          # (1, H-1+HORIZON, 4)
        pred = model.predict(sh, full_aw, horizon=HORIZON)[0]  # (HORIZON,18)
        true = S[ti, k0 + H:k0 + H + HORIZON]           # (HORIZON,18)
        errs += (pred[:, :3] - true[:, :3]).pow(2).sum(-1).sqrt().numpy()
        speeds.append(true[:, 3:6].norm(dim=-1).mean().item())
        cnt += 1

errs /= cnt
print(f"windows={cnt}  mean |v| in test={np.mean(speeds):.1f} m/s")
print("position RMSE by horizon step (m):")
for k in [0, 4, 9, 19, 29, HORIZON - 1]:
    if k < HORIZON:
        print(f"  step {k+1:2d} ({(k+1)*0.05:.2f}s): {errs[k]:.3f} m")
