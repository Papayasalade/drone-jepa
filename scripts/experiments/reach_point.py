"""'Ask the model for actions to reach a point.' Pure gradient planning query:
fix a target position, gradient-descend a 0.5s action plan so the model's PREDICTED
final position lands on the target, then EXECUTE those actions in the true sim and
see where the drone actually ends up. The model-predicted-vs-actual gap is exactly
the model exploitation, made visible.

  python scripts/reach_point.py artifacts/skyjepa_ctbr_rel.pt
"""
import copy
import sys

import numpy as np
import torch

from drone_jepa.data_gen.sim import BASE_PARAMS, _state_to_vec, rotor_force_limit
from drone_jepa.model.jepa import SkyJEPA
from drone_jepa.state import POS, VEL
from rotorpy.vehicles.multirotor import Multirotor

ckpt = sys.argv[1] if len(sys.argv) > 1 else "artifacts/skyjepa_ctbr_rel.pt"
model, cfg = SkyJEPA.from_checkpoint(ckpt, device="cpu")
H, T = model.H, model.T
HZ = 10  # plan 0.5 s ahead
params = copy.deepcopy(BASE_PARAMS); params["mass"] = 0.5
f_max = rotor_force_limit(params); dt = 0.05
hover = np.array([params["mass"] * 9.81, 0.0, 0.0, 0.0])
a_low = torch.tensor([0.0, -12.0, -12.0, -12.0]); a_high = torch.tensor([4 * f_max, 12.0, 12.0, 12.0])

# settled-hover start at p0 (history = repeated hover state)
p0 = np.array([0.0, 0.0, 1.5])
x0 = np.zeros(18); x0[POS] = p0; x0[6], x0[10], x0[14] = 1.0, 1.0, 1.0  # identity R
hist0 = torch.tensor(np.tile(x0, (H, 1)), dtype=torch.float32).unsqueeze(0)  # (1,H,18)
past0 = torch.tensor(np.tile(hover, (H - 1, 1)), dtype=torch.float32)


def plan_to(target):
    """Gradient-descend a HZ-step action plan so the model lands at `target`."""
    tgt = torch.tensor(target, dtype=torch.float32)
    a = torch.tensor(np.tile(hover, (HZ, 1)), dtype=torch.float32).requires_grad_(True)
    opt = torch.optim.Adam([a], lr=0.3)
    for _ in range(60):
        opt.zero_grad()
        win = torch.zeros(1, H + HZ, 4)
        win[0, :H - 1] = past0
        win[0, H - 1:H - 1 + HZ] = a
        win[0, H - 1 + HZ:] = a[-1:]
        pred = model.predict_grad(hist0, win, horizon=HZ)[0]  # (HZ,18)
        # land on the target and arrive ~stopped
        cost = (pred[-1, POS] - tgt).pow(2).sum() + 0.2 * pred[-1, VEL].pow(2).sum()
        cost.backward(); opt.step()
        with torch.no_grad():
            a.clamp_(a_low, a_high)
    a = a.detach()
    with torch.no_grad():
        win = torch.zeros(1, H + HZ, 4); win[0, :H - 1] = past0
        win[0, H - 1:H - 1 + HZ] = a; win[0, H - 1 + HZ:] = a[-1:]
        model_pred = model.predict(hist0, win, horizon=HZ)[0, -1, POS].numpy()
    return a, model_pred


def execute(a):
    """Run the planned actions in the TRUE sim; return the actual final position."""
    veh = Multirotor(params, control_abstraction="cmd_ctbr", aero=True)
    st = veh.initial_state if hasattr(veh, "initial_state") else None
    # build a RotorPy state dict at p0, hovering
    st = {"x": p0.copy(), "v": np.zeros(3), "q": np.array([0, 0, 0, 1.0]),
          "w": np.zeros(3), "wind": np.zeros(3),
          "rotor_speeds": np.full(4, np.sqrt(params["mass"] * 9.81 / (4 * params["k_eta"])))}
    for k in range(HZ):
        ak = a[k]
        st = veh.step(st, {"cmd_thrust": float(ak[0]), "cmd_w": ak[1:].numpy()}, dt)
    return st["x"]


print(f"model pos_mode={getattr(model,'pos_mode','?')}  plan horizon {HZ} ({HZ*dt}s)")
print(f"{'target':>18} | {'model THINKS it lands':>22} | {'ACTUALLY lands':>16} | model-err  real-err")
for off in [[1.5, 0, 0], [0, 0, 1.5], [2, 1.5, 0.5], [3, 0, 0], [-1.5, -1.5, 0.5]]:
    tgt = p0 + np.array(off)
    a, mpred = plan_to(tgt)
    real = execute(a)
    me = np.linalg.norm(mpred - tgt); re = np.linalg.norm(real - tgt)
    print(f"[{tgt[0]:4.1f},{tgt[1]:4.1f},{tgt[2]:4.1f}] | "
          f"[{mpred[0]:5.2f},{mpred[1]:5.2f},{mpred[2]:5.2f}] | "
          f"[{real[0]:5.2f},{real[1]:5.2f},{real[2]:5.2f}] | "
          f"{me:6.2f}m   {re:6.2f}m")
