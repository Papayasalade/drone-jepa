"""Quick controller bake-off on the SAME learned model: MPPI vs gradient-based MPC
(autodiff through the differentiable JEPA), tracking a reference with RotorPy as
the true sim. Tells us whether gradient control (short horizon) is worth porting
to Candle, before building a differentiable Candle DKI.

  python scripts/grad_vs_mppi.py artifacts/skyjepa_ctbr_rel.pt
"""
import copy
import sys

import numpy as np
import torch

from drone_jepa.control import MPPIController
from drone_jepa.control.gradient_mpc import GradientMPC
from drone_jepa.data_gen.sim import BASE_PARAMS, _state_to_vec, random_initial_state, rotor_force_limit
from drone_jepa.eval.closed_loop import ref_state
from drone_jepa.model.jepa import SkyJEPA
from drone_jepa.state import POS
from rotorpy.vehicles.multirotor import Multirotor

ckpt = sys.argv[1] if len(sys.argv) > 1 else "artifacts/skyjepa_ctbr_rel.pt"
model, cfg = SkyJEPA.from_checkpoint(ckpt, device="cpu")
H, T = model.H, model.T
params = copy.deepcopy(BASE_PARAMS)
params["mass"] = 0.5
f_max = rotor_force_limit(params)
dt = 0.05
hover = np.array([params["mass"] * 9.81, 0.0, 0.0, 0.0])
a_low = torch.tensor([0.0, -12.0, -12.0, -12.0])
a_high = torch.tensor([4 * f_max, 12.0, 12.0, 12.0])


def fly(controller, seconds=8.0):
    veh = Multirotor(params, control_abstraction="cmd_ctbr", aero=True)
    apply = lambda st, a: veh.step(st, {"cmd_thrust": float(a[0]), "cmd_w": a[1:].numpy()}, dt)
    rng = np.random.default_rng(0)
    stt = random_initial_state(rng, center=(1.2, 0.0, 1.5))
    x0 = _state_to_vec(stt)
    hist = torch.tensor(np.tile(x0, (H, 1)), dtype=torch.float32)
    ahist = torch.tensor(np.tile(hover, (H, 1)), dtype=torch.float32)
    P, R = [], []
    n = int(seconds / dt)
    for i in range(n):
        ref = torch.tensor(ref_state(i * dt, T, dt), dtype=torch.float32)
        a = controller.step(hist, ahist, ref)
        stt = apply(stt, a)
        x = _state_to_vec(stt)
        if not np.isfinite(x).all():
            return 99.9, i  # diverged
        hist = torch.cat([hist[1:], torch.tensor(x, dtype=torch.float32)[None]], 0)
        ahist = torch.cat([ahist[1:], a[None]], 0)
        P.append(x[POS]); R.append(ref[0, POS].numpy())
    P, R = np.array(P), np.array(R)
    return float(np.sqrt(((P - R) ** 2).sum(1).mean())), n


for hz in [15, 12, 10, 6]:
    mppi = MPPIController(model, horizon=hz, samples=256,
                          sigma=torch.tensor([0.6, 1.5, 1.5, 1.5]), noise_smooth=0.8,
                          action_low=a_low, action_high=a_high)
    mppi.nominal = torch.tensor(hover, dtype=torch.float32).expand(mppi.T, 4).clone()
    rmse_m, n_m = fly(mppi)
    grad = GradientMPC(model, horizon=hz, iters=15, lr=0.3,
                       action_low=a_low, action_high=a_high)
    grad.nominal = torch.tensor(hover, dtype=torch.float32).expand(grad.T, 4).clone()
    rmse_g, n_g = fly(grad)
    print(f"horizon {hz:2d}:  MPPI rmse {rmse_m:5.2f} m ({n_m} steps)   "
          f"GRAD rmse {rmse_g:5.2f} m ({n_g} steps)")
