"""Closed-loop MPPI flight evaluation (importable helper).

Flies a world model inside the MPPI controller on a RotorPy drone tracking a
circular reference, returning position-tracking RMSE. Handles both action modes
(rotor_force via cmd_motor_thrusts; ctbr via the inner rate loop).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from rotorpy.vehicles.multirotor import Multirotor

from ..control import MPPIController
from ..data_gen.sim import _state_to_vec, rotor_force_limit
from ..state import POS

CENTER = np.array([0.0, 0.0, 1.5])


def ref_state(t, L, dt, center=CENTER, radius=1.2, freq=0.15):
    out = np.zeros((L, 18)); out[:, 6:15] = np.eye(3).reshape(9)
    for k in range(L):
        tk = t + (k + 1) * dt; w = 2 * np.pi * freq
        out[k, POS] = center + np.array([radius * np.cos(w * tk) - radius,
                                         radius * np.sin(w * tk), 0.0])
        out[k, 3:6] = np.array([-radius * w * np.sin(w * tk),
                                radius * w * np.cos(w * tk), 0.0])
    return out


def fly(model, mode, params, init0, seconds=8.0, samples=384, smooth=0.8,
        ki=0.0, sampler="gaussian", unc_lambda=0.0, trust_lambda=0.0,
        trust_margin=1.5, act_trust_lambda=0.0, act_trust_margin=2.0, return_traj=False):
    """Closed-loop tracking. Returns RMSE [m] (or (P, R) if return_traj)."""
    H, T = model.H, model.T
    f_max = rotor_force_limit(params); dt = 0.05; n = int(seconds / dt)
    if mode == "ctbr":
        veh = Multirotor(params, control_abstraction="cmd_ctbr", aero=True)
        hover = np.array([params["mass"] * 9.81, 0.0, 0.0, 0.0])
        mppi = MPPIController(model, horizon=min(15, T), samples=samples,
                              sigma=torch.tensor([0.6, 1.5, 1.5, 1.5]), ki=ki,
                              action_low=torch.tensor([0., -12, -12, -12]),
                              action_high=torch.tensor([4 * f_max, 12., 12., 12.]),
                              noise_smooth=smooth, sampler=sampler, unc_lambda=unc_lambda,
                              trust_lambda=trust_lambda, trust_margin=trust_margin,
                              act_trust_lambda=act_trust_lambda, act_trust_margin=act_trust_margin)
        apply = lambda st, a: veh.step(st, {"cmd_thrust": float(a[0]),
                                            "cmd_w": a[1:].numpy()}, dt)
    else:
        veh = Multirotor(params, control_abstraction="cmd_motor_thrusts", aero=True)
        hover = np.full(4, params["mass"] * 9.81 / 4)
        mppi = MPPIController(model, horizon=min(15, T), samples=samples,
                              sigma=0.6, f_max=f_max, noise_smooth=smooth, ki=ki,
                              sampler=sampler, unc_lambda=unc_lambda,
                              trust_lambda=trust_lambda, trust_margin=trust_margin,
                              act_trust_lambda=act_trust_lambda, act_trust_margin=act_trust_margin)
        apply = lambda st, a: veh.step(st, {"cmd_motor_thrusts": a.numpy().clip(0, f_max)}, dt)
    mppi.nominal = torch.tensor(hover, dtype=torch.float32).expand(mppi.T, 4).clone()
    stt = copy.deepcopy(init0); x0 = _state_to_vec(stt)
    hist = torch.tensor(np.tile(x0, (H, 1)), dtype=torch.float32)
    ahist = torch.tensor(np.tile(hover, (H, 1)), dtype=torch.float32)
    P, R = [], []
    for i in range(n):
        ref = torch.tensor(ref_state(i * dt, T, dt), dtype=torch.float32)
        a = mppi.step(hist, ahist, ref); stt = apply(stt, a)
        x = _state_to_vec(stt)
        if not np.isfinite(x).all():
            return (np.array(P), np.array(R)) if return_traj else 99.9
        hist = torch.cat([hist[1:], torch.tensor(x, dtype=torch.float32)[None]], 0)
        ahist = torch.cat([ahist[1:], a[None]], 0)
        P.append(x[POS]); R.append(ref[0, POS].numpy())
    P, R = np.array(P), np.array(R)
    if return_traj:
        return P, R
    return float(np.sqrt(((P - R) ** 2).sum(1).mean()))
