"""Compare CTBR (cascaded inner-rate-loop) vs rotor-force MPPI control.

Flies the CTBR JEPA (via RotorPy's cmd_ctbr inner loop) and the 2x rotor-force
JEPA (via raw cmd_motor_thrusts) on the SAME domains/reference, reporting
closed-loop tracking RMSE. Also reports CTBR open-loop prediction accuracy.
"""

import copy

import numpy as np
import torch
from rotorpy.vehicles.multirotor import Multirotor

from drone_jepa.control import MPPIController
from drone_jepa.data_gen.sim import (_state_to_vec, random_initial_state,
                                     rotor_force_limit, sample_params)
from drone_jepa.eval.openloop import _test_trajectories, _long_windows
from drone_jepa.model.jepa import SkyJEPA
from drone_jepa.state import POS


def ref_state(t, L, dt, center, radius=1.2, freq=0.15):
    out = np.zeros((L, 18)); out[:, 6:15] = np.eye(3).reshape(9)
    for k in range(L):
        tk = t + (k + 1) * dt; w = 2 * np.pi * freq
        out[k, POS] = center + np.array([radius * np.cos(w * tk) - radius,
                                         radius * np.sin(w * tk), 0.0])
        out[k, 3:6] = np.array([-radius * w * np.sin(w * tk),
                                radius * w * np.cos(w * tk), 0.0])
    return out


def fly(model, mode, params, init0, seconds=8.0, samples=384, smooth=0.8):
    H, T = model.H, model.T
    f_max = rotor_force_limit(params); dt = 0.05
    center = np.array([0.0, 0.0, 1.5]); n = int(seconds / dt)
    if mode == "ctbr":
        veh = Multirotor(params, control_abstraction="cmd_ctbr", aero=True)
        hover = np.array([params["mass"] * 9.81, 0.0, 0.0, 0.0])
        mppi = MPPIController(model, horizon=min(15, T), samples=samples,
                              sigma=torch.tensor([0.6, 1.5, 1.5, 1.5]),
                              action_low=torch.tensor([0., -12, -12, -12]),
                              action_high=torch.tensor([4 * f_max, 12., 12., 12.]),
                              noise_smooth=smooth)
        apply = lambda st, a: veh.step(st, {"cmd_thrust": float(a[0]),
                                            "cmd_w": a[1:].numpy()}, dt)
    else:
        veh = Multirotor(params, control_abstraction="cmd_motor_thrusts", aero=True)
        hover = np.full(4, params["mass"] * 9.81 / 4)
        mppi = MPPIController(model, horizon=min(15, T), samples=samples,
                              sigma=0.6, f_max=f_max, noise_smooth=smooth)
        apply = lambda st, a: veh.step(st, {"cmd_motor_thrusts": a.numpy().clip(0, f_max)}, dt)
    mppi.nominal = torch.tensor(hover, dtype=torch.float32).expand(mppi.T, 4).clone()
    stt = copy.deepcopy(init0); x0 = _state_to_vec(stt)
    hist = torch.tensor(np.tile(x0, (H, 1)), dtype=torch.float32)
    ahist = torch.tensor(np.tile(hover, (H, 1)), dtype=torch.float32)
    P, R = [], []
    for i in range(n):
        ref = torch.tensor(ref_state(i * dt, T, dt, center), dtype=torch.float32)
        a = mppi.step(hist, ahist, ref); stt = apply(stt, a)
        x = _state_to_vec(stt)
        if not np.isfinite(x).all():
            return 99.9
        hist = torch.cat([hist[1:], torch.tensor(x, dtype=torch.float32)[None]], 0)
        ahist = torch.cat([ahist[1:], a[None]], 0)
        P.append(x[POS]); R.append(ref[0, POS].numpy())
    P, R = np.array(P), np.array(R)
    return np.sqrt(((P - R) ** 2).sum(1).mean())


def main():
    ctbr, _ = SkyJEPA.from_checkpoint("artifacts/skyjepa_ctbr.pt")
    rotor, _ = SkyJEPA.from_checkpoint("artifacts/skyjepa.pt")
    H = ctbr.H

    # open-loop accuracy of CTBR model
    st, ac = _test_trajectories("artifacts/dataset_ctbr.pt", all_traj=False)
    X, A = _long_windows(st, ac, H, 40, stride=20); X, A = X[:1200], A[:1200]
    with torch.no_grad():
        p = ctbr.predict(X[:, :H], A, horizon=40)
    pe = (p - X[:, H:H + 40])[..., POS].pow(2).sum(-1).mean(0).sqrt()
    print("CTBR open-loop pos err: 0.5s=%.3f 1s=%.3f 2s=%.3f m" % (pe[9], pe[19], pe[39]))

    print("\nclosed-loop MPPI tracking RMSE (same domains/reference):")
    print(f"{'seed':>4} {'CTBR (inner loop)':>18} {'2x rotor-force':>16}")
    cr, rr = [], []
    for seed in [1, 3, 5, 7]:
        rng = np.random.default_rng(seed)
        params = sample_params(rng)
        init0 = random_initial_state(rng, center=np.array([0.0, 0.0, 1.5]))
        c = fly(ctbr, "ctbr", params, init0)
        r = fly(rotor, "rotor_force", params, init0)
        cr.append(c); rr.append(r)
        print(f"{seed:>4} {c:>18.2f} {r:>16.2f}")
    print(f"{'mean':>4} {np.mean(cr):>18.2f} {np.mean(rr):>16.2f}")


if __name__ == "__main__":
    main()
