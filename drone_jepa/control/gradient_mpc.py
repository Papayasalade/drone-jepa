"""Gradient-based MPC — backprop through the differentiable JEPA to the actions.

The opposite extreme from MPPI's random sampling: treat the action plan as a
tensor of optimizable parameters, forward it through the (differentiable) world
model + DKI, and **descend the tracking cost's gradient** w.r.t. the actions.
Same `step()` interface as MPPIController so it drops into the same fly loop.

Expectation (and the point of having it in the race): gradient descent is the
*most aggressive* optimizer, so on an imperfect learned model it exploits model
error even harder than iCEM — a clean demonstration of "aggressive planner +
imperfect model = divergence" from the gradient end.
"""

from __future__ import annotations

import torch

from ..state import OMEGA, POS, ROT, VEL


class GradientMPC:
    def __init__(self, model, horizon: int = 15, iters: int = 12, lr: float = 0.3,
                 q_pos: float = 8.0, q_vel: float = 1.0, q_att: float = 2.0,
                 q_omega: float = 0.05, ki: float = 0.0, i_max: float = 1.0,
                 action_low=None, action_high=None, device="cpu"):
        self.model = model.to(device).eval()
        self.H = model.H
        self.T = min(horizon, model.T)
        self.iters = iters
        self.lr = lr
        self.q = dict(pos=q_pos, vel=q_vel, att=q_att, omega=q_omega)
        self.ki, self.i_max = ki, i_max
        self.integral = torch.zeros(3, device=torch.device(device))
        self.device = torch.device(device)
        dev = self.device
        self.a_low = (torch.as_tensor(action_low, dtype=torch.float32, device=dev)
                      if action_low is not None else torch.zeros(4, device=dev))
        self.a_high = (torch.as_tensor(action_high, dtype=torch.float32, device=dev)
                       if action_high is not None else torch.full((4,), 15.0, device=dev))
        self.nominal = (0.5 * (self.a_low + self.a_high)).expand(self.T, 4).clone()

    def _cost(self, state_hist, past_actions, a, ref):
        H, T = self.H, self.T
        # build the action window: past actions then the optimizable plan a
        win = torch.zeros(1, H + T, 4, device=self.device)
        if H - 1 > 0:
            win[0, :H - 1] = past_actions[-(H - 1):]
        win[0, H - 1:H - 1 + T] = a
        win[0, H - 1 + T:] = a[-1:]
        pred = self.model.predict_grad(state_hist[None], win, horizon=T)[0]  # (T,18)
        c = (self.q["pos"] * (pred[:, POS] - ref[:, POS]).pow(2).sum(-1)
             + self.q["vel"] * (pred[:, VEL] - ref[:, VEL]).pow(2).sum(-1)
             + self.q["att"] * (pred[:, ROT] - ref[:, ROT]).pow(2).sum(-1)
             + self.q["omega"] * (pred[:, OMEGA] - ref[:, OMEGA]).pow(2).sum(-1)).sum()
        return c

    def step(self, state_hist, past_actions, ref):
        T = self.T
        state_hist = state_hist.to(self.device)
        past_actions = past_actions.to(self.device)
        ref = ref.to(self.device)[:T]
        if self.ki > 0:
            now_err = state_hist[-1, POS] - ref[0, POS]
            self.integral = (self.integral + now_err).clamp(-self.i_max, self.i_max)
            ref = ref.clone()
            ref[:, POS] = ref[:, POS] - self.ki * self.integral
        a = self.nominal.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([a], lr=self.lr)
        for _ in range(self.iters):
            opt.zero_grad()
            cost = self._cost(state_hist, past_actions, a, ref)
            cost.backward()
            opt.step()
            with torch.no_grad():
                a.clamp_(self.a_low, self.a_high)
        a = a.detach()
        action = a[0].clone()
        self.nominal = torch.cat([a[1:], a[-1:]], dim=0)  # warm-start
        return action.clamp(self.a_low, self.a_high).cpu()
