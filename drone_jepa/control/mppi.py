"""Deployment MPPI using the learned SkyJEPA world model (Table II).

The model answers one question — "given my recent state/action history and a
candidate action sequence, where will I be over the next T steps?" — and MPPI
wraps it: sample S action sequences, roll each out through the model, cost them
against a reference, importance-weight, and execute the first action.

MPPI params (Table II): S=512, horizon=15, dt=0.05. Actions are rotor forces,
clamped to motor limits. Only the first action is applied; the nominal sequence
is warm-started (shifted) each control step for receding-horizon control.
"""

from __future__ import annotations

import torch

from ..state import OMEGA, POS, ROT, VEL


class MPPIController:
    def __init__(self, model, horizon: int = 15, samples: int = 512,
                 sigma: float = 1.0, temperature: float = 1.0,
                 f_min: float = 0.0, f_max: float = 15.0,
                 q_pos: float = 8.0, q_vel: float = 1.0, q_att: float = 2.0,
                 q_omega: float = 0.05, r_ctrl: float = 1e-3,
                 noise_smooth: float = 0.0, ki: float = 0.0, i_max: float = 1.0,
                 sampler: str = "gaussian", icem_iters: int = 3,
                 icem_elite_frac: float = 0.1, icem_beta: float = 2.5,
                 icem_knots: int | None = None, icem_keep_frac: float = 0.2,
                 unc_lambda: float = 0.0, trust_lambda: float = 0.0,
                 trust_margin: float = 1.5,
                 act_trust_lambda: float = 0.0, act_trust_margin: float = 2.0,
                 action_low=None, action_high=None, device="cpu"):
        self.model = model.to(device).eval()
        self.H = model.H
        self.T = min(horizon, model.T)
        self.S = samples
        self.temp = temperature
        self.q = dict(pos=q_pos, vel=q_vel, att=q_att, omega=q_omega)
        self.r = r_ctrl
        self.noise_smooth = noise_smooth
        self.ki = ki                       # integral gain on position error
        self.i_max = i_max                 # anti-windup clamp [m·s]
        self.integral = torch.zeros(3, device=torch.device(device))
        # iCEM (1) + colored noise (2) + spline/control-point basis (3)
        self.sampler = sampler
        self.icem_iters = icem_iters
        self.n_elite = max(8, int(icem_elite_frac * samples))
        self.icem_beta = icem_beta                       # colored-noise exponent
        self.icem_knots = icem_knots or min(self.T, 6)   # spline control points
        self.n_keep = max(2, int(icem_keep_frac * self.n_elite))
        self.kept = None                                 # elites carried across steps
        self.unc_lambda = unc_lambda                     # ensemble-disagreement penalty
        self.trust_lambda = trust_lambda                 # SIGReg latent trust region
        self.trust_margin = trust_margin
        self.act_trust_lambda = act_trust_lambda         # action-distribution trust region
        self.act_trust_margin = act_trust_margin
        m0 = self.model.members[0] if hasattr(self.model, "members") else self.model
        self.act_mean = getattr(m0, "action_mean", None)
        self.act_std = getattr(m0, "action_std", None)
        self.device = torch.device(device)
        # per-channel action bounds (default = scalar rotor-force limits on all 4)
        dev = self.device
        self.a_low = (torch.as_tensor(action_low, dtype=torch.float32, device=dev)
                      if action_low is not None else torch.full((4,), f_min, device=dev))
        self.a_high = (torch.as_tensor(action_high, dtype=torch.float32, device=dev)
                       if action_high is not None else torch.full((4,), f_max, device=dev))
        # sigma may be a scalar or per-channel (4,)
        self.sigma = torch.as_tensor(sigma, dtype=torch.float32, device=dev)
        # nominal action sequence (T, 4), warm-started across steps
        self.nominal = (0.5 * (self.a_low + self.a_high)).expand(self.T, 4).clone()

    @torch.no_grad()
    def _rollout_cost(self, state_hist, past_actions, candidates, ref):
        """candidates: (S,T,4). returns cost (S,)."""
        S, T = self.S, self.T
        H = self.H
        # assemble action windows (S, H+T, 4): past actions then candidates
        win = torch.zeros(S, H + T, 4, device=self.device)
        if H - 1 > 0:
            win[:, :H - 1] = past_actions[-(H - 1):].unsqueeze(0).expand(S, -1, -1)
        win[:, H - 1:H - 1 + T] = candidates
        win[:, H - 1 + T:] = candidates[:, -1:].expand(-1, H + T - (H - 1 + T), -1)
        sh = state_hist.unsqueeze(0).expand(S, -1, -1)
        latents = None
        if self.unc_lambda > 0 and hasattr(self.model, "predict_with_unc"):
            pred, unc = self.model.predict_with_unc(sh, win, horizon=T)  # (S,T,18),(S,T)
        elif self.trust_lambda > 0 and hasattr(self.model, "predict_with_latents"):
            pred, latents = self.model.predict_with_latents(sh, win, horizon=T)
            unc = None
        else:
            pred, unc = self.model.predict(sh, win, horizon=T), None

        perr = pred[..., POS] - ref[..., POS]
        verr = pred[..., VEL] - ref[..., VEL]
        rerr = pred[..., ROT] - ref[..., ROT]
        oerr = pred[..., OMEGA] - ref[..., OMEGA]
        cost = (self.q["pos"] * perr.pow(2).sum(-1)
                + self.q["vel"] * verr.pow(2).sum(-1)
                + self.q["att"] * rerr.pow(2).sum(-1)
                + self.q["omega"] * oerr.pow(2).sum(-1)).sum(-1)  # (S,)
        cost = cost + self.r * candidates.pow(2).sum((-1, -2))
        if self.act_trust_lambda > 0 and self.act_mean is not None:
            # action-space trust region: penalize candidate actions that stray
            # beyond the training action distribution (in std units) — aimed
            # directly at the OOD actions the prober+DKI decoder mishandles.
            na = (candidates - self.act_mean) / self.act_std            # (S,T,4)
            pen = (na.abs() - self.act_trust_margin).clamp_min(0).pow(2).sum((-1, -2))
            cost = cost + self.act_trust_lambda * pen
        if unc is not None:
            # uncertainty-aware planning: penalize plans the ensemble disagrees
            # on (off-distribution / model-exploiting actions) — stops the
            # aggressive sampler from running away into unreliable regions.
            cost = cost + self.unc_lambda * unc.sum(-1)
        if latents is not None:
            # SIGReg latent trust region: SIGReg makes encoded latents ~N(0,1),
            # so per-dim energy ~1 in-distribution. Penalize plans whose predicted
            # latents leave the manifold (energy > margin) — i.e. model exploitation.
            energy = latents.pow(2).mean(-1)                      # (S,T) per-dim energy
            cost = cost + self.trust_lambda * (energy - self.trust_margin).clamp_min(0).sum(-1)
        return cost

    @torch.no_grad()
    def step(self, state_hist, past_actions, ref):
        """One MPPI control step.

        state_hist:   (H, 18) recent states (last row = now).
        past_actions: (>=H-1, 4) recent applied actions.
        ref:          (T, 18) reference states for t+1 .. t+T (only pos/vel/rot/
                      omega slices are read).
        returns the first rotor-force action (4,) to execute.
        """
        T = self.T
        state_hist = state_hist.to(self.device)
        past_actions = past_actions.to(self.device)
        ref = ref.to(self.device)[:T]
        if self.ki > 0:
            # integral action: accumulate current position error and bias the
            # reference to drive steady-state offset (e.g. altitude) to zero.
            now_err = state_hist[-1, POS] - ref[0, POS]
            self.integral = (self.integral + now_err).clamp(-self.i_max, self.i_max)
            ref = ref.clone()
            ref[:, POS] = ref[:, POS] - self.ki * self.integral
        if self.sampler == "icem":
            return self._step_icem(state_hist, past_actions, ref)
        eps = torch.randn(self.S, T, 4, device=self.device)  # unit noise
        if self.noise_smooth > 0:
            # temporally smooth the noise (causal EMA) so candidate action
            # sequences are smooth — matches the smooth training distribution and
            # avoids the tumbling that jerky rotor-force noise causes.
            a = self.noise_smooth
            for t in range(1, T):
                eps[:, t] = a * eps[:, t - 1] + (1 - a) * eps[:, t]
            eps = eps / (1 - a)  # restore variance lost to the filter
        candidates = (self.nominal.unsqueeze(0) + eps * self.sigma)
        candidates = torch.maximum(torch.minimum(candidates, self.a_high), self.a_low)
        cost = self._rollout_cost(state_hist, past_actions, candidates, ref)

        w = torch.softmax(-(cost - cost.min()) / self.temp, dim=0)  # (S,)
        self.nominal = (w.view(-1, 1, 1) * candidates).sum(0)       # (T,4)
        action = self.nominal[0].clone()
        # warm-start: shift nominal forward one step
        self.nominal = torch.cat([self.nominal[1:], self.nominal[-1:]], dim=0)
        return torch.maximum(torch.minimum(action, self.a_high), self.a_low).cpu()

    # ----------------- iCEM sampler (colored noise + spline basis) --------- #
    def _colored_noise(self, S):
        """(2) colored noise with a 1/f^beta spectrum, sampled at (3) K spline
        control points and linearly interpolated to the horizon T. Returns
        unit-variance noise of shape (S, T, 4)."""
        K = self.icem_knots
        f = torch.fft.rfftfreq(K, device=self.device).clamp_min(1e-6)
        psd = f ** (-self.icem_beta / 2.0)
        psd[0] = 0.0  # drop DC -> zero-mean noise
        white = torch.randn(S, 4, K, device=self.device)
        knots = torch.fft.irfft(torch.fft.rfft(white, dim=-1) * psd, n=K, dim=-1)
        knots = knots / (knots.std() + 1e-6)   # GLOBAL unit-var scale (a flat
        #   individual sequence must NOT be blown up by its own tiny std)
        if K != self.T:                       # spline: interpolate knots -> horizon
            knots = torch.nn.functional.interpolate(
                knots, size=self.T, mode="linear", align_corners=True)
        return knots.permute(0, 2, 1)         # (S, T, 4)

    @torch.no_grad()
    def _step_icem(self, state_hist, past_actions, ref):
        """(1) iterative CEM: each iteration samples around the mean, keeps the
        low-cost elites, and refits the mean+std to them (so the distribution
        adapts — widens when lost, tightens near the optimum). Elites are carried
        across iterations and warm-started across control steps."""
        T, S = self.T, self.S
        mean = self.nominal                          # (T,4) warm-started
        std = self.sigma * torch.ones(T, 4, device=self.device)
        std_floor = 0.3 * self.sigma * torch.ones(T, 4, device=self.device)
        for _ in range(self.icem_iters):
            cand = mean.unsqueeze(0) + std * self._colored_noise(S)
            if self.kept is not None:                # inject carried elites
                cand[:self.kept.shape[0]] = self.kept
            cand = torch.maximum(torch.minimum(cand, self.a_high), self.a_low)
            cost = self._rollout_cost(state_hist, past_actions, cand, ref)
            elite_idx = torch.topk(-cost, self.n_elite).indices
            elites, ecost = cand[elite_idx], cost[elite_idx]
            # SOFT (importance-weighted) refit over the elites — averages in the
            # safe candidates instead of greedily chasing the single best, so it
            # does not exploit the world model's small biases (which made the
            # hard-elite CEM drift/diverge). This keeps it MPPI-robust.
            w = torch.softmax(-(ecost - ecost.min()) / self.temp, dim=0).view(-1, 1, 1)
            mean = (w * elites).sum(0)
            std = (w * (elites - mean).pow(2)).sum(0).sqrt().clamp_min(std_floor)
            self.kept = elites[:self.n_keep]          # carry best to next iter
        action = mean[0].clone()
        # warm-start mean and kept elites by shifting one step forward
        self.nominal = torch.cat([mean[1:], mean[-1:]], dim=0)
        if self.kept is not None:
            self.kept = torch.cat([self.kept[:, 1:], self.kept[:, -1:]], dim=1)
        return torch.maximum(torch.minimum(action, self.a_high), self.a_low).cpu()
