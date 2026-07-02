"""SkyJEPA assembled model: encoders + GRU predictor + prober + DKI.

Tensor conventions for a training window of length L = H + T (history then
horizon), batch B:
  X : (B, L, 18) physical state trajectory
  A : (B, L, 4)  rotor-force trajectory; A[:, k] takes state k -> state k+1

Index t = H - 1 is "now": the last history step. We predict the latents and
physical states for the future steps t+1 .. t+T (indices H .. H+T-1).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .dki import DKI
from .encoders import ActionEncoder, StateEncoder
from .predictor import Predictor
from .prober import Prober


@dataclass
class LatentOut:
    s_now: torch.Tensor       # (B, 16)      encoded current latent s_t
    s_pred: torch.Tensor      # (B, T, 16)   predicted future latents s_bar
    s_target: torch.Tensor    # (B, T, 16)   encoded-true future latents
    s_all: torch.Tensor       # (B, L, 16)   all per-step encoded latents (for SIGReg)


class SkyJEPA(nn.Module):
    def __init__(self, history: int = 10, horizon: int = 20,
                 dt: float = 0.05, m_nominal: float = 0.5,
                 width_mult: int = 1, prober_hidden: int | None = None,
                 action_mode: str = "rotor_force", probabilistic: bool = False,
                 learn_thrust: bool = False, pos_mode: str = "zero",
                 n_params: int = 0, param_emb: int = 8,
                 mass_aware: bool = False, prober_inputs: str = "full"):
        super().__init__()
        # mass_aware: the ONLY drone knowledge is the true mass, used analytically
        # (0 learned params): actions are hover-normalized before the encoders
        # (thrust in units of m*g, the RL policy's de-normalization trick), and the
        # DKI's nominal inverse mass becomes 1/m_true instead of 1/m_nominal.
        self.mass_aware = mass_aware
        self.H = history
        self.T = horizon
        self.width_mult = width_mult
        self.action_mode = action_mode
        # how absolute position (state dims 0:3) is handled before the neural nets:
        #   "zero"     -> dropped entirely (translation-invariant; dynamics ignore it)
        #   "relative" -> expressed relative to the decision-time position (also
        #                 translation-invariant, but keeps within-window displacement)
        self.pos_mode = pos_mode
        self.probabilistic = probabilistic
        self.learn_thrust = learn_thrust
        w = width_mult
        # base widths are the paper's shapes; width_mult scales them uniformly.
        # fractional widths are allowed (rounded) so we can probe intermediate
        # capacities between the integer steps (1x=9.5K, 1.5x=~20K, 2x=~34.5K).
        iw = lambda x: max(1, int(round(x)))
        state_ch = [iw(8 * w), iw(8 * w), iw(16 * w)]
        action_ch = [iw(4 * w), iw(4 * w), iw(8 * w)]
        state_latent, action_latent = iw(16 * w), iw(8 * w)
        prober_hidden = prober_hidden if prober_hidden is not None else iw(40 * w)
        # drone-param conditioning (n_params>0): a small encoder maps the physical
        # params -> a param embedding fed to the predictor + prober, so the model is
        # TOLD the drone (mass/inertia/...) instead of inferring it from 0.5s history.
        self.n_params = n_params
        self.param_emb = param_emb
        param_dim = param_emb if n_params > 0 else 0
        self.param_dim = param_dim
        self.state_encoder = StateEncoder(state_ch)
        self.action_encoder = ActionEncoder(action_ch)
        self.predictor = Predictor(state_latent, action_latent, param_dim=param_dim)
        self.prober_inputs = prober_inputs
        self.prober = Prober(state_latent=state_latent, hidden=prober_hidden,
                             action_mode=action_mode, probabilistic=probabilistic,
                             learn_thrust=learn_thrust, param_dim=param_dim,
                             inputs=prober_inputs)
        self.dki = DKI(dt=dt, m_nominal=m_nominal, action_mode=action_mode)
        # input normalization (encoder sees normalized features; DKI uses raw).
        self.register_buffer("state_mean", torch.zeros(18))
        self.register_buffer("state_std", torch.ones(18))
        self.register_buffer("action_mean", torch.zeros(4))
        self.register_buffer("action_std", torch.ones(4))
        if n_params > 0:
            self.param_enc = nn.Sequential(
                nn.Linear(n_params, 16), nn.GELU(), nn.Linear(16, param_emb))
            self.register_buffer("param_mean", torch.zeros(n_params))
            self.register_buffer("param_std", torch.ones(n_params))
        self.prober_hidden = prober_hidden

    @classmethod
    def from_checkpoint(cls, path: str, device="cpu"):
        """Rebuild a SkyJEPA from a saved checkpoint (honors prober width)."""
        ck = torch.load(path, weights_only=False, map_location=device)
        cfg = ck["config"]
        model = cls(history=cfg["history"], horizon=cfg["horizon"],
                    width_mult=cfg.get("width_mult", 1),
                    prober_hidden=cfg.get("prober_hidden"),
                    action_mode=cfg.get("action_mode", "rotor_force"),
                    probabilistic=cfg.get("probabilistic", False),
                    learn_thrust=cfg.get("learn_thrust", False),
                    pos_mode=cfg.get("pos_mode", "zero"),
                    n_params=cfg.get("n_params", 0),
                    param_emb=cfg.get("param_emb", 8),
                    mass_aware=cfg.get("mass_aware", False),
                    prober_inputs=cfg.get("prober_inputs", "full")).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        return model, cfg

    @torch.no_grad()
    def fit_normalization(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        """states: (..., 18), actions: (..., 4). Sets normalization buffers."""
        s = states.reshape(-1, 18)
        a = actions.reshape(-1, 4)
        self.state_mean.copy_(s.mean(0))
        self.state_std.copy_(s.std(0).clamp_min(1e-4))
        self.action_mean.copy_(a.mean(0))
        self.action_std.copy_(a.std(0).clamp_min(1e-4))

    def _norm_state(self, X, ref):
        """Normalize state, with translation-invariant handling of position.
        `ref` is the decision-time position (B,...,3) to measure against in
        relative mode. Quad dynamics don't depend on absolute position, so we
        either zero it or express it relative to `ref`. The DKI uses raw position."""
        Xn = (X - self.state_mean) / self.state_std
        if self.pos_mode == "relative":
            rel = (X[..., :3] - ref) / self.state_std[:3]
            return torch.cat([rel, Xn[..., 3:]], dim=-1)
        return torch.cat([torch.zeros_like(Xn[..., :3]), Xn[..., 3:]], dim=-1)

    def _norm_action(self, A, M=None):
        A = self._hover_norm(A, M)
        return (A - self.action_mean) / self.action_std

    def _hover_norm(self, A, M):
        """mass_aware: express thrust in units of the drone's own hover thrust.
        M: (B,) true masses. For ctbr only channel 0 is a force; for rotor_force
        all four are. No-op when not mass_aware or M is None (falls back to the
        raw-Newton convention). fit_normalization must then be given
        hover-normalized actions so the z-score buffers match."""
        if not self.mass_aware or M is None:
            return A
        hover = (M * 9.81).reshape(-1, *([1] * (A.dim() - 1)))
        if self.action_mode == "ctbr":
            return torch.cat([A[..., :1] / hover, A[..., 1:]], dim=-1)
        return A / (hover / 4.0)

    @torch.no_grad()
    def fit_param_normalization(self, params: torch.Tensor) -> None:
        """params: (N, n_params). Sets param-normalization buffers (z-score)."""
        if self.n_params == 0:
            return
        p = params.reshape(-1, self.n_params).float()
        self.param_mean.copy_(p.mean(0))
        self.param_std.copy_(p.std(0).clamp_min(1e-8))

    def _param_lat(self, P):
        """Encode a (B, n_params) drone-param vector -> (B, param_emb), or None."""
        if self.n_params == 0 or P is None:
            return None
        Pn = (P - self.param_mean) / self.param_std
        return self.param_enc(Pn)

    # ---- Stage 1: latent dynamics --------------------------------------
    def latent_forward(self, X: torch.Tensor, A: torch.Tensor, P=None,
                       M=None) -> LatentOut:
        H, T = self.H, self.T
        ref = X[:, H - 1:H, :3]                         # decision-time position (B,1,3)
        S = self.state_encoder(self._norm_state(X, ref))  # (B, L, 16)
        Z = self.action_encoder(self._norm_action(A, M))  # (B, L, 8)
        s_now = S[:, H - 1]                   # (B, 16)
        z_seq = Z[:, H - 1:H - 1 + T]         # (B, T, 8)  actions at t..t+T-1
        s_pred = self.predictor(s_now, z_seq, self._param_lat(P))  # (B, T, 16)
        s_target = S[:, H:H + T]             # (B, T, 16) encoded-true future
        return LatentOut(s_now=s_now, s_pred=s_pred, s_target=s_target, s_all=S)

    # ---- Stage 2: physical rollout via prober + DKI --------------------
    def physical_rollout(self, X: torch.Tensor, A: torch.Tensor,
                         s_now: torch.Tensor, s_pred: torch.Tensor,
                         return_logvar: bool = False, P=None, M=None):
        """Integrate physical state from the true current state x_t forward T
        steps. Returns predicted states (B, T, 18) for t+1 .. t+T (or, if
        return_logvar, also (B, T, 18) per-step next-state log-variance).

        The latent driving the transition [t+k -> t+k+1] is the latent *at* t+k:
        s0 for k=0, then the predicted latents for k>=1.
        """
        H = self.H
        T = s_pred.shape[1]                   # horizon (may exceed training T)
        x = X[:, H - 1]                       # true current physical state
        ref = X[:, H - 1, :3]                 # decision-time position (B,3), fixed over rollout
        # source latent for each transition step k = 0..T-1
        src = torch.cat([s_now.unsqueeze(1), s_pred[:, :T - 1]], dim=1)  # (B,T,16)
        p_lat = self._param_lat(P)
        preds, logvars = [], []
        ts_mass = None
        if self.mass_aware and M is not None:
            ts_mass = (1.0 / M).reshape(-1, 1)          # true inverse mass for the DKI
        for k in range(T):
            a_k = A[:, H - 1 + k]
            dvdot, K, lv, ts = self.prober(src[:, k], self._norm_state(x, ref),
                                           self._norm_action(a_k, M), p_lat)
            x = self.dki.step(x, a_k, dvdot, K,
                              thrust_scale=ts_mass if ts_mass is not None else ts)
            preds.append(x)
            if return_logvar:
                logvars.append(lv)
        states = torch.stack(preds, dim=1)
        if return_logvar:
            return states, torch.stack(logvars, dim=1)
        return states

    # ---- Deployment inference (no future ground truth) -----------------
    @torch.no_grad()
    def predict(self, state_hist: torch.Tensor, action_window: torch.Tensor,
                horizon: int | None = None, P=None, M=None):
        """Predict the future physical trajectory from history alone.

        state_hist:    (B, H, 18) the H most recent states (index H-1 = now).
        action_window: (B, H+L, 4) actions; indices H-1 .. H-1+L-1 are the L
                       future actions a_t .. a_{t+L-1}.
        horizon L:     defaults to the training T; may be larger to probe the
                       compounding regime (the GRU/DKI unroll arbitrarily).
        returns        (B, L, 18) predicted states for t+1 .. t+L.
        This is the function MPPI queries for every candidate action sequence.
        """
        H = self.H
        T = horizon if horizon is not None else self.T
        S = self.state_encoder(self._norm_state(state_hist, state_hist[:, H - 1:H, :3]))
        Z = self.action_encoder(self._norm_action(action_window, M))
        s_now = S[:, H - 1]
        z_seq = Z[:, H - 1:H - 1 + T]
        s_pred = self.predictor(s_now, z_seq, self._param_lat(P))
        # reuse physical rollout; it only reads state_hist[:, H-1] and the actions
        return self.physical_rollout(state_hist, action_window, s_now, s_pred, P=P, M=M)

    def predict_grad(self, state_hist, action_window, horizon=None):
        """Same as predict() but WITHOUT no_grad — gradients flow to
        action_window, so a gradient-based controller can backprop the cost
        through the differentiable model+DKI to the actions."""
        H = self.H
        T = horizon if horizon is not None else self.T
        S = self.state_encoder(self._norm_state(state_hist, state_hist[:, H - 1:H, :3]))
        Z = self.action_encoder(self._norm_action(action_window))
        s_now = S[:, H - 1]
        s_pred = self.predictor(s_now, Z[:, H - 1:H - 1 + T])
        return self.physical_rollout(state_hist, action_window, s_now, s_pred)

    @torch.no_grad()
    def predict_with_latents(self, state_hist, action_window, horizon=None):
        """Returns (states (B,L,18), predicted latents s_pred (B,L,latent)).

        SIGReg trains the latent space to be ~N(0,1), so the magnitude of these
        predicted latents is a free out-of-distribution signal: when a candidate
        action plan drives s_pred off the training manifold (energy >> 1 per dim),
        the plan is exploiting the model — exactly what a latent trust region
        penalizes. Used by MPPI's `trust_lambda`."""
        H = self.H
        T = horizon if horizon is not None else self.T
        S = self.state_encoder(self._norm_state(state_hist, state_hist[:, H - 1:H, :3]))
        Z = self.action_encoder(self._norm_action(action_window))
        s_now = S[:, H - 1]
        s_pred = self.predictor(s_now, Z[:, H - 1:H - 1 + T])
        states = self.physical_rollout(state_hist, action_window, s_now, s_pred)
        return states, s_pred

    @torch.no_grad()
    def predict_with_unc(self, state_hist, action_window, horizon=None):
        """Probabilistic models only: returns (mean states (B,L,18), per-step
        aleatoric uncertainty (B,L) = mean predicted std in normalized units)."""
        H = self.H
        T = horizon if horizon is not None else self.T
        S = self.state_encoder(self._norm_state(state_hist, state_hist[:, H - 1:H, :3]))
        Z = self.action_encoder(self._norm_action(action_window))
        s_now = S[:, H - 1]
        s_pred = self.predictor(s_now, Z[:, H - 1:H - 1 + T])
        states, logvar = self.physical_rollout(state_hist, action_window, s_now,
                                               s_pred, return_logvar=True)
        unc = torch.exp(0.5 * logvar).mean(-1)   # (B, L) avg predicted std
        return states, unc
