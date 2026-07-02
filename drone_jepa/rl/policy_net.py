"""RL policy network, sized so the *actor* (the part that actually flies, like the
JEPA model in the demo) has ~9.5K parameters — comparable to SkyJEPA's ~9.6K.

Two GELU hidden layers (width 85), continuous CTBR actions. Implements PufferLib's
encode_observations / decode_actions interface so `pufferlib.pufferl` PPO trains it.
The value and log-std heads are tiny training-only extras (not used to fly).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    import pufferlib.pytorch
    _init = pufferlib.pytorch.layer_init
except Exception:  # allow standalone param-count checks without pufferlib
    def _init(layer, std=1.0):
        return layer


class Policy(nn.Module):
    def __init__(self, env, hidden_size: int = 85, encoder_layers: int = 2):
        super().__init__()
        obs_dim = int(np.prod(env.single_observation_space.shape))
        act_dim = int(env.single_action_space.shape[0])
        self.is_continuous = True
        self.hidden_size = hidden_size
        # ACTOR trunk (the deployed model). Reactive: obs -> h -> h (2 layers).
        # Recurrent: a single obs -> h layer, because the LSTMWrapper adds the memory,
        # so the ~9.5K budget goes to the recurrent cell, not a 2nd MLP layer.
        layers = [nn.Linear(obs_dim, hidden_size), nn.GELU()]
        for _ in range(encoder_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.GELU()]
        self.encoder = nn.Sequential(*layers)
        self.decoder_mean = _init(nn.Linear(hidden_size, act_dim), std=0.01)
        # training-only heads (NOT part of the flying policy). Start exploration
        # gentle (std≈0.6) so the untrained policy doesn't tumble-crash instantly.
        self.decoder_logstd = nn.Parameter(-0.5 * torch.ones(1, act_dim))
        self.value = _init(nn.Linear(hidden_size, 1), std=1.0)

    def encode_observations(self, observations, state=None):
        return self.encoder(observations.view(observations.shape[0], -1).float())

    def decode_actions(self, hidden):
        mean = self.decoder_mean(hidden)
        std = torch.exp(self.decoder_logstd.expand_as(mean))
        return torch.distributions.Normal(mean, std), self.value(hidden)

    def forward(self, observations, state=None):
        return self.decode_actions(self.encode_observations(observations, state))

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)

    # ---- param accounting ----
    def param_counts(self):
        actor = sum(p.numel() for n, p in self.named_parameters()
                    if n.startswith("encoder") or n.startswith("decoder_mean"))
        total = sum(p.numel() for p in self.parameters())
        return actor, total


if __name__ == "__main__":
    class _Env:  # minimal stand-in with the two spaces
        class _S:
            shape = (21,)
        class _A:
            shape = (4,)
        single_observation_space = _S()
        single_action_space = _A()

    for h in (80, 85, 90):
        p = Policy(_Env(), hidden_size=h)
        a, t = p.param_counts()
        print(f"hidden={h}: actor(deployed)={a}  total(train)={t}   [JEPA actor ~9578]")
