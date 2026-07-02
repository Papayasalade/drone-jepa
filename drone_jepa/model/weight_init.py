"""Principled weight initialization for the stage-1 latent-dynamics nets.

Motivation (see NOTES / blog experiments): the fly-vs-crash MPPI lottery is a
STAGE-1 phenomenon (encoders + GRU predictor), and those layers currently use
PyTorch stock defaults — Conv1d/Linear on the legacy `kaiming_uniform_(a=sqrt5)`
and, critically, the GRU recurrent matrix on a plain `uniform(+/-1/sqrt(H))`. A
random recurrent matrix has a seed-dependent spectral radius, so some seeds
expand and some contract through the T-step latent rollout — a plausible
mechanism for the basin lottery.

`scheme="orthogonal"` applies the standard literature recipe to stage 1:
  - Conv1d / Linear (GELU nonlinearity)     -> He/Kaiming-normal (relu proxy)
  - GRU hidden->hidden (per gate block)      -> orthogonal (Saxe et al. 2013),
                                                pins the spectral radius ~1 for
                                                every seed
  - GRU input->hidden (per gate block)       -> Xavier/Glorot-uniform
  - all biases                               -> zero

The prober's residual output head keeps its own near-zero init (set in Prober),
so this only touches the stage-1 submodules where the lottery lives.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _init_linear_conv(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")  # GELU ~ relu proxy
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _init_grucell(cell: nn.GRUCell, update_gate_bias: float = 1.0) -> None:
    # PyTorch GRU gate order in the 3H-stacked params is (reset r, update z, new n).
    # h' = (1-z)*n + z*h, so a POSITIVE update-gate (z) bias makes the cell default
    # to carrying its state -> stable recurrence early in training (the GRU analog of
    # the LSTM forget-gate-bias=1 trick, Jozefowicz et al. 2015). This is the piece of
    # the canonical RNN init recipe that orthogonal-alone omits.
    H = cell.hidden_size
    for name, p in cell.named_parameters():
        if "weight_ih" in name:          # (3H, in): xavier per gate block
            for k in range(3):
                nn.init.xavier_uniform_(p[k * H:(k + 1) * H])
        elif "weight_hh" in name:        # (3H, H): orthogonal per gate block
            for k in range(3):
                nn.init.orthogonal_(p[k * H:(k + 1) * H])
        elif "bias" in name:
            nn.init.zeros_(p)
    # set update-gate (block index 1) bias -> carry-state. Split across ih+hh so the
    # effective bias is `update_gate_bias` (they sum in the gate pre-activation).
    with torch.no_grad():
        cell.bias_ih[H:2 * H].fill_(update_gate_bias / 2.0)
        cell.bias_hh[H:2 * H].fill_(update_gate_bias / 2.0)


def apply_init_scheme(model: nn.Module, scheme: str = "default") -> None:
    """Re-initialize the stage-1 nets in place. `default` is a no-op (keeps the
    PyTorch defaults, so existing runs are unchanged). Call on CPU before .to()
    (orthogonal_ uses QR, which is CPU-safe and avoids MPS/QR gaps)."""
    if scheme == "default":
        return
    if scheme != "orthogonal":
        raise ValueError(f"unknown init scheme: {scheme}")
    for enc in (model.state_encoder, model.action_encoder):
        enc.apply(_init_linear_conv)
    # predictor: He-init its Linear readout/h0, orthogonal-init the GRU cell
    model.predictor.apply(_init_linear_conv)   # skips the GRUCell (not Linear/Conv)
    _init_grucell(model.predictor.cell)
