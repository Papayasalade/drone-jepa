"""Two-stage SkyJEPA training.

Stage 1: learn the latent dynamics (encoders + GRU predictor) with the latent
         consistency loss + SIGReg. Collapse is monitored via latent std.
Stage 2: freeze encoders + predictor; train the prober (+ DKI) with a supervised
         physical-state L2.

Optimizer follows the paper: Adam, weight decay 1e-5, grad clip 0.5, LR warmup
0 -> 5e-3 then cosine -> 1e-4. Step budgets are CLI-configurable so a laptop run
is quick; pass larger values to approach the paper's 50 epochs / batch 2048.
"""

from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from .data_gen.dataset import load_dataset
from .model.jepa import LatentOut, SkyJEPA
from .model.weight_init import apply_init_scheme
from .model.losses import latent_loss, nll_physical_loss, physical_loss


def lr_at(step, total, warmup=4000, lr_max=5e-3, lr_min=1e-4):
    if step < warmup:
        return lr_max * step / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * min(t, 1.0)))


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


@torch.no_grad()
def val_stage1(model, X, A):
    """Held-out latent metrics: collapse probe + prediction MSE (overfit check)."""
    was_training = model.training
    model.eval()
    out = model.latent_forward(X, A)
    std = out.s_all.reshape(-1, out.s_all.shape[-1]).std(0).mean().item()
    val_pred = torch.nn.functional.mse_loss(out.s_pred, out.s_target).item()
    if was_training:
        model.train()
    return std, val_pred


@torch.no_grad()
def val_stage2(model, X, A):
    """Held-out physical position RMSE (train vs this = overfit gap)."""
    was_training = model.training
    model.eval()
    H, T = model.H, model.T
    out = model.latent_forward(X, A)
    pred = model.physical_rollout(X, A, out.s_now, out.s_pred)
    _, comps = physical_loss(pred, X[:, H:H + T], std=model.state_std)
    if was_training:
        model.train()
    return comps["pos"].item()


def train_stage1(model, loader, val_X, val_A, steps, device, lambda_sig,
                 ckpt_every=0, save_ckpt=None, ema_tau=0.0):
    params = (list(model.state_encoder.parameters())
              + list(model.action_encoder.parameters())
              + list(model.predictor.parameters()))
    opt = torch.optim.Adam(params, lr=0.0, weight_decay=1e-5)
    # EMA target encoder (basin-structure ablation): s_target comes from a slow
    # momentum copy of the state encoder, so encoder and predictor cannot
    # co-adapt through the target. SIGReg still acts on the online latents.
    ema_enc = None
    if ema_tau > 0:
        import copy as _copy
        ema_enc = _copy.deepcopy(model.state_encoder)
        for p in ema_enc.parameters():
            p.requires_grad_(False)
    model.train()
    step = 0
    while step < steps:
        for X, A in loader:
            if step >= steps:
                break
            X, A = X.to(device), A.to(device)
            set_lr(opt, lr_at(step, steps))
            out = model.latent_forward(X, A)
            if ema_enc is not None:
                H, T = model.H, model.T
                with torch.no_grad():
                    S_t = ema_enc(model._norm_state(X, X[:, H - 1:H, :3]))
                out = LatentOut(s_now=out.s_now, s_pred=out.s_pred,
                                s_target=S_t[:, H:H + T], s_all=out.s_all)
            loss, comps = latent_loss(out, lambda_sig=lambda_sig)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            opt.step()
            if ema_enc is not None:
                with torch.no_grad():
                    for pe, po in zip(ema_enc.parameters(),
                                      model.state_encoder.parameters()):
                        pe.lerp_(po, 1.0 - ema_tau)
            if step % max(1, steps // 20) == 0:
                std, val_pred = val_stage1(model, val_X, val_A)
                print(f"  [s1 {step:5d}] loss={loss.item():.4f} "
                      f"pred={comps['pred']:.4f} sig={comps['sigreg']:.4f} "
                      f"latent_std={std:.3f} val_pred={val_pred:.4f}", flush=True)
            step += 1
            if ckpt_every and save_ckpt and step % ckpt_every == 0:
                save_ckpt(f"s1_{step}")


def train_stage2(model, loader, val_X, val_A, steps, device,
                 ckpt_every=0, save_ckpt=None):
    # freeze everything but the prober
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.prober.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(model.prober.parameters(), lr=0.0, weight_decay=1e-5)
    H, T = model.H, model.T
    model.train()
    step = 0
    while step < steps:
        for X, A in loader:
            if step >= steps:
                break
            X, A = X.to(device), A.to(device)
            set_lr(opt, lr_at(step, steps))
            with torch.no_grad():
                out = model.latent_forward(X, A)
            target = X[:, H:H + T]
            if model.probabilistic:
                pred, logvar = model.physical_rollout(X, A, out.s_now, out.s_pred,
                                                      return_logvar=True)
                loss, comps = nll_physical_loss(pred, logvar, target, model.state_std)
            else:
                pred = model.physical_rollout(X, A, out.s_now, out.s_pred)
                loss, comps = physical_loss(pred, target, std=model.state_std)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.prober.parameters(), 0.5)
            opt.step()
            if step % max(1, steps // 20) == 0:
                val_pos = val_stage2(model, val_X, val_A)
                print(f"  [s2 {step:5d}] loss={loss.item():.4f} "
                      f"pos={comps['pos']:.4f} vel={comps['vel']:.4f} "
                      f"rot={comps['rot']:.4f} omega={comps['omega']:.4f} "
                      f"val_pos={val_pos:.4f}", flush=True)
            step += 1
            if ckpt_every and save_ckpt and step % ckpt_every == 0:
                save_ckpt(f"s2_{step}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="artifacts/dataset_small.pt")
    ap.add_argument("--history", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--stage1-steps", type=int, default=3000)
    ap.add_argument("--stage2-steps", type=int, default=2000)
    ap.add_argument("--lambda-sig", type=float, default=0.02)
    ap.add_argument("--width-mult", type=float, default=1.0,
                    help="scale all internal widths, fractional ok "
                         "(1=9.5K, 1.25=14K, 1.5=20K, 2=34K params)")
    ap.add_argument("--action-mode", choices=["rotor_force", "ctbr"],
                    default="rotor_force")
    ap.add_argument("--probabilistic", action="store_true",
                    help="variance head + NLL loss (aleatoric uncertainty)")
    ap.add_argument("--learn-thrust", action="store_true",
                    help="(Fix A) learn the thrust-scale in the prober")
    ap.add_argument("--prober-hidden", type=int, default=None,
                    help="prober MLP width (default: 40*width_mult)")
    ap.add_argument("--pos-mode", choices=["zero", "relative"], default="zero",
                    help="how to hide absolute position from the nets (translation invariance)")
    ap.add_argument("--prober-inputs", choices=["full", "latent_action", "latent"],
                    default="full",
                    help="prober ablation: 'latent' = the paper-literal psi(latent); "
                         "'full' = (latent, running state, action) — our deviation")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="save an intermediate checkpoint every N steps (each stage); "
                         "0 = off. Files: <out>_s1_<step>.pt / <out>_s2_<step>.pt")
    ap.add_argument("--out", default="artifacts/skyjepa.pt")
    ap.add_argument("--init-from", default=None,
                    help="load encoders+predictor from a checkpoint and skip stage 1")
    ap.add_argument("--warm-start", default=None,
                    help="load the FULL model from a checkpoint, then run BOTH training "
                         "stages normally (basin-survival test; unlike --init-from, "
                         "stage 1 continues training from the loaded weights)")
    ap.add_argument("--ema-tau", type=float, default=0.0,
                    help="stage-1 EMA target encoder momentum (0 = off = paper setting); "
                         "e.g. 0.99. Targets come from the momentum copy (implicit stop-grad)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0,
                    help="global seed: set once before model build, so it seeds ALL "
                         "weight init (encoders+predictor+prober) AND data-shuffle order")
    ap.add_argument("--init-seed", type=int, default=None,
                    help="override the seed used for weight INIT only (defaults to --seed)")
    ap.add_argument("--data-seed", type=int, default=None,
                    help="re-seed the RNG for DATA-shuffle order only, after model build "
                         "(defaults to --seed's post-init state; set to decouple from init)")
    ap.add_argument("--init-scheme", choices=["default", "orthogonal"],
                    default="default",
                    help="stage-1 weight init: 'default' = PyTorch stock; 'orthogonal' = "
                         "He-conv + orthogonal-GRU-recurrent (spectral radius ~1 every seed)")
    args = ap.parse_args()

    # --seed alone reproduces the original behavior exactly (init-seed at the top,
    # no reseed before data). Passing --init-seed / --data-seed decouples the two
    # sources of run-to-run randomness so their effects can be separated.
    torch.manual_seed(args.seed if args.init_seed is None else args.init_seed)
    device = torch.device(args.device)
    ds = load_dataset(args.data, args.history, args.horizon, stride=args.stride)
    train = ds["train"]
    loader = DataLoader(train, batch_size=args.batch, shuffle=True, drop_last=True)
    print(f"train windows: {len(train)}  ({len(loader)} batches/epoch)")

    model = SkyJEPA(history=args.history, horizon=args.horizon,
                    width_mult=args.width_mult,
                    prober_hidden=args.prober_hidden,
                    action_mode=args.action_mode,
                    probabilistic=args.probabilistic,
                    learn_thrust=args.learn_thrust,
                    pos_mode=args.pos_mode,
                    prober_inputs=args.prober_inputs)
    apply_init_scheme(model, args.init_scheme)  # on CPU (orthogonal_ uses QR)
    model = model.to(device)
    model.fit_normalization(ds["raw"]["states"], ds["raw"]["actions"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: width_mult={args.width_mult}  params={n_params}")

    # Decouple data-shuffle order from weight init: after the model has consumed
    # RNG for its init, optionally reset the global RNG so the loader/val ordering
    # is governed by --data-seed instead of the post-init state.
    if args.data_seed is not None:
        torch.manual_seed(args.data_seed)

    # a fixed held-out batch (whole domains held out) for live val metrics
    val_X, val_A = next(iter(DataLoader(ds["val"], batch_size=512, shuffle=True)))
    val_X, val_A = val_X.to(device), val_A.to(device)
    print(f"train windows: {len(train)}  val windows: {len(ds['val'])}  "
          f"(split_by=domain)")

    cfg_dict = {"history": args.history, "horizon": args.horizon,
                "width_mult": args.width_mult,
                "prober_hidden": model.prober_hidden,
                "action_mode": args.action_mode,
                "probabilistic": args.probabilistic,
                "learn_thrust": args.learn_thrust,
                "pos_mode": args.pos_mode,
                "prober_inputs": args.prober_inputs}
    stem = args.out[:-3] if args.out.endswith(".pt") else args.out

    def save_ckpt(tag):
        path = f"{stem}_{tag}.pt"
        torch.save({"model": model.state_dict(), "config": cfg_dict}, path)
        print(f"    [ckpt] {path}", flush=True)

    if args.warm_start:
        sd = torch.load(args.warm_start, weights_only=False)["model"]
        model.load_state_dict(sd)
        print(f"warm-started FULL model from {args.warm_start}")

    if args.init_from:
        # reuse a trained stage 1 (encoders + predictor + normalization); skip it
        sd = torch.load(args.init_from, weights_only=False)["model"]
        sd = {k: v for k, v in sd.items() if not k.startswith("prober.")}
        model.load_state_dict(sd, strict=False)  # prober (re)initialized fresh
        print(f"loaded stage-1 (encoders+predictor+norm) from {args.init_from}")
    else:
        print("=== Stage 1: latent dynamics ===")
        train_stage1(model, loader, val_X, val_A, args.stage1_steps, device,
                     args.lambda_sig, args.ckpt_every, save_ckpt,
                     ema_tau=args.ema_tau)
    print("=== Stage 2: prober ===")
    train_stage2(model, loader, val_X, val_A, args.stage2_steps, device,
                 args.ckpt_every, save_ckpt)

    torch.save({"model": model.state_dict(), "config": cfg_dict}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
