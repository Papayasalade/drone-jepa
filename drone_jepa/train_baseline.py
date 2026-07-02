"""Train the direct autoregressive baseline (matched-capacity control)."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from .data_gen.dataset import load_dataset
from .model.baseline import DirectBaseline
from .model.losses import physical_loss
from .train import lr_at, set_lr


def train_baseline(model, loader, steps, device):
    opt = torch.optim.Adam(model.parameters(), lr=0.0, weight_decay=1e-5)
    H, T = model.H, model.T
    model.train()
    step = 0
    while step < steps:
        for X, A in loader:
            if step >= steps:
                break
            X, A = X.to(device), A.to(device)
            set_lr(opt, lr_at(step, steps))
            pred = model.rollout(X, A)
            target = X[:, H:H + T]
            loss, comps = physical_loss(pred, target, std=model.state_std)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            if step % max(1, steps // 20) == 0:
                print(f"  [base {step:5d}] loss={loss.item():.4f} "
                      f"pos={comps['pos']:.4f} vel={comps['vel']:.4f} "
                      f"rot={comps['rot']:.4f} omega={comps['omega']:.4f}", flush=True)
            step += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="artifacts/dataset_small.pt")
    ap.add_argument("--history", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="artifacts/baseline.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    ds = load_dataset(args.data, args.history, args.horizon, stride=args.stride)
    loader = DataLoader(ds["train"], batch_size=args.batch, shuffle=True, drop_last=True)
    model = DirectBaseline(args.history, args.horizon).to(device)
    model.fit_normalization(ds["raw"]["states"], ds["raw"]["actions"])
    print(f"train windows: {len(ds['train'])}")
    print("=== Baseline: direct autoregressive next-state ===")
    train_baseline(model, loader, args.steps, device)
    torch.save({"model": model.state_dict(),
                "config": {"history": args.history, "horizon": args.horizon}}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
