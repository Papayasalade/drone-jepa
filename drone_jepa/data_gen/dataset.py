"""Windowed dataset: slice trajectories into (X, A) windows of length H+T.

Each item is a contiguous window of L = history + horizon steps:
  X : (L, 18) physical states
  A : (L, 4)  rotor-force actions, with A[k] taking state k -> state k+1
The model reads index t = H-1 as "now" and predicts steps H .. H+T-1.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    def __init__(self, states: torch.Tensor, actions: torch.Tensor,
                 history: int = 10, horizon: int = 20, stride: int = 1):
        assert states.shape[:2] == actions.shape[:2]
        self.states = states
        self.actions = actions
        self.L = history + horizon
        # build a flat index of (traj, start) windows
        N, S = states.shape[0], states.shape[1]
        idx = []
        for n in range(N):
            for s in range(0, S - self.L + 1, stride):
                idx.append((n, s))
        self.index = idx

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        n, s = self.index[i]
        X = self.states[n, s:s + self.L]
        A = self.actions[n, s:s + self.L]
        return X, A


def load_dataset(path: str, history: int = 10, horizon: int = 20,
                 stride: int = 1, val_frac: float = 0.1, test_frac: float = 0.1,
                 split_by: str = "domain"):
    """Load a collected .pt file and split 80/10/10.

    split_by="domain" (default) holds out WHOLE domains, so val/test measure
    generalization to unseen dynamics — not just unseen trajectories of domains
    already in the train set. This is the honest overfitting check for a
    domain-randomized world model. split_by="trajectory" is the looser split.
    """
    data = torch.load(path, weights_only=False)
    states, actions = data["states"], data["actions"]
    N = states.shape[0]
    g = torch.Generator().manual_seed(0)

    if split_by == "domain" and "domain" in data:
        domains = data["domain"]
        uniq = torch.unique(domains)
        dperm = uniq[torch.randperm(len(uniq), generator=g)]
        n_test_d = max(1, int(len(uniq) * test_frac))
        n_val_d = max(1, int(len(uniq) * val_frac))
        test_d = set(dperm[:n_test_d].tolist())
        val_d = set(dperm[n_test_d:n_test_d + n_val_d].tolist())
        test_i = torch.tensor([i for i in range(N) if domains[i].item() in test_d])
        val_i = torch.tensor([i for i in range(N) if domains[i].item() in val_d])
        seen = test_d | val_d
        train_i = torch.tensor([i for i in range(N) if domains[i].item() not in seen])
    else:
        perm = torch.randperm(N, generator=g)
        n_test, n_val = int(N * test_frac), int(N * val_frac)
        test_i = perm[:n_test]
        val_i = perm[n_test:n_test + n_val]
        train_i = perm[n_test + n_val:]

    def make(ix, stride):
        return WindowDataset(states[ix], actions[ix], history, horizon, stride)

    return {
        "train": make(train_i, stride),
        "val": make(val_i, max(stride, horizon)),  # sparser val windows
        "test": make(test_i, max(stride, horizon)),
        "raw": data,
    }
