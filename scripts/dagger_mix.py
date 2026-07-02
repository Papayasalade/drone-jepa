"""Mix a base trajectory dataset with DAgger-collected records (generalized
version of the E8 mixer): chops the base 200-step trajs into 40-step blocks
(lossless) and appends the collector's 40-step records.

  .venv/bin/python scripts/dagger_mix.py BASE.pt DAGGER.bin OUT.pt
"""
from __future__ import annotations

import struct
import sys

import numpy as np
import torch


def load_bin(path):
    buf = open(path, "rb").read()
    n, steps, n_act = struct.unpack_from("<III", buf, 0)
    off = 12
    ns = n * steps * 18
    states = np.frombuffer(buf, "<f4", ns, off).reshape(n, steps, 18).copy(); off += ns * 4
    na = n * steps * n_act
    actions = np.frombuffer(buf, "<f4", na, off).reshape(n, steps, n_act).copy()
    return torch.from_numpy(states), torch.from_numpy(actions)


def main():
    base_pt, dagger_bin, out = sys.argv[1:4]
    d = torch.load(base_pt, weights_only=False)
    S, A, dom = d["states"], d["actions"], d["domain"]
    N, T = S.shape[0], S.shape[1]
    assert T % 40 == 0, f"base steps {T} not divisible by 40"
    k = T // 40
    S40 = S.reshape(N * k, 40, 18).contiguous()
    A40 = A.reshape(N * k, 40, A.shape[-1]).contiguous()
    dom40 = dom.repeat_interleave(k)

    DS, DA = load_bin(dagger_bin)
    # each ~20-record group gets its own pseudo-domain, above existing ids,
    # so the by-domain train/val split keeps functioning
    dd = torch.arange(len(DS)) // 20 + int(dom40.max()) + 1000
    torch.save({"states": torch.cat([S40, DS]), "actions": torch.cat([A40, DA]),
                "domain": torch.cat([dom40, dd]), "dt": d["dt"]}, out)
    print(f"{out}: base {tuple(S40.shape)} + dagger {tuple(DS.shape)}")


if __name__ == "__main__":
    main()
