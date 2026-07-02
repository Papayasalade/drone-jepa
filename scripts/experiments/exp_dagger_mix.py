"""E8 mixer: full-mix 200-step trajs -> 40-step blocks (lossless reshape), plus
the DAgger-collected 40-step records, written as standard {states,actions,domain,dt}
datasets. Also writes the chopped-only control.

  .venv/bin/python e8_mix.py  (from repo root)
"""
import struct, sys
import numpy as np
import torch

FM = "artifacts/skyjepa_rotor_mix_recovery/dataset.pt"
DAGGER_BIN = "artifacts/dagger_s0.bin"


def load_bin(path):
    buf = open(path, "rb").read()
    n, steps, n_act = struct.unpack_from("<III", buf, 0)
    off = 12
    ns = n * steps * 18
    states = np.frombuffer(buf, "<f4", ns, off).reshape(n, steps, 18).copy(); off += ns * 4
    na = n * steps * n_act
    actions = np.frombuffer(buf, "<f4", na, off).reshape(n, steps, n_act).copy()
    return torch.from_numpy(states), torch.from_numpy(actions)


d = torch.load(FM, weights_only=False)
S, A, dom = d["states"], d["actions"], d["domain"]
N, T = S.shape[0], S.shape[1]
assert T % 40 == 0
k = T // 40
S40 = S.reshape(N * k, 40, 18).contiguous()
A40 = A.reshape(N * k, 40, 4).contiguous()
dom40 = dom.repeat_interleave(k)
torch.save({"states": S40, "actions": A40, "domain": dom40, "dt": d["dt"]},
           "artifacts/e8_fullmix40.pt")
print(f"control: {S40.shape}")

DS, DA = load_bin(DAGGER_BIN)
# dagger data = one fixed drone; give each record its own pseudo-domain far above
# the existing ids so the domain split keeps working
dd = torch.arange(len(DS)) // 20 + int(dom40.max()) + 1000
S_mix = torch.cat([S40, DS]); A_mix = torch.cat([A40, DA])
dom_mix = torch.cat([dom40, dd])
torch.save({"states": S_mix, "actions": A_mix, "domain": dom_mix, "dt": d["dt"]},
           "artifacts/e8_fullmix40_dagger.pt")
print(f"dagger-mixed: {S_mix.shape}  ({len(DS)} dagger records)")
