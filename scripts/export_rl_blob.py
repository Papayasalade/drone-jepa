"""Export the trained RL actor (the ~9.5K-param policy that flies) to a zero-dep
binary blob for the WASM `RlPolicy` drone. Only the actor is needed — the value
and log-std heads are training-only.

Format (little-endian):  magic b"RLB1\\n"
  u32 obs_dim, u32 hidden, u32 act_dim, u32 n_tensors
  per tensor: u16 name_len, name, u8 ndim, ndim*u32 dims, prod(dims)*f32

  python scripts/export_rl_blob.py artifacts/skyrl_ctbr.pt
"""
import struct
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "web-demo" / "rotor-rs" / "assets"

ckpt = sys.argv[1] if len(sys.argv) > 1 else "artifacts/skyrl_ctbr.pt"
stem = sys.argv[2] if len(sys.argv) > 2 else "skyrl_ctbr"
ck = torch.load(ROOT / ckpt, map_location="cpu", weights_only=False)
sd = ck["model"]
cfg = ck["config"]

# the actor: two-layer trunk + action-mean head
names = ["encoder.0.weight", "encoder.0.bias", "encoder.2.weight", "encoder.2.bias",
         "decoder_mean.weight", "decoder_mean.bias"]

buf = bytearray(b"RLB1\n")
buf += struct.pack("<III", int(cfg["obs_dim"]), int(cfg["hidden"]), int(cfg["act_dim"]))
buf += struct.pack("<I", len(names))
n_params = 0
for name in names:
    t = sd[name].detach().cpu().contiguous().float()
    nb = name.encode()
    buf += struct.pack("<H", len(nb)) + nb
    dims = list(t.shape)
    buf += struct.pack("<B", len(dims))
    for d in dims:
        buf += struct.pack("<I", int(d))
    buf += t.flatten().numpy().astype("<f4").tobytes()
    n_params += t.numel()

ASSETS.mkdir(parents=True, exist_ok=True)
out = ASSETS / f"{stem}.rlb"
out.write_bytes(buf)
print(f"wrote {out.relative_to(ROOT)}  ({len(buf)} bytes, {n_params} actor params, "
      f"obs={cfg['obs_dim']} hidden={cfg['hidden']} act={cfg['act_dim']})")
