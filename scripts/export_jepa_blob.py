"""Export a trained SkyJEPA checkpoint to a zero-dependency binary blob that the
Rust `rotor-rs` crate embeds via `include_bytes!` and parses without serde — so
the JEPA world model can run the MPPI rollouts in the browser (WASM).

The blob is self-describing and little-endian:

    magic           : 8 bytes  = b"JBLOB1\\n"
    action_mode     : u16 len + utf8 bytes
    history H       : u32
    horizon T       : u32
    dt, m_nominal, g: 3 x f64
    n_tensors       : u32
    per tensor:
        u16 name_len, name utf8
        u8  ndim, ndim x u32 dims
        prod(dims) x f32   (raw values, row-major)

Usage:  python scripts/export_jepa_blob.py [stem]      (default: skyjepa_ctbr_1x)
Reads   web-demo/jepa-rs/weights/<stem>.{safetensors,json}
Writes  web-demo/rotor-rs/assets/<stem>.jblob
"""
import json
import struct
import sys
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "web-demo" / "jepa-rs" / "weights"
ASSETS = ROOT / "web-demo" / "rotor-rs" / "assets"


def main() -> None:
    stem = sys.argv[1] if len(sys.argv) > 1 else "skyjepa_ctbr_1x"
    cfg = json.loads((WEIGHTS / f"{stem}.json").read_text())
    ASSETS.mkdir(parents=True, exist_ok=True)

    tensors = {}
    with safe_open(str(WEIGHTS / f"{stem}.safetensors"), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    buf = bytearray()
    buf += b"JBLOB1\n"
    am = cfg["action_mode"].encode("utf-8")
    buf += struct.pack("<H", len(am)) + am
    pm = cfg.get("pos_mode", "zero").encode("utf-8")
    buf += struct.pack("<H", len(pm)) + pm
    buf += struct.pack("<II", int(cfg["history"]), int(cfg["horizon"]))
    buf += struct.pack("<ddd", float(cfg["dt"]), float(cfg["m_nominal"]), float(cfg["g"]))
    buf += struct.pack("<I", len(tensors))

    for name, t in tensors.items():
        t = t.detach().to("cpu").float().contiguous()
        nb = name.encode("utf-8")
        buf += struct.pack("<H", len(nb)) + nb
        dims = list(t.shape)
        buf += struct.pack("<B", len(dims))
        for d in dims:
            buf += struct.pack("<I", int(d))
        flat = t.flatten().numpy().astype("<f4").tobytes()
        buf += flat

    out = ASSETS / f"{stem}.jblob"
    out.write_bytes(buf)
    n_params = sum(int(t.numel()) for t in tensors.values())
    print(f"wrote {out.relative_to(ROOT)} ({len(buf)} bytes, {len(tensors)} tensors, {n_params} params)")
    print(f"  action_mode={cfg['action_mode']} H={cfg['history']} T={cfg['horizon']} dt={cfg['dt']}")


if __name__ == "__main__":
    main()
