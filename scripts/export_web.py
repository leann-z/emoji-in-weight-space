import json
import struct
from pathlib import Path

import numpy as np
import torch

import model
from embed import align

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "atlas.bin"


def unit_square(xy):
    # Uniform scale so the page shows the layout with its real aspect ratio
    span = xy.max(0) - xy.min(0)
    return (xy - xy.min(0)) / span.max() + (1 - span / span.max()) / 2


def main():
    ck = torch.load(ROOT / "results" / "weights.pt", weights_only=False)
    cfg = model.SirenConfig(**ck["config"])
    data = np.load(ROOT / "results" / "data.npz")
    layout = np.load(ROOT / "results" / "layout.npz")
    flat = model.flatten(ck["params"], cfg).numpy()
    w16 = flat.astype("<f2")

    xy = unit_square(layout["weight_xy"])
    pxy = unit_square(align(layout["pixel_xy"], layout["weight_xy"]))
    header = {
        "n": int(flat.shape[0]),
        "params_per_net": int(flat.shape[1]),
        "config": ck["config"],
        "layers": [list(s) for s in model.layer_shapes(cfg)],
        "names": [str(s) for s in data["names"]],
        "groups": [str(s) for s in data["groups"]],
        "subgroups": [str(s) for s in data["subgroups"]],
        "codepoints": [str(s) for s in data["codepoints"]],
        "psnr": [round(float(p), 2) for p in ck["psnr"]],
        "xy": [[round(float(a), 5), round(float(b), 5)] for a, b in xy],
        "pxy": [[round(float(a), 5), round(float(b), 5)] for a, b in pxy],
    }
    js = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    js += b" " * ((-(8 + len(js))) % 8)
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(b"WSA1")
        f.write(struct.pack("<I", len(js)))
        f.write(js)
        f.write(w16.tobytes())
    err = np.abs(w16.astype(np.float32) - flat).max()
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB), "
          f"max float16 rounding error {err:.2e}")


if __name__ == "__main__":
    main()
