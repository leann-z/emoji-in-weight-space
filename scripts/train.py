import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

# Run any op MPS lacks on the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image

import model

ROOT = Path(__file__).resolve().parent.parent
SIZE = 64


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def psnr(pred, target):
    """Per-image PSNR for tensors in [0, 1], shape (N, ...)."""
    mse = ((pred - target) ** 2).flatten(1).mean(1)
    return -10 * torch.log10(mse.clamp_min(1e-10))


def render(params, cfg, size=SIZE, chunk=512):
    coords = model.coord_grid(size, params["W0"].device)
    n = params["W0"].shape[0]
    outs = []
    with torch.no_grad():
        for s in range(0, n, chunk):
            p = {k: v[s:s + chunk] for k, v in params.items()}
            outs.append(model.to_image(model.forward(p, coords, cfg), size))
    return torch.cat(outs)


def blend(params, i, j, t):
    """(1 - t) * params[i] + t * params[j], with i, j, t indexed per output network."""
    out = {}
    for k, v in params.items():
        tt = t.view(-1, *[1] * (v.dim() - 1))
        out[k] = (1 - tt) * v[i] + tt * v[j]
    return out


def fit(images, cfg, steps, lr, lr_min, chunk, interp, log_every=250):
    """images: (N, S, S, 3) in [0, 1]. Returns trained params
    """
    dev = device()
    n = images.shape[0]
    coords = model.coord_grid(SIZE, dev)
    target = images.to(dev).reshape(n, SIZE * SIZE, 3) * 2 - 1
    params = {k: v.requires_grad_() for k, v in model.shared_init(cfg, n, dev).items()}
    opt = torch.optim.Adam(params.values(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr_min)
    gen = torch.Generator().manual_seed(1)
    t0 = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        log = step % log_every == 0 or step == steps
        fit_psnr, mix_psnr = [], []
        # Sum of per-network losses: each network gets its own gradient, and
        # Adam is elementwise, so without interp this equals N independent optimisers.
        for s in range(0, n, chunk):
            idx = torch.arange(s, min(s + chunk, n), device=dev)
            sub = {k: v[idx] for k, v in params.items()}
            out = model.forward(sub, coords, cfg)
            ((out - target[idx]) ** 2).mean(dim=(1, 2)).sum().backward()
            if log:
                with torch.no_grad():
                    fit_psnr.append(psnr((out.clamp(-1, 1) + 1) / 2, (target[idx] + 1) / 2))
        if interp > 0:
            perm = torch.randperm(n, generator=gen).to(dev)
            tmix = torch.rand(n, generator=gen).to(dev)
            for s in range(0, n, chunk):
                idx = torch.arange(s, min(s + chunk, n), device=dev)
                j, t = perm[idx], tmix[idx]
                out = model.forward(blend(params, idx, j, t), coords, cfg)
                tt = t.view(-1, 1, 1)
                want = (1 - tt) * target[idx] + tt * target[j]
                (interp * ((out - want) ** 2).mean(dim=(1, 2)).sum()).backward()
                if log:
                    with torch.no_grad():
                        mix_psnr.append(psnr((out.clamp(-1, 1) + 1) / 2, (want + 1) / 2))
        opt.step()
        sched.step()
        if log:
            p = torch.cat(fit_psnr)
            msg = f"  step {step:5d}  fit PSNR {p.mean():.2f}  p5 {p.quantile(0.05):.2f}"
            if mix_psnr:
                msg += f"  blend PSNR {torch.cat(mix_psnr).mean():.2f}"
            print(f"{msg}  {time.time() - t0:.0f}s", flush=True)
    return {k: v.detach().cpu() for k, v in params.items()}


def midpoint_psnr(params, images, cfg, pairs=512, seed=2):
    """PSNR of the 50/50 weight blend's render against the 50/50 image blend."""
    g = torch.Generator().manual_seed(seed)
    n = images.shape[0]
    i = torch.randint(0, n, (pairs,), generator=g)
    j = torch.randint(0, n, (pairs,), generator=g)
    keep = i != j
    i, j = i[keep], j[keep]
    half = torch.full((len(i),), 0.5)
    mid = render({k: v.to(device()) for k, v in blend(params, i, j, half).items()}, cfg).cpu()
    mid = (mid * 255).round() / 255
    return psnr(mid, 0.5 * (images[i] + images[j])).numpy()


def recon_grid(orig, recon, idx, path):
    pad, rows = 4, []
    for r in range(4):
        row = []
        for i in idx[r * 4:(r + 1) * 4]:
            pair = np.concatenate([orig[i], np.full((SIZE, 2, 3), 200, np.uint8), recon[i]], axis=1)
            row.append(np.pad(pair, ((pad, pad), (pad, pad), (0, 0)), constant_values=255))
        rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).resize((grid.shape[1] * 3, grid.shape[0] * 3), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-min", type=float, default=1e-5)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--omega", type=float, default=30.0)
    ap.add_argument("--omega-hidden", type=float, default=30.0)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--interp", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/weights.pt")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    cfg = model.SirenConfig(width=args.width, depth=args.depth,
                            omega_first=args.omega, omega_hidden=args.omega_hidden)
    data = np.load(ROOT / "results" / "data.npz")
    images_u8 = data["images"]
    if args.limit:
        rng = np.random.default_rng(0)
        images_u8 = images_u8[np.sort(rng.choice(len(images_u8), args.limit, replace=False))]
    images = torch.from_numpy(images_u8).float() / 255
    print(f"{len(images)} networks, {model.num_params(cfg)} params each, device {device()}")

    t0 = time.time()
    params = fit(images, cfg, args.steps, args.lr, args.lr_min, args.chunk, args.interp)
    train_time = time.time() - t0

    # Evaluate the stored weights, quantised to 8-bit pixels like the originals
    recon = render({k: v.to(device()) for k, v in params.items()}, cfg).cpu()
    recon_u8 = (recon * 255).round().byte().numpy()
    p = psnr(torch.from_numpy(recon_u8).float() / 255, images).numpy()
    print(f"mean PSNR {p.mean():.2f} dB, 5th percentile {np.percentile(p, 5):.2f} dB, "
          f"min {p.min():.2f} dB, train time {train_time:.0f}s")
    mid = midpoint_psnr(params, images, cfg)
    print(f"weight midpoint vs pixel cross-fade: mean PSNR {mid.mean():.2f} dB")

    if args.limit:
        return
    torch.save({"params": params, "config": asdict(cfg), "psnr": torch.from_numpy(p)}, ROOT / args.out)
    metrics = {"n": len(p), "params_per_net": model.num_params(cfg), "steps": args.steps,
               "lr": args.lr, "lr_min": args.lr_min, "interp": args.interp, "config": asdict(cfg),
               "psnr_mean": float(p.mean()), "psnr_p5": float(np.percentile(p, 5)),
               "psnr_min": float(p.min()), "midpoint_vs_crossfade_psnr": float(mid.mean()),
               "train_seconds": round(train_time)}
    (ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    idx = np.random.default_rng(1).choice(len(p), 16, replace=False)
    (ROOT / "media").mkdir(exist_ok=True)
    recon_grid(images_u8, recon_u8, idx, ROOT / "media" / "reconstructions.png")
    print("wrote results/weights.pt, results/metrics.json, media/reconstructions.png")


if __name__ == "__main__":
    main()
