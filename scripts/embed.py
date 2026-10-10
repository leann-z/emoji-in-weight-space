import json
import time
from pathlib import Path

import numpy as np
import torch
import umap
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import model

ROOT = Path(__file__).resolve().parent.parent
SEED = 0
PCA_DIMS = 50
UMAP_KW = dict(n_neighbors=50, min_dist=0.5, metric="cosine", random_state=SEED)
K = 10


def embed(x):
    x = StandardScaler().fit_transform(x)
    x = PCA(PCA_DIMS, random_state=SEED).fit_transform(x)
    xy = umap.UMAP(n_components=2, **UMAP_KW).fit_transform(x)
    return x, xy.astype(np.float32)


def normalise(xy):
    lo, hi = xy.min(0), xy.max(0)
    return (xy - lo) / (hi - lo)


def snap_to_grid(xy):
    # Exact assignment minimizing total squared displacement
    n = len(xy)
    side = int(np.ceil(np.sqrt(n)))
    pts = normalise(xy)
    pts[:, 1] = 1 - pts[:, 1] 
    t = (np.arange(side) + 0.5) / side
    gx, gy = np.meshgrid(t, t)
    cells = np.stack([gx.ravel(), gy.ravel()], 1)
    cost = ((pts[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    grid = np.zeros((n, 2), np.int32)
    grid[rows, 0] = cols[rows] % side   # column
    grid[rows, 1] = cols[rows] // side  # row
    return grid, side


def align(src, ref):
    # UMAP's orientation is arbitrary, so rotate/reflect src onto ref before
    a, b = src - src.mean(0), ref - ref.mean(0)
    r, _ = orthogonal_procrustes(a, b)
    return a @ r


def knn_overlap(a, b, k=K):
    ia = NearestNeighbors(n_neighbors=k + 1).fit(a).kneighbors(a, return_distance=False)[:, 1:]
    ib = NearestNeighbors(n_neighbors=k + 1).fit(b).kneighbors(b, return_distance=False)[:, 1:]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ia, ib)]))


def main():
    ck = torch.load(ROOT / "results" / "weights.pt", weights_only=False)
    cfg = model.SirenConfig(**ck["config"])
    weights = model.flatten(ck["params"], cfg).numpy().astype(np.float64)
    pixels = np.load(ROOT / "results" / "data.npz")["images"].reshape(len(weights), -1) / 255.0

    t0 = time.time()
    w_pca, w_xy = embed(weights)
    p_pca, p_xy = embed(pixels)
    grid, side = snap_to_grid(w_xy)
    print(f"embedded and snapped {len(weights)} points to a {side}x{side} grid in {time.time() - t0:.0f}s")

    stats = {
        "pca_dims": PCA_DIMS, "umap": UMAP_KW, "k": K, "grid_side": side,
        "knn_overlap_weight_vs_pixel_2d": knn_overlap(w_xy, p_xy),
        "knn_overlap_weight_vs_pixel_pca50": knn_overlap(w_pca, p_pca),
        # Jitter breaks distance ties between cells of the integer grid
        "knn_overlap_umap_vs_grid": knn_overlap(w_xy, grid.astype(np.float32)
                                                + np.random.default_rng(0).normal(0, 1e-3, grid.shape)),
        "knn_overlap_random_baseline": K / (len(weights) - 1),
    }
    np.savez(ROOT / "results" / "layout.npz", weight_xy=w_xy, pixel_xy=p_xy, grid=grid, side=side)
    (ROOT / "results" / "layout.json").write_text(json.dumps(stats, indent=2))
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
