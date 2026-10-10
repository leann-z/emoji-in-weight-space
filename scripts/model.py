import math
from dataclasses import dataclass

import torch

# N SIRENs stored as one dict of tensors with a leading network dimension:
#   W{i} (N, out, in), b{i} (N, out) for i = 0..depth.
# Layers 0..depth-1 compute sin(omega * (W h + b)); layer `depth` is linear.


@dataclass
class SirenConfig:
    width: int = 32
    depth: int = 3          # number of sine layers
    omega_first: float = 30.0
    omega_hidden: float = 30.0
    seed: int = 0


def layer_shapes(cfg):
    dims = [2] + [cfg.width] * cfg.depth + [3]
    return [(dims[i + 1], dims[i]) for i in range(len(dims) - 1)]


def param_names(cfg):
    names = []
    for i in range(cfg.depth + 1):
        names += [f"W{i}", f"b{i}"]
    return names


def num_params(cfg):
    return sum(o * i + o for o, i in layer_shapes(cfg))


def shared_init(cfg, n, device="cpu"):
    # Every network starts from the same draw. Without a shared start, hidden-unit
    # permutation symmetry makes weight vectors of different networks incomparable.
    g = torch.Generator().manual_seed(cfg.seed)
    params = {}
    for i, (fo, fi) in enumerate(layer_shapes(cfg)):
        if i == 0:
            bound = 1.0 / fi
        else:
            bound = math.sqrt(6.0 / fi) / cfg.omega_hidden
        w = (torch.rand(fo, fi, generator=g) * 2 - 1) * bound
        # PyTorch Linear default for biases: U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        b = (torch.rand(fo, generator=g) * 2 - 1) / math.sqrt(fi)
        params[f"W{i}"] = w.expand(n, fo, fi).clone().to(device)
        params[f"b{i}"] = b.expand(n, fo).clone().to(device)
    return params


def coord_grid(size, device="cpu"):
    """(size*size, 2) coordinates in [-1, 1], row-major, (x, y) order."""
    t = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    return torch.stack([xx.flatten(), yy.flatten()], dim=-1)


def forward(params, coords, cfg):
    """coords (P, 2) shared, or (N, P, 2). Returns (N, P, 3) in roughly [-1, 1]."""
    n = params["W0"].shape[0]
    h = coords if coords.dim() == 3 else coords.expand(n, *coords.shape)
    for i in range(cfg.depth + 1):
        h = torch.baddbmm(params[f"b{i}"].unsqueeze(1), h, params[f"W{i}"].transpose(1, 2))
        if i < cfg.depth:
            h = torch.sin((cfg.omega_first if i == 0 else cfg.omega_hidden) * h)
    return h


def to_image(out, size):
    """(N, P, 3) in [-1, 1] -> (N, size, size, 3) in [0, 1]."""
    return ((out + 1) / 2).clamp(0, 1).reshape(out.shape[0], size, size, 3)


def flatten(params, cfg):
    return torch.cat([params[k].reshape(params[k].shape[0], -1) for k in param_names(cfg)], dim=1)


def unflatten(flat, cfg):
    n = flat.shape[0]
    params, o = {}, 0
    for i, (fo, fi) in enumerate(layer_shapes(cfg)):
        params[f"W{i}"] = flat[:, o:o + fo * fi].reshape(n, fo, fi); o += fo * fi
        params[f"b{i}"] = flat[:, o:o + fo].reshape(n, fo); o += fo
    return params
