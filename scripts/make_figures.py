import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

import model
from embed import align
from train import device

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "media"

# Poster and GIF: white paper, so each network's white background disappears into it.
PAPER = (255, 255, 255)
LABEL = (82, 81, 78)
SURFACE = "#1a1a19"
INK, INK_2, MUTED = "#ffffff", "#c3c2b7", "#898781"

# Smileys and People are merged, as on most emoji keyboards, which leaves eight groups.
GROUPS = ["Smileys & People", "Animals & Nature", "Food & Drink", "Activities",
          "Travel & Places", "Objects", "Symbols", "Flags"]
# Hues follow the content where they can (yellow faces, green nature, red flags).
# Eight hues cannot all be told apart in a scatter, so the figure also has one
# small panel per group with that group highlighted.
GROUP_COLORS = ["#c98500", "#008300", "#d95926", "#d55181",
                "#3987e5", "#9085e9", "#199e70", "#e66767"]
FADED = "#383835"

# Morph pairs, picked by name after looking at the map: far apart in weight space
# and visually unlike each other.
MORPH_PAIRS = [
    ("grinning face", "keycap: 7"),
    ("skull", "flag: Brazil"),
    ("purple heart", "fire"),
    ("red apple", "globe showing Americas"),
]


def load():
    ck = torch.load(ROOT / "results" / "weights.pt", weights_only=False)
    cfg = model.SirenConfig(**ck["config"])
    data = np.load(ROOT / "results" / "data.npz")
    layout = np.load(ROOT / "results" / "layout.npz")
    return ck, cfg, data, layout


def merged_groups(groups):
    return np.array(["Smileys & People" if g in ("Smileys & Emotion", "People & Body") else g
                     for g in groups])


def render_u8(params, cfg, size, chunk=256):
    dev = device()
    coords = model.coord_grid(size, dev)
    n = params["W0"].shape[0]
    out = []
    with torch.no_grad():
        for s in range(0, n, chunk):
            p = {k: v[s:s + chunk].to(dev) for k, v in params.items()}
            img = model.to_image(model.forward(p, coords, cfg), size)
            out.append((img * 255).round().byte().cpu())
    return torch.cat(out).numpy()


def poster(scale=1):
    ck, cfg, data, layout = load()
    cell = 64 * scale
    gap, margin = 4 * scale, 48 * scale
    side = int(layout["side"])
    # Each network is evaluated at the poster's cell size rather than upscaled from 64 px.
    thumbs = render_u8(ck["params"], cfg, cell)
    size = margin * 2 + side * cell + (side - 1) * gap
    canvas = Image.new("RGB", (size, size), PAPER)
    for img, (c, r) in zip(thumbs, layout["grid"]):
        canvas.paste(Image.fromarray(img), (margin + c * (cell + gap), margin + r * (cell + gap)))
    name = "atlas.png" if scale == 1 else f"atlas@{scale}x.png"
    canvas.save(MEDIA / name, optimize=True)
    print(f"wrote media/{name} ({size}x{size})")


def _blank(ax):
    ax.set_facecolor(SURFACE)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def compare():
    _, _, data, layout = load()
    groups = merged_groups(data["groups"])
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    fig = plt.figure(figsize=(13, 10.4), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 8, height_ratios=[5.2, 1, 1], hspace=0.12, wspace=0.06,
                          left=0.035, right=0.985, top=0.95, bottom=0.075)
    panels = [("Weight space", layout["weight_xy"]),
              ("Pixel space", align(layout["pixel_xy"], layout["weight_xy"]))]
    for k, (title, xy) in enumerate(panels):
        ax = fig.add_subplot(gs[0, 4 * k:4 * k + 4])
        _blank(ax)
        for g, col in zip(GROUPS, GROUP_COLORS):
            m = groups == g
            ax.scatter(xy[m, 0], xy[m, 1], s=8, c=col, linewidths=0.3, edgecolors=SURFACE)
        ax.set_title(title, color=INK, fontsize=13, loc="left", pad=6)
        ax.set_aspect("equal", adjustable="datalim")
        for j, (g, col) in enumerate(zip(GROUPS, GROUP_COLORS)):
            sm = fig.add_subplot(gs[1 + k, j])
            _blank(sm)
            m = groups == g
            sm.scatter(xy[~m, 0], xy[~m, 1], s=0.6, c=FADED, linewidths=0)
            sm.scatter(xy[m, 0], xy[m, 1], s=1.6, c=col, linewidths=0)
            sm.set_aspect("equal", adjustable="datalim")
            if k == 0:
                sm.set_title(g, color=INK_2, fontsize=8.5, pad=3)
            if j == 0:
                sm.set_ylabel(title.split()[0], color=MUTED, fontsize=8.5)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=6, mfc=c, mec="none") for c in GROUP_COLORS]
    leg = fig.legend(handles, GROUPS, loc="lower center", ncol=8, frameon=False, fontsize=9.5,
                     handletextpad=0.3, columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.text(0.5, 0.045, "Each point is one emoji. Weight space: UMAP of its network's weights. "
             "Pixel space: UMAP of its raw pixels, same procedure.", ha="center", color=MUTED, fontsize=9)
    fig.savefig(MEDIA / "weight_vs_pixel.png", dpi=150, facecolor=SURFACE)
    print("wrote media/weight_vs_pixel.png")


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * t)


def morph(frames=30, cell=160, hold=8):
    ck, cfg, data, layout = load()
    names = list(data["names"])
    flat = model.flatten(ck["params"], cfg)
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 12)
    pad, label_h = 14, 40
    w = 2 * cell + 3 * pad
    h = 2 * (cell + label_h) + 3 * pad
    ts = ease(np.linspace(0, 1, frames))
    seqs = []
    for a, b in MORPH_PAIRS:
        ia, ib = names.index(a), names.index(b)
        mix = torch.stack([(1 - t) * flat[ia] + t * flat[ib] for t in ts]).float()
        seqs.append(render_u8(model.unflatten(mix, cfg), cfg, cell))
    order = [0] * hold + list(range(frames)) + [frames - 1] * hold + list(range(frames - 1, -1, -1))
    out = []
    for f in order:
        im = Image.new("RGB", (w, h), PAPER)
        d = ImageDraw.Draw(im)
        for k, (a, b) in enumerate(MORPH_PAIRS):
            x = pad + (k % 2) * (cell + pad)
            y = pad + (k // 2) * (cell + label_h + pad)
            im.paste(Image.fromarray(seqs[k][f]), (x, y))
            d.text((x + 2, y + cell + 5), a, fill=LABEL, font=font)
            d.text((x + 2, y + cell + 22), f"→ {b}", fill=LABEL, font=font)
        out.append(im.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    out[0].save(MEDIA / "morph.gif", save_all=True, append_images=out[1:], duration=50,
                loop=0, optimize=True, disposal=1)
    mb = (MEDIA / "morph.gif").stat().st_size / 1e6
    print(f"wrote media/morph.gif ({len(out)} frames, {mb:.2f} MB)")


def main():
    MEDIA.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["poster", "compare", "morph"]
    if "poster" in which:
        poster(1)
        poster(2)
    if "compare" in which:
        compare()
    if "morph" in which:
        morph()


if __name__ == "__main__":
    main()
