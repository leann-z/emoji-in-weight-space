import re
import subprocess
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
NOTO = RAW / "noto-emoji"
NOTO_URL = "https://github.com/googlefonts/noto-emoji.git"
NOTO_DIRS = ["2D/png/128", "third_party/region-flags/png"]
EMOJI_TEST_URL = "https://unicode.org/Public/18.0.0/emoji/emoji-test.txt"
EMOJI_TEST = RAW / "emoji-test-18.0.txt"
SIZE = 64
SKIN_TONES = {0x1F3FB, 0x1F3FC, 0x1F3FD, 0x1F3FE, 0x1F3FF}


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    if not NOTO.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none",
                        "--sparse", NOTO_URL, str(NOTO)], check=True)
    subprocess.run(["git", "-C", str(NOTO), "sparse-checkout", "set", *NOTO_DIRS], check=True)
    if not EMOJI_TEST.exists():
        urllib.request.urlretrieve(EMOJI_TEST_URL, EMOJI_TEST)
    sha = subprocess.run(["git", "-C", str(NOTO), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return sha


def parse_emoji_test():
    """Map codepoint tuple (FE0F stripped) -> (name, group, subgroup)."""
    table = {}
    group = subgroup = None
    line_re = re.compile(r"^([0-9A-F ]+?)\s*;\s*(\S+)\s*#\s*\S+\s+E\d+\.\d+\s+(.*)$")
    for line in EMOJI_TEST.read_text(encoding="utf-8").splitlines():
        if line.startswith("# group:"):
            group = line.split(":", 1)[1].strip()
        elif line.startswith("# subgroup:"):
            subgroup = line.split(":", 1)[1].strip()
        elif line and not line.startswith("#"):
            m = line_re.match(line)
            if not m or m.group(2) != "fully-qualified":
                continue
            cps = tuple(int(c, 16) for c in m.group(1).split() if c != "FE0F")
            table.setdefault(cps, (m.group(3).strip(), group, subgroup))
    return table


def flag_codepoints(stem):
    """'US' -> regional indicators, 'GB-ENG' -> black flag + tag sequence."""
    if "-" not in stem:
        return tuple(0x1F1E6 + ord(c) - ord("A") for c in stem)
    tag = stem.replace("-", "").lower()
    return (0x1F3F4, *(0xE0000 + ord(c) for c in tag), 0xE007F)


def to_rgb64(path, is_flag):
    im = Image.open(path).convert("RGBA")
    if is_flag:
        # Region flags are rectangular; fit inside the same padding Noto uses for emoji.
        w, h = im.size
        s = 112 / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        canvas = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        canvas.paste(im, ((128 - im.width) // 2, (128 - im.height) // 2))
        im = canvas
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, im).convert("RGB")
    return np.asarray(rgb.resize((SIZE, SIZE), Image.LANCZOS), dtype=np.uint8)


def build():
    sha = download()
    table = parse_emoji_test()
    items = []
    for p in sorted((NOTO / "2D/png/128").glob("emoji_u*.png")):
        cps = tuple(int(c, 16) for c in p.stem[len("emoji_u"):].split("_"))
        if SKIN_TONES & set(cps) or cps not in table:
            continue
        name, group, sub = table[cps]
        if group == "Component":
            continue
        items.append((p, False, cps, name, group, sub))
    for p in sorted((NOTO / "third_party/region-flags/png").glob("*.png")):
        cps = flag_codepoints(p.stem)
        if cps in table:
            name, group, sub = table[cps]
            items.append((p, True, cps, name, group, sub))

    images = np.stack([to_rgb64(p, f) for p, f, *_ in items])
    names = np.array([it[3] for it in items])
    groups = np.array([it[4] for it in items])
    subgroups = np.array([it[5] for it in items])
    codepoints = np.array([" ".join(f"{c:04X}" for c in it[2]) for it in items])
    out = ROOT / "results" / "data.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, images=images, names=names, groups=groups,
                        subgroups=subgroups, codepoints=codepoints, noto_sha=sha)

    print(f"noto-emoji commit {sha[:10]}")
    print(f"{len(images)} images, {len(set(subgroups))} subgroups")
    for g, n in Counter(groups).most_common():
        print(f"  {n:5d}  {g}")
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
