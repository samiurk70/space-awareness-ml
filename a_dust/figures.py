"""Project A figures. Reads prepared datasets and results, writes results/figures/*.

Usage: python a_dust/figures.py --data /content/data/dust_yolo --only method
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_rgb(path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)[..., ::-1]


def pick_frame(gt):
    """A test frame registered from stars, with the most GT RSOs."""
    cands = [g for g in gt if g["split"] == "test" and g["registration"] == "stars" and g["rsos"]]
    return max(cands, key=lambda g: (len(g["rsos"]), g["key"]))


def fig_method(data, out, key=None):
    gt = json.loads((data / "gt.json").read_text())
    g = next(x for x in gt if x["key"] == key) if key else pick_frame(gt)
    v1 = load_rgb(data / "V1" / "images" / "test" / f"{g['key']}.png")
    v3 = load_rgb(data / "V3" / "images" / "test" / f"{g['key']}.png")
    prev, cur, res, mask = v1[..., 0], v1[..., 1], v3[..., 1], v3[..., 2]
    overlay = np.dstack([prev, cur, prev])

    panels = [(cur, "gray", "Raw frame t"),
              (overlay, None, "t-1 (magenta) over t (green)"),
              (res, "gray", "Star-registered residual"),
              (mask, "gray", "Catalogue star mask")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (img, cmap, title) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        for r in g["rsos"]:
            ax.add_patch(plt.Circle((r["x"], r["y"]), 7, fill=False, color="#ffb000", lw=1.2))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{g['session']}, frame {g['frame']}. Amber circles: ground-truth RSOs", fontsize=11)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "a_method.png", dpi=150, bbox_inches="tight")
    print("wrote", out / "a_method.png", "from", g["key"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ap.add_argument("--only", choices=["method"], default="method")
    ap.add_argument("--key", default=None, help="gt.json key of the frame for a_method.png")
    args = ap.parse_args()
    fig_method(args.data, args.out, args.key)


if __name__ == "__main__":
    main()
