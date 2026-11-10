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


VARIANT_LABEL = {"V0": "V0 single frame", "V1": "V1 raw two-frame", "V2": "V2 registered residual",
                 "V3": "V3 residual + star mask"}
COLOURS = {"V0": "#8c8c8c", "V1": "#5b8db8", "V2": "#d9a441", "V3": "#e0681f"}


def fig_ablation(res, out):
    vs = sorted(res["variants"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m in zip(axes, ["R", "F1"]):
        for i, v in enumerate(vs):
            s = res["variants"][v]["metrics"][m]
            ax.bar(i, s["mean"], yerr=s["std"], color=COLOURS[v], capsize=4, alpha=0.85)
            ax.scatter(i + np.linspace(-0.12, 0.12, len(s["values"])), s["values"], color="k", s=12, zorder=3)
        ax.set_xticks(range(len(vs)), vs)
        ax.set_ylim(0, 1)
        ax.set_title(f"{'Recall' if m == 'R' else 'F1'} at conf {res['protocol']['conf_operating_point']}, SET 1")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Bars: mean over seeds, error bars: std, dots: individual seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "a_ablation.png", dpi=150)


def fig_pr(res, out):
    fig, ax = plt.subplots(figsize=(6, 5))
    for v, d in sorted(res["variants"].items()):
        r, p = np.array(d["pr_curve"]["recall"]), d["pr_curve"]["precision"]
        ok = np.array([x is not None for x in p])
        ax.plot(r[ok], np.array(p, dtype=object)[ok].astype(float), color=COLOURS[v],
                label=f"{VARIANT_LABEL[v]} (AP {d['metrics']['AP']['mean']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Centroid-matched PR on SET 1, mean over seeds")
    fig.tight_layout()
    fig.savefig(out / "a_pr.png", dpi=150)


def bin_labels(b, unit):
    """Edges are stored with None for infinite ends."""
    e = b["edges"]
    out = []
    for i, n in enumerate(b["counts"]):
        lo = 0 if e[i] is None else e[i]
        out.append(f"{lo:.1f} to {e[i + 1]:.1f} {unit}\n(n={n})" if e[i + 1] is not None
                   else f"over {lo:.1f} {unit}\n(n={n})")
    return out


def grouped_recall(ax, res, name, labels, title):
    vs = sorted(res["variants"])
    w = 0.8 / len(vs)
    for k, v in enumerate(vs):
        cells = res["variants"][v]["breakdowns"][name]
        x = np.arange(len(cells)) + (k - (len(vs) - 1) / 2) * w
        ax.bar(x, [c["mean"] if c else 0 for c in cells], w, yerr=[c["std"] if c else 0 for c in cells],
               color=COLOURS[v], capsize=2, label=v)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def fig_breakdowns(res, out):
    b = res["bins"]
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped_recall(ax, res, "brightness", ["faint", "middle", "bright"],
                   "Recall by RSO brightness tercile (peak DN, 5x5 window)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "a_brightness.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    grouped_recall(axes[0], res, "speed", bin_labels(b["speed"], "px/frame"), "Recall by RSO speed tercile")
    grouped_recall(axes[1], res, "star_distance", bin_labels(b["star_distance"], "px"),
                   "Recall by distance to nearest catalogue star")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "a_speed_stars.png", dpi=150)


def fig_gif(data, results, out, session=None, max_frames=40, fps=3):
    import imageio.v2 as imageio
    tracks = json.loads((results / "a_tracks.json").read_text())
    gt = {g["key"]: g for g in json.loads((data / "gt.json").read_text())}
    session = session or max(tracks, key=lambda s: sum(len(gt[k]["rsos"]) for k in tracks[s]["frames"]))
    keys = tracks[session]["frames"][:max_frames]
    frames = []
    for t, key in enumerate(keys):
        img = cv2.imread(str(data / "V0" / "images" / "test" / f"{key}.png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2RGB)
        for r in gt[key]["rsos"]:
            cv2.circle(img, (int(r["x"] * 2), int(r["y"] * 2)), 9, (120, 200, 255), 1, cv2.LINE_AA)
        for pts in tracks[session]["tracks"].values():
            tail = [(int(pts[str(s)][0] * 2), int(pts[str(s)][1] * 2)) for s in range(t - 6, t + 1) if str(s) in pts]
            for a, (p, q) in enumerate(zip(tail[:-1], tail[1:])):
                fade = (a + 1) / len(tail)
                cv2.line(img, p, q, (int(255 * fade), int(160 * fade), 0), 2, cv2.LINE_AA)
            if str(t) in pts:
                cv2.circle(img, tail[-1], 3, (255, 160, 0), -1, cv2.LINE_AA)
        cv2.putText(img, f"{session}  frame {gt[key]['frame']}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)
        frames.append(img)
    imageio.mimsave(out / "a_tracks.gif", frames, duration=1000 / fps, loop=0)
    print("wrote", out / "a_tracks.gif", f"{(out / 'a_tracks.gif').stat().st_size / 1e6:.1f} MB", session)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    ap.add_argument("--only", choices=["method", "ablation", "pr", "breakdowns", "gif", "all"], default="all")
    ap.add_argument("--key", default=None, help="gt.json key of the frame for a_method.png")
    ap.add_argument("--session", default=None, help="SET 1 session for a_tracks.gif")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    todo = ["method", "ablation", "pr", "breakdowns", "gif"] if args.only == "all" else [args.only]
    res = json.loads((args.results / "a_results.json").read_text()) if set(todo) - {"method"} else None
    for name in todo:
        if name == "method":
            fig_method(args.data, args.out, args.key)
        elif name == "ablation":
            fig_ablation(res, args.out)
        elif name == "pr":
            fig_pr(res, args.out)
        elif name == "breakdowns":
            fig_breakdowns(res, args.out)
        elif name == "gif" and (args.results / "a_tracks.json").exists():
            fig_gif(args.data, args.results, args.out, args.session)
    plt.close("all")


if __name__ == "__main__":
    main()
