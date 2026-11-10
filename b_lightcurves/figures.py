"""Project B figures from b_results.json and saved run outputs.

Usage: python b_lightcurves/figures.py --robo6 /content/data/robo6 --cache CACHE --runs runs/b
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data import HZ, load_split  # noqa: E402
from train import run_name  # noqa: E402

SCORE_LABEL = {"msp": "Max softmax", "maxlogit": "Max logit", "energy": "Energy", "mahalanobis": "Mahalanobis",
               "knn": "kNN (k=10)"}


def fig_example(robo6, cache, out):
    mag, lab, ids = load_split(robo6, "train", cache)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for ax, c in zip(axes.ravel(), sorted(set(lab))):
        i = np.where(lab == c)[0][np.argmax((mag[lab == c] != 0).sum(1))]
        obs = np.nonzero(mag[i])[0]
        ax.plot(obs / HZ, mag[i][obs], ".", ms=1.5, color="#e0681f")
        ax.invert_yaxis()
        ax.set_title(f"{c}  (id {ids[i]})", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Magnitude")
    fig.suptitle("One RoBo6 training light curve per class (most observed points)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "b_example.png", dpi=150)


def fig_heatmap(res, out):
    t = res["loco"]["table"]
    rows, cols = list(t), res["protocol"]["scores"]
    m = np.array([[t[r]["auroc"][c]["mean"] for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(m, vmin=0.5, vmax=1.0, cmap="viridis")
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            s = t[r]["auroc"][c]
            ax.text(j, i, f"{s['mean']:.2f}\n±{s['std']:.2f}", ha="center", va="center", fontsize=8,
                    color="w" if s["mean"] < 0.8 else "k")
    ax.set_xticks(range(len(cols)), [SCORE_LABEL[c] for c in cols], fontsize=9)
    ax.set_yticks(range(len(rows)), rows, fontsize=9)
    ax.set_ylabel("Held-out (unknown) class")
    fig.colorbar(im, ax=ax, label="AUROC, unknown vs known")
    ax.set_title("Open-set detection, leave-one-class-out, full 1,000 s, mean ± std over seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "b_openset_heatmap.png", dpi=150)


def fig_budget(res, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if "closed_set" in res:
        b = res["closed_set"]["budget"]
        s = sorted(b, key=int)
        axes[0].errorbar([int(x) for x in s], [b[x]["accuracy"]["mean"] for x in s],
                         [b[x]["accuracy"]["std"] for x in s], marker="o", capsize=3, label="closed-set, 6 classes")
    if "loco" in res:
        b = res["loco"]["budget"]
        s = sorted(b, key=int)
        axes[0].errorbar([int(x) for x in s], [b[x]["known_accuracy"]["mean"] for x in s],
                         [b[x]["known_accuracy"]["std"] for x in s], marker="s", capsize=3,
                         label="known classes, leave-one-out models")
        for n in res["protocol"]["scores"]:
            best = n == res["loco"]["best_score"]
            axes[1].errorbar([int(x) for x in s], [b[x]["auroc"][n]["mean"] for x in s],
                             [b[x]["auroc"][n]["std"] for x in s] if best else None, marker="o" if best else None,
                             capsize=3, lw=2.2 if best else 1, alpha=1 if best else 0.6,
                             label=SCORE_LABEL[n] + (" (best)" if best else ""))
    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("Unknown-detection AUROC")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Seconds observed (log scale)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Observation budget: test light curves truncated to their first N seconds", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "b_budget.png", dpi=150)


def fig_embedding(res, runs, out):
    t, best = res["loco"]["table"], res["loco"]["best_score"]
    order = sorted(t, key=lambda c: t[c]["auroc"][best]["mean"])
    fold = order[len(order) // 2]
    o = np.load(runs / run_name(fold, 0) / "outputs.npz")
    e, lab = o["emb_1000"], o["test_labels"]
    e = e - e.mean(0)
    xy = e @ np.linalg.svd(e, full_matrices=False)[2][:2].T
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in sorted(set(lab)):
        k = lab == c
        if c == fold:
            ax.scatter(*xy[k].T, marker="x", s=22, c="k", label=f"{c} (never seen in training)", zorder=3)
        else:
            ax.scatter(*xy[k].T, s=6, alpha=0.6, label=c)
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"Test embeddings, model trained without {fold} (seed 0)\n"
                 f"median fold by {SCORE_LABEL[best]} AUROC", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "b_embedding.png", dpi=150)


def fig_confusion(res, out):
    cs = res["closed_set"]
    m, cl = np.array(cs["confusion_seed0"]), cs["classes"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(m, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(cl)):
        for j in range(len(cl)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if m[i, j] > 0.5 else "k")
    ax.set_xticks(range(len(cl)), cl, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(cl)), cl, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Closed-set confusion, full 1,000 s, seed 0 (rows sum to 1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "b_confusion.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robo6", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    res = json.loads((args.results / "b_results.json").read_text())
    fig_example(args.robo6, args.cache, args.out)
    if "closed_set" in res:
        fig_confusion(res, args.out)
    fig_budget(res, args.out)
    if "loco" in res:
        fig_heatmap(res, args.out)
        fig_embedding(res, args.runs, args.out)
    plt.close("all")
    print("wrote B figures to", args.out)


if __name__ == "__main__":
    main()
