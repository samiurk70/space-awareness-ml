"""Project B metrics from saved run outputs: closed-set, leave-one-class-out open-set, observation budget.

Unknown scores (higher = more likely unknown): negative max softmax, negative max logit, energy
(-logsumexp of logits), Mahalanobis distance to the nearest class mean (shared covariance, fitted on train
embeddings), and 1 minus the cosine similarity to the 10th nearest train embedding.
FPR95: fraction of unknown test samples accepted as known when 95% of known test samples are accepted.

Usage: python b_lightcurves/openset.py --runs runs/b --results results
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

SCORES = ["msp", "maxlogit", "energy", "mahalanobis", "knn"]
KNN_K = 10
HIST_BINS = 30
PAPER = [
    {"method": "ResNet (RoBo6 paper)", "accuracy": 0.694, "macro_f1": 0.600, "source": "same split, their implementation"},
    {"method": "Astroconformer (RoBo6 paper, best)", "accuracy": 0.725, "macro_f1": 0.684,
     "source": "same split, their implementation"},
]


def unknown_scores(logits, emb, train_emb, train_y):
    k = logits.shape[1]
    s = {"msp": -softmax(logits, 1).max(1), "maxlogit": -logits.max(1), "energy": -logsumexp(logits, 1)}
    means = np.stack([train_emb[train_y == c].mean(0) for c in range(k)])
    centred = train_emb - means[train_y]
    cov = centred.T @ centred / len(centred)
    prec = np.linalg.inv(cov + 1e-4 * np.trace(cov) / len(cov) * np.eye(len(cov)))
    diff = emb[:, None, :] - means[None]
    s["mahalanobis"] = np.einsum("ncd,de,nce->nc", diff, prec, diff).min(1)
    a = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-12)
    b = train_emb / np.linalg.norm(train_emb, axis=1, keepdims=True).clip(1e-12)
    kth_sim = -np.partition(-(a @ b.T), KNN_K - 1, axis=1)[:, KNN_K - 1]
    s["knn"] = 1 - kth_sim
    return s


def fpr95(known, unknown):
    return float((unknown <= np.percentile(known, 95)).mean())


def summarise(vals):
    vals = [float(v) for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "values": vals}


def evaluate_run(path):
    o = np.load(path / "outputs.npz")
    known, holdout = list(o["known"]), str(o["holdout"])
    labels = o["test_labels"]
    is_known = np.isin(labels, known)
    y = np.array([known.index(c) if c in known else -1 for c in labels])
    res = {"name": path.name, "holdout": holdout or None, "seed": int(path.name.rsplit("_s", 1)[1]),
           "known": known, "budget": {}}
    for s in o["budgets"].tolist():
        logits, emb = o[f"logits_{s}"], o[f"emb_{s}"]
        pred = logits.argmax(1)
        r = {"accuracy": float(accuracy_score(y[is_known], pred[is_known])),
             "macro_f1": float(f1_score(y[is_known], pred[is_known], average="macro"))}
        if holdout:
            sc = unknown_scores(logits, emb, o["train_emb"], o["train_y"])
            r["auroc"] = {n: float(roc_auc_score(~is_known, v)) for n, v in sc.items()}
            r["fpr95"] = {n: fpr95(v[is_known], v[~is_known]) for n, v in sc.items()}
            if s == 1000:
                res["hist"] = {}
                for n, v in sc.items():
                    edges = np.linspace(*np.percentile(v, [0.5, 99.5]), HIST_BINS + 1)
                    res["hist"][n] = {"edges": edges.tolist(),
                                      "known": np.histogram(v[is_known], edges, density=True)[0].tolist(),
                                      "unknown": np.histogram(v[~is_known], edges, density=True)[0].tolist()}
        else:
            r["confusion"] = confusion_matrix(y, pred, labels=range(len(known)), normalize="true").tolist()
        res["budget"][s] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()

    runs = [evaluate_run(p) for p in sorted(args.runs.iterdir()) if (p / "DONE").exists()]
    closed = [r for r in runs if r["holdout"] is None]
    loco = [r for r in runs if r["holdout"]]
    budgets = sorted({int(s) for r in runs for s in r["budget"]})
    out = {"protocol": {"split": "RoBo6 train.csv / test.csv as released", "budgets_s": budgets,
                        "scores": SCORES, "fpr95": "unknowns accepted at 95% known acceptance",
                        "std": "sample std over seeds (ddof=1)",
                        "note": "the released CSVs contain 128 train and 27 test exact duplicate rows; kept so "
                                "counts match the paper"},
           "paper": PAPER}

    if closed:
        out["closed_set"] = {
            "seeds": [r["seed"] for r in closed], "classes": closed[0]["known"],
            "accuracy": summarise(r["budget"][1000]["accuracy"] for r in closed),
            "macro_f1": summarise(r["budget"][1000]["macro_f1"] for r in closed),
            "confusion_seed0": next((r["budget"][1000]["confusion"] for r in closed if r["seed"] == 0),
                                    closed[0]["budget"][1000]["confusion"]),
            "budget": {s: {"accuracy": summarise(r["budget"][s]["accuracy"] for r in closed),
                           "macro_f1": summarise(r["budget"][s]["macro_f1"] for r in closed)} for s in budgets}}

    if loco:
        by_class = defaultdict(list)
        for r in loco:
            by_class[r["holdout"]].append(r)
        table = {c: {"known_accuracy": summarise(r["budget"][1000]["accuracy"] for r in rs),
                     "known_macro_f1": summarise(r["budget"][1000]["macro_f1"] for r in rs),
                     "auroc": {n: summarise(r["budget"][1000]["auroc"][n] for r in rs) for n in SCORES},
                     "fpr95": {n: summarise(r["budget"][1000]["fpr95"][n] for r in rs) for n in SCORES},
                     "seeds": [r["seed"] for r in rs]}
                 for c, rs in sorted(by_class.items())}
        mean_auroc = {n: float(np.mean([table[c]["auroc"][n]["mean"] for c in table])) for n in SCORES}
        best = max(mean_auroc, key=mean_auroc.get)
        out["loco"] = {
            "table": table, "mean_auroc_over_classes": mean_auroc, "best_score": best,
            "hist_seed0": {c: next((r["hist"] for r in rs if r["seed"] == 0), rs[0]["hist"]) for c, rs in by_class.items()},
            "budget": {s: {"known_accuracy": summarise(r["budget"][s]["accuracy"] for r in loco),
                           "auroc": {n: summarise(r["budget"][s]["auroc"][n] for r in loco) for n in SCORES}}
                       for s in budgets}}

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "b_results.json").write_text(json.dumps(out, indent=1))
    if closed:
        cs = out["closed_set"]
        print(f"closed-set accuracy {cs['accuracy']['mean']:.3f}+-{cs['accuracy']['std']:.3f} "
              f"macro-F1 {cs['macro_f1']['mean']:.3f}+-{cs['macro_f1']['std']:.3f} seeds {cs['seeds']}")
    if loco:
        print("best unknown score:", out["loco"]["best_score"],
              {n: round(v, 3) for n, v in out["loco"]["mean_auroc_over_classes"].items()})
    print("wrote", args.results / "b_results.json")


if __name__ == "__main__":
    main()
