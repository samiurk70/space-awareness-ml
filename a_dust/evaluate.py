"""Project A evaluation on SET 1 (test) for every finished run.

Primary metric: centroid matching. A prediction is a TP if its centre is within 3 px of an
unmatched GT centre, greedy in order of confidence. Secondary: ultralytics mAP50 on the inflated boxes.

Usage: python a_dust/evaluate.py --data /content/data/dust_yolo --dust /content/data/DUST --runs runs/a
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MATCH_PX = 3.0
CONF_OP = 0.25
PRED_MIN_CONF = 0.01
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)
RECALL_GRID = np.linspace(0, 1, 101)
STAR_EDGES = [0, 5, 15, np.inf]
STAR_WEIGHT = 0.5
CONTEXT = [
    {"method": "OrbitTrack", "source": "their paper, different pipeline", "P": 0.972, "R": 0.742},
    {"method": "EfficientDet (Roboflow)", "source": "DUST paper, random frame split", "mAP50": 0.61, "P": 0.73, "R": 0.51},
]


def predict(run, data, variant):
    cache = run / "test_preds.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from ultralytics import YOLO
    model = YOLO(str(run / "weights" / "best.pt"))
    # a folder source streams in batches; a Python list is loaded and run as one batch of every image
    out = {}
    for r in model.predict(str(data / variant / "images" / "test"), imgsz=640, batch=16, conf=PRED_MIN_CONF,
                           max_det=100, stream=True, verbose=False):
        xywh, conf = r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy()
        out[Path(r.path).stem] = [[round(float(b[0]), 2), round(float(b[1]), 2), round(float(c), 4)]
                                  for b, c in zip(xywh, conf)]
    cache.write_text(json.dumps(out))
    return out


def map50(run, data, variant):
    cache = run / "test_map50.json"
    if not cache.exists():
        from ultralytics import YOLO
        m = YOLO(str(run / "weights" / "best.pt")).val(
            data=str(data / variant / "data.yaml"), split="test", imgsz=640, conf=0.001, plots=False,
            verbose=False, project=str(run), name="test_val", exist_ok=True)
        cache.write_text(json.dumps({"mAP50": float(m.box.map50)}))
    return json.loads(cache.read_text())["mAP50"]


def match(preds, gts):
    """Greedy centroid matching, highest confidence first.

    Returns (conf, is_tp) per prediction and, per GT, the confidence of the prediction that matched it
    (0 if missed). Greedy order makes the matching at any threshold t a prefix of this one.
    """
    taken = np.zeros(len(gts), bool)
    gt_conf = np.zeros(len(gts))
    dets = []
    for x, y, c in sorted(preds, key=lambda p: -p[2]):
        if len(gts):
            d = np.hypot(gts[:, 0] - x, gts[:, 1] - y)
            d[taken] = np.inf
            j = int(np.argmin(d))
            if d[j] <= MATCH_PX:
                taken[j], gt_conf[j] = True, c
                dets.append((c, 1))
                continue
        dets.append((c, 0))
    return dets, gt_conf


def curves(dets, n_gt):
    dets = np.array(sorted(dets, key=lambda d: -d[0])) if dets else np.zeros((0, 2))
    conf, tp = dets[:, 0], dets[:, 1]
    tpc = np.cumsum(tp)
    n = np.arange(1, len(tp) + 1)
    prec, rec = tpc / n, tpc / n_gt
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)

    def at(t):
        k = int((conf >= t).sum())
        if k == 0:
            return 0.0, 0.0, 0.0
        return float(prec[k - 1]), float(rec[k - 1]), float(f1[k - 1])

    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.maximum.accumulate(np.concatenate([[1], prec, [0]])[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    env = mpre[1:-1]
    pr = [float(env[np.searchsorted(rec, r)]) if len(rec) and r <= rec[-1] else None for r in RECALL_GRID]
    b = int(np.argmax(f1)) if len(f1) else 0
    P, R, F = at(CONF_OP)
    return {
        "scalars": {"P": P, "R": R, "F1": F, "AP": ap,
                    "best_F1": float(f1[b]) if len(f1) else 0.0,
                    "best_F1_conf": float(conf[b]) if len(f1) else None,
                    "best_F1_P": float(prec[b]) if len(f1) else 0.0,
                    "best_F1_R": float(rec[b]) if len(f1) else 0.0},
        "threshold": [at(t) for t in THRESHOLDS],
        "pr": pr,
    }


def flux(img, x, y):
    """Background-subtracted 5x5 sum. Peak DN is unusable here: most RSOs saturate the 8-bit range."""
    x, y = int(round(x)), int(round(y))
    patch = img[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].astype(np.float32)
    bg = np.median(img[max(y - 7, 0):y + 8, max(x - 7, 0):x + 8])
    return float(patch.sum() - bg * patch.size)


def gt_attributes(gt, dust, data):
    """Per test RSO: speed in px/frame from the MOT track, distance to nearest catalogue star, flux."""
    import cv2
    mot = {}
    for sess in {g["session"] for g in gt}:
        rows = np.loadtxt(dust / "SET 1" / sess / "RSO_Annotations" / f"{sess}_MOT_gt.txt", delimiter=",", ndmin=2)
        mot[sess] = {(int(f), int(t)): (x + w / 2, y + h / 2) for f, t, x, y, w, h, *_ in rows}
    for g in gt:
        m = mot[g["session"]]
        here = {t: c for (f, t), c in m.items() if f == g["frame"]}
        corr = dust / "SET 1" / g["session"] / "Corr" / f"{g['key'].split('__', 1)[1]}.csv"
        stars = None
        if corr.exists():
            df = pd.read_csv(corr)
            stars = df.loc[df.match_weight >= STAR_WEIGHT, ["index_x", "index_y"]].to_numpy()
        img = cv2.imread(str(data / "V0" / "images" / "test" / f"{g['key']}.png"), cv2.IMREAD_GRAYSCALE)
        for r in g["rsos"]:
            r["flux"] = flux(img, r["x"], r["y"])
            tid = min(here, key=lambda t: np.hypot(here[t][0] - r["x"], here[t][1] - r["y"])) if here else None
            nb = [m.get((g["frame"] + s, tid)) for s in (-1, 1)] if tid is not None else []
            nb = [c for c in nb if c is not None]
            r["speed"] = float(np.hypot(nb[0][0] - r["x"], nb[0][1] - r["y"])) if nb else None
            r["star_dist"] = (float(np.hypot(stars[:, 0] - r["x"], stars[:, 1] - r["y"]).min())
                              if stars is not None and len(stars) else None)


def bin_index(values, edges):
    return [None if v is None else int(np.searchsorted(edges, v, side="right") - 1) for v in values]


def summarise(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": [float(v) for v in vals]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--dust", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()

    gt = [g for g in json.loads((args.data / "gt.json").read_text()) if g["split"] == "test"]
    gt_attributes(gt, args.dust, args.data)
    rsos = [(g, r) for g in gt for r in g["rsos"]]
    n_gt = len(rsos)

    peak = np.array([r["peak_dn"] for _, r in rsos], float)
    bright = np.array([r["flux"] for _, r in rsos], float)
    speed = [r["speed"] for _, r in rsos]
    bright_edges = [-np.inf, *np.percentile(bright, [100 / 3, 200 / 3]), np.inf]
    known_speed = np.array([s for s in speed if s is not None])
    speed_edges = [-np.inf, *np.percentile(known_speed, [100 / 3, 200 / 3]), np.inf]
    groups = {
        "brightness": bin_index(list(bright), bright_edges),
        "speed": bin_index(speed, speed_edges),
        "star_distance": bin_index([r["star_dist"] for _, r in rsos], STAR_EDGES),
    }
    edges = {"brightness": bright_edges, "speed": speed_edges, "star_distance": STAR_EDGES}
    sessions = sorted({g["session"] for g in gt})

    per_run = defaultdict(list)
    for run in sorted(p for p in args.runs.iterdir() if (p / "DONE").exists()):
        variant, seed = run.name.split("_s")
        preds = predict(run, args.data, variant)
        dets, gt_conf = [], []
        for g in gt:
            gts = np.array([[r["x"], r["y"]] for r in g["rsos"]]).reshape(-1, 2)
            d, c = match(preds.get(g["key"], []), gts)
            dets += d
            gt_conf += list(c)
        gt_conf = np.array(gt_conf)
        hit = gt_conf >= CONF_OP
        res = curves(dets, n_gt)
        res["seed"] = int(seed)
        res["scalars"]["mAP50_inflated"] = map50(run, args.data, variant)
        sess_of = np.array([g["session"] for g, _ in rsos])
        res["per_session_recall"] = {s: float(hit[sess_of == s].mean()) for s in sessions}
        res["breakdowns"] = {}
        for name, idx in groups.items():
            idx = np.array([-1 if i is None else i for i in idx])
            res["breakdowns"][name] = [float(hit[idx == b].mean()) if (idx == b).any() else None
                                       for b in range(len(edges[name]) - 1)]
        per_run[variant].append(res)
        print(f"{run.name}: P {res['scalars']['P']:.3f} R {res['scalars']['R']:.3f} "
              f"F1 {res['scalars']['F1']:.3f} AP {res['scalars']['AP']:.3f}")

    variants = {}
    for v, runs in sorted(per_run.items()):
        thr = np.array([r["threshold"] for r in runs])
        pr = np.array([[np.nan if p is None else p for p in r["pr"]] for r in runs], float)
        variants[v] = {
            "seeds": [r["seed"] for r in runs],
            "metrics": {k: summarise([r["scalars"][k] for r in runs]) for k in runs[0]["scalars"]},
            "per_session_recall": {s: summarise([r["per_session_recall"][s] for r in runs]) for s in sessions},
            "breakdowns": {n: [summarise([r["breakdowns"][n][b] for r in runs]) for b in range(len(edges[n]) - 1)]
                           for n in groups},
            "threshold_curve": {"conf": THRESHOLDS.tolist(),
                                **{m: {"mean": thr[:, :, i].mean(0).tolist(),
                                       "std": thr[:, :, i].std(0, ddof=1 if len(runs) > 1 else 0).tolist()}
                                   for i, m in enumerate(["P", "R", "F1"])}},
            "pr_curve": {"recall": RECALL_GRID.tolist(),
                         "precision": [None if np.isnan(c).all() else float(np.nanmean(c)) for c in pr.T]},
        }

    prep = json.loads((args.results / "prep_stats.json").read_text())
    out = {
        "protocol": {"train": "SET 2 minus validation sessions", "val": prep["val_sessions"], "test": "all of SET 1",
                     "match_px": MATCH_PX, "conf_operating_point": CONF_OP, "std": "sample std over seeds (ddof=1)",
                     "n_test_frames": len(gt), "n_test_rsos": n_gt, "sessions": sessions,
                     "brightness_measure": "5x5 sum minus 15x15 median background, DN",
                     "peak_dn_saturated_fraction": float((peak >= 255).mean())},
        "registration": {k: prep[k] for k in ["registration", "registration_fraction", "star_fit_err_px",
                                              "ecc_vs_stars_px", "no_corr_for_star_mask", "rso_in_star_mask"]},
        "bins": {n: {"edges": [None if not np.isfinite(e) else float(e) for e in edges[n]],
                     "counts": [int(sum(1 for i in groups[n] if i == b)) for b in range(len(edges[n]) - 1)],
                     "unknown": int(sum(1 for i in groups[n] if i is None))} for n in groups},
        "variants": variants,
        "context": CONTEXT,
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "a_results.json").write_text(json.dumps(out, indent=1))
    print("wrote", args.results / "a_results.json")


if __name__ == "__main__":
    main()
