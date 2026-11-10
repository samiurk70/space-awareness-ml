"""DUST -> YOLO datasets, one per channel variant (V0 to V3).

Protocol: train on SET 2, validate on 2 held-out SET 2 sessions, test on all of SET 1.
The first frame of every session is dropped (it has no previous frame).

Usage: python a_dust/prep.py --dust /content/data/DUST --out /content/data/dust_yolo
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

IMG = 256
VARIANTS = ["V0", "V1", "V2", "V3"]
DEFAULT_VAL = ["2023_05_10", "2023_07_20"]
MIN_BOX_PX = 6
STAR_RADIUS = 2
STAR_WEIGHT = 0.5
SKY_MATCH_DEG = 1e-4
CORNERS = np.array([[x, y, 1.0] for x in (0, IMG - 1) for y in (0, IMG - 1)])


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img.ndim == 3:
        img = img[..., 0]
    assert img.shape == (IMG, IMG) and img.dtype == np.uint8, (path, img.shape, img.dtype)
    return img


def read_corr(path):
    return pd.read_csv(path) if path.exists() else None


def read_yolo(path):
    """Returns GT centres and sizes in pixels, shape (n, 4): xc, yc, w, h."""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 4))
    rows = np.loadtxt(path, ndmin=2)
    assert set(rows[:, 0].astype(int)) <= {1}, f"unexpected class in {path}"
    return rows[:, 1:5] * IMG


def star_matches(corr_prev, corr_cur):
    """Pairs of (field_x, field_y) for the same catalogue star in both frames.

    index_id is a per-frame index, not a stable star id, so stars are matched on
    their catalogue sky position (index_ra, index_dec) instead.
    """
    tree = cKDTree(corr_prev[["index_ra", "index_dec"]].to_numpy())
    dist, idx = tree.query(corr_cur[["index_ra", "index_dec"]].to_numpy())
    ok = dist < SKY_MATCH_DEG
    src = corr_prev[["field_x", "field_y"]].to_numpy()[idx[ok]]
    dst = corr_cur[["field_x", "field_y"]].to_numpy()[ok]
    return src.astype(np.float32), dst.astype(np.float32)


def point_sources(img):
    """High-pass then blur. Without the high-pass, static background structure pulls ECC towards identity."""
    hp = np.clip(img.astype(np.float32) - cv2.medianBlur(img, 9).astype(np.float32), 0, None)
    return cv2.GaussianBlur(hp, (0, 0), 2.0)


def register(prev, cur, corr_prev, corr_cur):
    """2x3 matrix mapping prev pixel coords onto cur, plus the method used and the star fit error."""
    if corr_prev is not None and corr_cur is not None:
        src, dst = star_matches(corr_prev, corr_cur)
        if len(src) >= 4:
            M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5)
            if M is not None and inl.sum() >= 4 and 0.95 < np.hypot(M[0, 0], M[1, 0]) < 1.05:
                inl = inl.ravel().astype(bool)
                fit = src[inl] @ M[:, :2].T + M[:, 2]
                err = float(np.median(np.linalg.norm(fit - dst[inl], axis=1)))
                return M, "stars", err
    try:
        W = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
        _, W = cv2.findTransformECC(point_sources(cur), point_sources(prev), W, cv2.MOTION_EUCLIDEAN, crit, None, 1)
        return cv2.invertAffineTransform(W), "ecc", None
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32), "identity", None


def residual(prev, cur, M):
    """Positive part of cur minus registered prev, zero where prev has no data."""
    warped = cv2.warpAffine(prev, M, (IMG, IMG), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.ones_like(prev), M, (IMG, IMG), flags=cv2.INTER_NEAREST) > 0
    r = np.clip(2.0 * (cur.astype(np.float32) - warped.astype(np.float32)), 0, 255).astype(np.uint8)
    r[~valid] = 0
    return r


def star_mask(corr):
    m = np.zeros((IMG, IMG), np.uint8)
    if corr is None:
        return m
    for x, y in corr.loc[corr.match_weight >= STAR_WEIGHT, ["index_x", "index_y"]].to_numpy():
        cv2.circle(m, (int(round(x)), int(round(y))), STAR_RADIUS, 255, -1)
    return m


def peak_dn(img, xc, yc):
    x, y = int(round(xc)), int(round(yc))
    return int(img[max(y - 2, 0):y + 3, max(x - 2, 0):x + 3].max())


def yolo_lines(boxes):
    out = []
    for xc, yc, w, h in boxes:
        w, h = max(w, MIN_BOX_PX), max(h, MIN_BOX_PX)
        out.append(f"0 {xc / IMG:.6f} {yc / IMG:.6f} {w / IMG:.6f} {h / IMG:.6f}")
    return "\n".join(out)


def mot_centres(path):
    """frame (1-indexed) -> list of (track_id, xc, yc). MOT x, y are the top-left corner."""
    out = defaultdict(list)
    if path.exists():
        for f, tid, x, y, w, h, *_ in np.loadtxt(path, delimiter=",", ndmin=2):
            out[int(f)].append((int(tid), x + w / 2, y + h / 2))
    return out


def sessions(dust):
    for set_name in ["SET 1", "SET 2"]:
        for s in sorted((dust / set_name).iterdir()):
            if s.is_dir():
                yield set_name, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dust", type=Path, required=True, help="folder containing 'SET 1' and 'SET 2'")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--val", nargs="+", default=DEFAULT_VAL, help="SET 2 sessions held out for validation")
    args = ap.parse_args()

    for v in VARIANTS:
        for split in ["train", "val", "test"]:
            (args.out / v / "images" / split).mkdir(parents=True, exist_ok=True)
            (args.out / v / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"frames": Counter(), "rsos": Counter(), "registration": Counter(),
             "registration_by_session": {}, "no_corr_for_star_mask": 0, "star_fit_err_px": [],
             "ecc_vs_stars_px": [],
             "rso_in_star_mask": [0, 0], "frames_without_yolo_file": 0,
             "yolo_vs_mot_count_mismatch": 0, "yolo_vs_mot_max_dist_px": 0.0, "sessions": {}}
    gt = []

    for set_name, sdir in sessions(args.dust):
        name = sdir.name
        split = "test" if set_name == "SET 1" else ("val" if name in args.val else "train")
        frames = sorted((sdir / "Images").glob("*.png"))
        mot = mot_centres(sdir / "RSO_Annotations" / f"{name}_MOT_gt.txt")
        reg_counts = Counter()
        prev, corr_prev = None, None
        for i, fp in enumerate(frames):
            cur = read_gray(fp)
            corr_cur = read_corr(sdir / "Corr" / f"{fp.stem}.csv")
            if i == 0:
                prev, corr_prev = cur, corr_cur
                continue
            yolo_path = sdir / "RSO_Annotations" / "YOLO" / f"{fp.stem}.txt"
            stats["frames_without_yolo_file"] += not yolo_path.exists()
            boxes = read_yolo(yolo_path)

            mot_here = mot.get(i + 1, [])
            if len(mot_here) != len(boxes):
                stats["yolo_vs_mot_count_mismatch"] += 1
            elif len(boxes):
                d = np.linalg.norm(boxes[:, None, :2] - np.array([m[1:] for m in mot_here])[None], axis=2)
                stats["yolo_vs_mot_max_dist_px"] = max(stats["yolo_vs_mot_max_dist_px"], float(d.min(1).max()))

            M, method, err = register(prev, cur, corr_prev, corr_cur)
            reg_counts[method] += 1
            if err is not None:
                stats["star_fit_err_px"].append(err)
                E, _, _ = register(prev, cur, None, None)
                stats["ecc_vs_stars_px"].append(float(np.linalg.norm(CORNERS @ (M - E).T, axis=1).max()))
            R = residual(prev, cur, M)
            S = star_mask(corr_cur)
            stats["no_corr_for_star_mask"] += corr_cur is None
            for xc, yc, *_ in boxes:
                stats["rso_in_star_mask"][1] += 1
                stats["rso_in_star_mask"][0] += int(S[min(int(round(yc)), IMG - 1), min(int(round(xc)), IMG - 1)] > 0)

            channels = {"V0": (cur, cur, cur), "V1": (prev, cur, cur), "V2": (cur, R, cur), "V3": (cur, R, S)}
            key = f"{name}__{fp.stem}"
            labels = yolo_lines(boxes)
            for v, (r, g, b) in channels.items():
                cv2.imwrite(str(args.out / v / "images" / split / f"{key}.png"), np.dstack([b, g, r]))
                (args.out / v / "labels" / split / f"{key}.txt").write_text(labels)

            gt.append({"key": key, "session": name, "set": set_name, "split": split, "frame": i + 1,
                       "registration": method,
                       "rsos": [{"x": float(xc), "y": float(yc), "w": float(w), "h": float(h),
                                 "peak_dn": peak_dn(cur, xc, yc)} for xc, yc, w, h in boxes]})
            stats["frames"][split] += 1
            stats["rsos"][split] += len(boxes)
            prev, corr_prev = cur, corr_cur
        stats["registration"].update(reg_counts)
        stats["registration_by_session"][name] = dict(reg_counts)
        stats["sessions"][name] = {"set": set_name, "split": split, "frames_used": len(frames) - 1}

    for v in VARIANTS:
        (args.out / v / "data.yaml").write_text(
            f"path: {(args.out / v).resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            "names:\n  0: rso\n")
    (args.out / "gt.json").write_text(json.dumps(gt))

    hit, tot = stats["rso_in_star_mask"]
    total_reg = sum(stats["registration"].values())
    stats["rso_in_star_mask"] = {"inside": hit, "total": tot, "fraction": hit / max(tot, 1)}
    stats["registration_fraction"] = {k: v / total_reg for k, v in stats["registration"].items()}
    for k in ["star_fit_err_px", "ecc_vs_stars_px"]:
        e = stats.pop(k)
        stats[k] = {"median": float(np.median(e)) if e else None,
                    "p95": float(np.percentile(e, 95)) if e else None, "n": len(e)}
    stats["val_sessions"] = args.val
    stats = json.loads(json.dumps(stats))
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "prep_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    if stats["rso_in_star_mask"]["fraction"] > 0.05:
        raise SystemExit("STOP: many GT RSO centres fall inside the star mask. Check coordinates before training.")


if __name__ == "__main__":
    main()
