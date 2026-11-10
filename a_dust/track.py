"""Constant-velocity tracker on V3 detections, scored with MOTA / IDF1 on SET 1.

Usage: python a_dust/track.py --data /content/data/dust_yolo --dust /content/data/DUST --run runs/a/V3_s0
Self-check: python a_dust/track.py --selftest
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

GATE_PX, INIT_GATE_PX, MAX_MISS, MIN_LEN, CONF, MATCH_PX = 6.0, 15.0, 2, 3, 0.25, 3.0


def track(frames):
    """frames: list of (n, 2) detection centres. Returns {track_id: {frame_index: (x, y)}}.

    A track with one point has no velocity yet, so its first association uses the wider INIT_GATE_PX.
    """
    active, ended, next_id = [], [], 0
    for t, dets in enumerate(frames):
        dets = np.asarray(dets, float).reshape(-1, 2)
        pairs = []
        if active and len(dets):
            pred = np.array([tr["pos"] + tr["vel"] * (t - tr["t"]) for tr in active])
            gate = np.array([INIT_GATE_PX if len(tr["pts"]) == 1 else GATE_PX for tr in active])
            cost = np.linalg.norm(pred[:, None] - dets[None], axis=2)
            pairs = [(i, j) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] <= gate[i]]
        for i, j in pairs:
            tr = active[i]
            tr["vel"] = (dets[j] - tr["pos"]) / (t - tr["t"])
            tr["pos"], tr["t"], tr["pts"][t] = dets[j], t, tuple(map(float, dets[j]))
        for j in sorted(set(range(len(dets))) - {j for _, j in pairs}):
            active.append({"id": next_id, "pos": dets[j], "vel": np.zeros(2), "t": t,
                           "pts": {t: tuple(map(float, dets[j]))}})
            next_id += 1
        ended += [tr for tr in active if t - tr["t"] >= MAX_MISS]
        active = [tr for tr in active if t - tr["t"] < MAX_MISS]
    return {tr["id"]: tr["pts"] for tr in ended + active if len(tr["pts"]) >= MIN_LEN}


def accumulate(gt_frames, tracks):
    """gt_frames: list of [(gt_id, x, y), ...] aligned with the tracker's frame indices."""
    import motmetrics as mm
    acc = mm.MOTAccumulator(auto_id=True)
    for t, gts in enumerate(gt_frames):
        hyp = [(tid, pts[t]) for tid, pts in tracks.items() if t in pts]
        g = np.array([[x, y] for _, x, y in gts], float).reshape(-1, 2)
        h = np.array([p for _, p in hyp], float).reshape(-1, 2)
        acc.update([i for i, _, _ in gts], [i for i, _ in hyp], np.sqrt(mm.distances.norm2squared_matrix(g, h, max_d2=MATCH_PX ** 2)))
    return acc


def summary(accs, names):
    import motmetrics as mm
    df = mm.metrics.create().compute_many(accs, names=names, generate_overall=True, metrics=[
        "mota", "idf1", "num_switches", "num_false_positives", "num_misses", "num_unique_objects", "mostly_tracked"])
    return {n: {k: float(v) for k, v in row.items()} for n, row in df.iterrows()}


def selftest():
    frames = [[(20 + 5 * t, 30 + 2 * t), (200 - 4 * t, 120 + 3 * t)] for t in range(12)]
    gt = [[(0, *f[0]), (1, *f[1])] for f in frames]
    s = summary([accumulate(gt, track(frames))], ["synthetic"])["OVERALL"]
    assert s["num_switches"] == 0 and s["mota"] == 1.0, s
    print("selftest ok:", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--data", type=Path)
    ap.add_argument("--dust", type=Path)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    gt = [g for g in json.loads((args.data / "gt.json").read_text()) if g["split"] == "test"]
    preds = json.loads((args.run / "test_preds.json").read_text())
    accs, names, all_tracks = [], [], {}
    for sess in sorted({g["session"] for g in gt}):
        frames = sorted((g for g in gt if g["session"] == sess), key=lambda g: g["frame"])
        rows = np.loadtxt(args.dust / "SET 1" / sess / "RSO_Annotations" / f"{sess}_MOT_gt.txt", delimiter=",", ndmin=2)
        mot = [[(int(r[1]), r[2] + r[4] / 2, r[3] + r[5] / 2) for r in rows if int(r[0]) == g["frame"]] for g in frames]
        dets = [[p[:2] for p in preds.get(g["key"], []) if p[2] >= CONF] for g in frames]
        tracks = track(dets)
        accs.append(accumulate(mot, tracks))
        names.append(sess)
        all_tracks[sess] = {"frames": [g["key"] for g in frames],
                            "tracks": {str(k): {str(t): p for t, p in v.items()} for k, v in tracks.items()}}
    res_path = args.results / "a_results.json"
    res = json.loads(res_path.read_text())
    res["tracking"] = {"run": args.run.name, "conf": CONF, "gate_px": GATE_PX, "init_gate_px": INIT_GATE_PX,
                       "max_miss": MAX_MISS, "min_len": MIN_LEN, "match_px": MATCH_PX, "per_session": summary(accs, names)}
    res_path.write_text(json.dumps(res, indent=1))
    (args.results / "a_tracks.json").write_text(json.dumps(all_tracks))
    print(json.dumps(res["tracking"]["per_session"]["OVERALL"], indent=1))


if __name__ == "__main__":
    main()
