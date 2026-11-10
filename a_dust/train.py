"""Train one YOLO11n detector on one channel variant.

Usage: python a_dust/train.py --data /content/data/dust_yolo --variant V3 --seed 0 --project runs/a
Skips if the run already finished, resumes from last.pt if it was interrupted.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--variant", choices=["V0", "V1", "V2", "V3"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    run = args.project / f"{args.variant}_s{args.seed}"
    done = run / "DONE"
    if done.exists():
        print("already trained:", run)
        return
    last = run / "weights" / "last.pt"
    if last.exists():
        try:
            YOLO(str(last)).train(resume=True)
        except AssertionError as e:
            print("nothing to resume, treating as finished:", e)
    else:
        YOLO("yolo11n.pt").train(
            data=str(args.data / args.variant / "data.yaml"), imgsz=640, epochs=args.epochs, batch=32,
            seed=args.seed, deterministic=True, hsv_h=0, hsv_s=0, hsv_v=0, mosaic=0.5, degrees=0,
            fliplr=0.5, flipud=0.5, patience=20, cache="ram",
            # Channels are not colours: no ToGray, CLAHE or blur from the default albumentations set.
            augmentations=[],
            project=str(args.project), name=run.name, exist_ok=True)
    done.write_text("ok\n")


if __name__ == "__main__":
    main()
