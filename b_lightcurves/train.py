"""Train the 1D ResNet on RoBo6, closed-set (6 classes) or leave-one-class-out (5 classes).

Usage: python b_lightcurves/train.py --robo6 /content/data/robo6 --cache CACHE --runs runs/b --seed 0 [--holdout "DELTA 4 R/B"]
Writes model.pt and outputs.npz (test logits and embeddings at every observation budget, train embeddings),
then DONE. Finished runs are skipped.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, get_worker_info

sys.path.insert(0, str(Path(__file__).parent))
from data import LightCurves, load_split  # noqa: E402
from model import ResNet1D  # noqa: E402

BUDGETS = [30, 60, 120, 300, 600, 1000]


def run_name(holdout, seed):
    return f"closed_s{seed}" if holdout is None else f"loco_{re.sub(r'[^a-z0-9]+', '', holdout.lower())}_s{seed}"


def reseed_worker(_):
    """Each worker gets its own crop RNG, different every epoch, fixed by the run seed."""
    info = get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)


@torch.no_grad()
def infer(model, mag, seconds, device):
    model.eval()
    logits, emb = [], []
    for x, _ in DataLoader(LightCurves(mag, np.zeros(len(mag)), seconds=seconds), batch_size=256, num_workers=2):
        with torch.autocast("cuda", enabled=device == "cuda"):
            lo, e = model(x.to(device))
        logits.append(lo.float().cpu())
        emb.append(e.float().cpu())
    return torch.cat(logits).numpy(), torch.cat(emb).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robo6", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True, help="folder for the decoded .npy arrays")
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--holdout", default=None, help="class left out of training, e.g. 'DELTA 4 R/B'")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    run = args.runs / run_name(args.holdout, args.seed)
    if (run / "DONE").exists():
        print("already trained:", run)
        return
    run.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tr_mag, tr_lab, _ = load_split(args.robo6, "train", args.cache)
    te_mag, te_lab, te_ids = load_split(args.robo6, "test", args.cache)
    classes = sorted(set(tr_lab))
    assert args.holdout is None or args.holdout in classes, (args.holdout, classes)
    known = [c for c in classes if c != args.holdout]
    idx = np.isin(tr_lab, known)
    y = np.array([known.index(c) for c in tr_lab[idx]])
    counts = np.bincount(y, minlength=len(known))
    weight = torch.tensor(len(y) / (len(known) * counts), dtype=torch.float32, device=device)

    dl = DataLoader(LightCurves(tr_mag[idx], y, crop_p=0.5, seed=args.seed), batch_size=64, shuffle=True,
                    num_workers=2, drop_last=True, worker_init_fn=reseed_worker,
                    generator=torch.Generator().manual_seed(args.seed))
    model = ResNet1D(len(known)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(dl))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for x, t in dl:
            x, t = x.to(device), t.to(device)
            with torch.autocast("cuda", enabled=device == "cuda"):
                loss = F.cross_entropy(model(x)[0], t, weight=weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        print(f"{run.name} epoch {ep + 1}/{args.epochs} loss {total / len(dl):.4f}", flush=True)

    torch.save({"state_dict": model.state_dict(), "known": known}, run / "model.pt")
    out = {"known": np.array(known), "holdout": np.array(args.holdout or ""), "test_labels": te_lab.astype(str),
           "test_ids": te_ids.astype(str), "train_y": y, "budgets": np.array(BUDGETS)}
    _, out["train_emb"] = infer(model, tr_mag[idx], None, device)
    for s in BUDGETS:
        out[f"logits_{s}"], out[f"emb_{s}"] = infer(model, te_mag, s, device)
    np.savez_compressed(run / "outputs.npz", **out)
    (run / "DONE").write_text("ok\n")
    print("wrote", run)


if __name__ == "__main__":
    main()
