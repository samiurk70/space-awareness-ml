"""RoBo6 loading.

The released files are already on a uniform 0.1 s grid of 10,000 points (1,000 s) with missing samples
stored as 0, so no resampling is done. The CSV's mag column holds the *_time.txt path; the magnitude
file is the same path with _mag.txt.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

N = 10_000
HZ = 10
EXPECTED = {"train": 5676, "test": 1404}
EXPECTED_TRAIN_CLASS = {"CZ-3B R/B": 2266, "DELTA 4 R/B": 233}


def load_split(root, split, cache_dir=None):
    """Returns (mag float32 (n, N), labels as class names, ids)."""
    root = Path(root)
    df = pd.read_csv(root / f"{split}.csv")
    assert len(df) == EXPECTED[split], (split, len(df))
    if split == "train":
        counts = df.label.value_counts()
        for k, v in EXPECTED_TRAIN_CLASS.items():
            assert counts[k] == v, (k, counts[k])
    cache = Path(cache_dir or root) / f"{split}_mag.npy"
    if cache.exists():
        mag = np.load(cache)
    else:
        mag = np.stack([np.loadtxt(root / p.replace("_time.txt", "_mag.txt"), dtype=np.float32)
                        for p in df.mag])
        np.save(cache, mag)
    assert mag.shape == (len(df), N), mag.shape
    return mag, df.label.to_numpy(), (df.id.astype(str) + "_" + df.part.astype(str)).to_numpy()


def to_input(mag, seconds=None):
    """[standardised magnitude, observed mask], keeping only the first `seconds` of the series.

    Standardisation uses observed points inside the kept window only, so a truncated input carries
    no information from later in the light curve.
    """
    m = mag.astype(np.float32).copy()
    if seconds is not None:
        m[int(seconds * HZ):] = 0
    mask = m != 0
    if mask.sum() > 1:
        v = m[mask]
        m[mask] = (v - v.mean()) / (v.std() + 1e-6)
    m[~mask] = 0
    return np.stack([m, mask.astype(np.float32)])


def random_window(mag, rng, lo_s=30, hi_s=1000, tries=5):
    """Random contiguous window of lo_s to hi_s seconds, shifted to the start so training inputs look
    like the truncated test inputs."""
    for _ in range(tries):
        length = int(rng.uniform(lo_s, hi_s) * HZ)
        start = int(rng.integers(0, N - length + 1))
        w = np.zeros(N, np.float32)
        w[:length] = mag[start:start + length]
        if (w != 0).sum() > 1:
            return w
    return mag


class LightCurves(Dataset):
    def __init__(self, mag, y, crop_p=0.0, seconds=None, seed=0):
        self.mag, self.y, self.crop_p, self.seconds = mag, np.asarray(y), crop_p, seconds
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        m = self.mag[i]
        if self.crop_p and self.rng.random() < self.crop_p:
            m = random_window(m, self.rng)
        return torch.from_numpy(to_input(m, self.seconds)), int(self.y[i])
