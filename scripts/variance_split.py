"""How much of each feature set's variance lies between videos rather than within them.

For every video in use, the frames' features are split into the video's own mean and each frame's
difference from it. The between-video share is the part of the total variance (summed over
feature dimensions, over all frames of all videos) that the per-video means carry: what
subtracting each video's mean (the "centred" settings) removes.

Usage: python -I scripts/variance_split.py data/ms-tcn/data/breakfast/groundTruth checks/excluded.txt \
           i3d=cache/features/i3d dinov2-vits14=data/features/dinov2-vits14 ...
Writes results/variance_split.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def split_variance(folder: Path, vids: list[str]) -> dict:
    n_total, sum_total, within = 0, None, 0.0
    means, counts = [], []
    for v in vids:
        x = np.load(folder / f"{v}.npy").astype(np.float64)  # (dim, frames)
        mu = x.mean(axis=1)
        within += ((x - mu[:, None]) ** 2).sum()
        means.append(mu)
        counts.append(x.shape[1])
        sum_total = x.sum(axis=1) if sum_total is None else sum_total + x.sum(axis=1)
        n_total += x.shape[1]
    grand = sum_total / n_total
    means, counts = np.array(means), np.array(counts)
    between = (counts[:, None] * (means - grand) ** 2).sum()
    return {"videos": len(vids), "frames": n_total, "dim": len(grand),
            "between share": between / (between + within)}


def main() -> None:
    gt, excluded = Path(sys.argv[1]), set(Path(sys.argv[2]).read_text().split())
    vids = sorted(p.stem for p in gt.glob("*.txt") if p.stem not in excluded)
    rows = []
    for arg in sys.argv[3:]:
        name, folder = arg.split("=", 1)
        rows.append({"features": name, **split_variance(Path(folder), vids)})
        print(f"{name}: {100 * rows[-1]['between share']:.1f}% of the variance is between videos "
              f"({rows[-1]['videos']} videos, {rows[-1]['frames']} frames)", flush=True)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "variance_split.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
