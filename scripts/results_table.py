"""The results table: each setting averaged over the 4 splits per seed, then mean and spread over
seeds, plus each setting's difference from a reference setting, paired by split and seed.

Usage: python -I scripts/results_table.py runs/breakfast i3d dinov2-vits14 dinov2-vits14-motion4 ...
The first name is the reference for the differences. Writes results/summary.csv.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["f1@10", "f1@25", "f1@50", "edit", "acc", "train acc"]  # train acc: last epoch, training videos
PAPER = [52.6, 48.1, 37.9, 61.7, 66.3, float("nan")]  # MS-TCN paper, Breakfast, I3D, all 1,712 videos


def load(folder: Path) -> dict:
    """{(split, seed): [metrics..., training accuracy of the last epoch]} for the four main splits."""
    out = {}
    for p in folder.glob("split[0-9]/seed*/metrics.json"):
        m = json.loads(p.read_text())
        last = (p.parent / "train.log").read_text().strip().splitlines()[-1]
        train_acc = 100 * float(last.split("acc =")[1].split()[0])
        out[(m["split"], m["seed"])] = [m[k] for k in KEYS] + [train_acc]
    return out


def per_seed(runs: dict) -> np.ndarray:
    """One row per seed that has all four splits: the mean over splits."""
    by_seed = defaultdict(dict)
    for (split, seed), v in runs.items():
        by_seed[seed][split] = v
    return np.array([np.mean(list(s.values()), axis=0) for s in by_seed.values() if len(s) == 4])


def fmt(mean, sd) -> list[str]:
    return [f"{m:.1f} ± {s:.1f}" for m, s in zip(mean, sd)]


def main() -> None:
    runs_dir, names = Path(sys.argv[1]), sys.argv[2:]
    data = {n: load(runs_dir / n) for n in names}
    rows = []
    print("| Features | " + " | ".join(KEYS) + " | runs |")
    print("|---" * (len(KEYS) + 2) + "|")
    print("| I3D, paper (1,712 videos) | " + " | ".join(f"{v:.1f}" if v == v else "" for v in PAPER) + " | |")
    for n in names:
        a = per_seed(data[n])
        if not len(a):
            continue
        mean, sd = a.mean(0), a.std(0, ddof=1) if len(a) > 1 else np.zeros(len(KEYS))
        rows.append({"name": n, "runs": len(data[n]), "seeds": len(a),
                     **{k: m for k, m in zip(KEYS, mean)}, **{f"{k} sd": s for k, s in zip(KEYS, sd)}})
        print(f"| {n} | " + " | ".join(fmt(mean, sd)) + f" | {len(data[n])} |")
    ref = names[0]
    print(f"\nDifference from {ref}, paired by split and seed (mean over splits per seed, then over seeds):")
    for n in names[1:]:
        common = sorted(set(data[n]) & set(data[ref]))
        diff = {k: list(np.subtract(data[n][k], data[ref][k])) for k in common}
        a = per_seed(diff)
        if not len(a):
            continue
        sd = a.std(0, ddof=1) if len(a) > 1 else np.zeros(len(KEYS))
        print(f"| {n} - {ref} | " + " | ".join(f"{m:+.1f} ± {s:.1f}" for m, s in zip(a.mean(0), sd))
              + f" | {len(common)} |")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
