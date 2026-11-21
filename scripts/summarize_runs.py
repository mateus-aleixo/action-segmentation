"""Average MS-TCN runs over the splits for each seed, then report the mean and spread over seeds.

Usage: python -I scripts/summarize_runs.py runs/gtea/i3d
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

KEYS = ["f1@10", "f1@25", "f1@50", "edit", "acc"]
PAPER = {  # MS-TCN paper (Abu Farha and Gall, CVPR 2019), rows for MS-TCN without fine-tuning
    "gtea": [85.8, 83.4, 69.8, 79.0, 76.3],
    "breakfast": [52.6, 48.1, 37.9, 61.7, 66.3],
}


def main() -> None:
    folder = Path(sys.argv[1])
    runs = [json.loads(p.read_text()) for p in folder.glob("split*/seed*/metrics.json")]
    by_seed = defaultdict(dict)
    for r in runs:
        by_seed[r["seed"]][r["split"]] = r
    splits = sorted({r["split"] for r in runs})
    print(f"{folder}: {len(runs)} runs, splits {', '.join(splits)}, seeds {sorted(by_seed)}")
    rows = []
    for seed, per_split in sorted(by_seed.items()):
        if len(per_split) != len(splits):
            print(f"  seed {seed}: only {len(per_split)} of {len(splits)} splits, left out")
            continue
        row = [np.mean([per_split[s][k] for s in splits]) for k in KEYS]
        rows.append(row)
        print(f"  seed {seed:>10}: " + "  ".join(f"{k} {v:5.1f}" for k, v in zip(KEYS, row)))
    if rows:
        a = np.array(rows)
        print("  mean       : " + "  ".join(f"{k} {v:5.1f}" for k, v in zip(KEYS, a.mean(0))))
        print("  sd (seeds) : " + "  ".join(f"{k} {v:5.1f}" for k, v in zip(KEYS, a.std(0, ddof=1) if len(a) > 1 else 0 * a[0])))
    dataset = runs[0]["dataset"] if runs else ""
    if dataset in PAPER:
        print("  paper      : " + "  ".join(f"{k} {v:5.1f}" for k, v in zip(KEYS, PAPER[dataset])))


if __name__ == "__main__":
    main()
