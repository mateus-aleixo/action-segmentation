"""How often each setting labels a frame with an action that belongs to another breakfast.

A video's activity is the end of its name (P14_webcam01_P14_milk: milk), and an activity's
actions are every action its ground truth uses in the videos in use, plus the background. A
predicted frame counts as "another activity" when its action is not among those of its video's
activity. The share is over all test frames of the 12 runs per setting.

Usage: python -I scripts/other_activity.py data runs/breakfast i3d dinov2-vits14 ...
Writes results/other_activity.csv.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    bf = Path(sys.argv[1]) / "ms-tcn" / "data" / "breakfast"
    runs, names = Path(sys.argv[2]), sys.argv[3:]
    mapping = [line.split() for line in (bf / "mapping.txt").read_text().splitlines() if line.strip()]
    actions = {name: int(i) for i, name in mapping}
    excluded = set((ROOT / "checks" / "excluded.txt").read_text().split())
    allowed = defaultdict(lambda: {actions["SIL"]})
    for f in (bf / "groundTruth").glob("*.txt"):
        if f.stem not in excluded:
            allowed[f.stem.split("_")[-1]].update(actions[a] for a in f.read_text().split("\n")[:-1])
    rows = []
    for name in names:
        other = total = 0
        for pred_file in sorted((runs / name).glob("split[0-9]/seed*/predictions.npz")):
            preds = np.load(pred_file)
            for vid in preds.files:
                ok = np.array(sorted(allowed[vid[:-4].split("_")[-1]]))
                p = preds[vid]
                other += int((~np.isin(p, ok)).sum())
                total += len(p)
        rows.append({"name": name, "frames": total, "other activity share": other / total})
        print(f"{name}: {100 * other / total:.1f}% of test frames labelled with another activity's action")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "other_activity.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
