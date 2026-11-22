"""Step 6: F1 per action for each feature setting, next to how much each action moves.

For every run folder given, the test predictions of all splits and seeds are scored per action:
the official F1 rule (a predicted segment is a hit when it overlaps a ground-truth segment of the
same action by at least the IoU threshold, each ground-truth segment counted once), tallied by
action. Summed over the per-action tallies, the counts equal the official totals, which is
checked. Motion per action is the mean pixel frame difference (scripts/frame_motion.py) over the
frames labelled with that action, from every video in use.

Usage: python -I scripts/per_action.py data runs/breakfast \
           dinov2-vits14 i3d [more run names]
Writes results/per_action.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mstcn.eval import f_score  # noqa: E402

THRESHOLDS = (0.1, 0.25, 0.5)


def segments(labels: np.ndarray):
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    return labels[starts], starts, ends


def counts_per_action(pred: np.ndarray, gt: np.ndarray, overlap: float, n: int):
    """The official f_score loop, with each hit, false positive and miss credited to its action."""
    pl, ps, pe = segments(pred)
    yl, ys, ye = segments(gt)
    tp, fp, fn = np.zeros(n), np.zeros(n), np.zeros(n)
    hits = np.zeros(len(yl), bool)
    for j in range(len(pl)):
        inter = np.minimum(pe[j], ye) - np.maximum(ps[j], ys)
        union = np.maximum(pe[j], ye) - np.minimum(ps[j], ys)
        iou = inter / union * (yl == pl[j])
        idx = iou.argmax()
        if iou[idx] >= overlap and not hits[idx]:
            tp[pl[j]] += 1
            hits[idx] = True
        else:
            fp[pl[j]] += 1
    np.add.at(fn, yl[~hits], 1)
    return tp, fp, fn


def main() -> None:
    data = Path(sys.argv[1])
    runs = Path(sys.argv[2])
    names = sys.argv[3:]
    bf = data / "ms-tcn" / "data" / "breakfast"
    mapping = [line.split() for line in (bf / "mapping.txt").read_text().splitlines() if line.strip()]
    actions = {name: int(i) for i, name in mapping}
    index_to_action = {i: name for name, i in actions.items()}
    n = len(actions)
    gt_cache = {}

    def gt(vid: str) -> np.ndarray:
        if vid not in gt_cache:
            lines = (bf / "groundTruth" / vid).read_text().split("\n")[:-1]
            gt_cache[vid] = np.array([actions[a] for a in lines], dtype=np.int16)
        return gt_cache[vid]

    table = pd.DataFrame({"action": [index_to_action[i] for i in range(n)]})
    for name in names:
        frames_hit = np.zeros(n)
        frames_all = np.zeros(n)
        tallies = {t: [np.zeros(n), np.zeros(n), np.zeros(n)] for t in THRESHOLDS}
        n_runs = 0
        for pred_file in sorted((runs / name).glob("split[0-9]/seed*/predictions.npz")):
            n_runs += 1
            preds = np.load(pred_file)
            official = np.zeros((len(THRESHOLDS), 3))
            for vid in preds.files:
                p = preds[vid].astype(np.int16)
                y = gt(vid)
                np.add.at(frames_all, y, 1)
                np.add.at(frames_hit, y[p == y], 1)
                for k, t in enumerate(THRESHOLDS):
                    tp, fp, fn = counts_per_action(p, y, t, n)
                    for acc, c in zip(tallies[t], (tp, fp, fn)):
                        acc += c
                    if n_runs == 1:  # check the per-action tallies against the official rule once
                        official[k] += f_score([index_to_action[i] for i in p],
                                               [index_to_action[i] for i in y], t)
            if n_runs == 1:
                mine = np.array([[c.sum() for c in tallies[t]] for t in THRESHOLDS])
                assert np.allclose(mine, official), (pred_file, mine, official)
        for t in THRESHOLDS:
            tp, fp, fn = tallies[t]
            with np.errstate(invalid="ignore", divide="ignore"):
                table[f"{name} f1@{int(t * 100)}"] = 100 * 2 * tp / (2 * tp + fp + fn)
        with np.errstate(invalid="ignore", divide="ignore"):
            table[f"{name} frame recall"] = 100 * frames_hit / frames_all
        table[f"{name} runs"] = n_runs
        print(f"{name}: {n_runs} runs", flush=True)

    motion = np.load(data / "motion" / "frame_diff.npz")
    excluded = set((ROOT / "checks" / "excluded.txt").read_text().split())
    m_sum, m_n, segs = np.zeros(n), np.zeros(n), np.zeros(n)
    for vid_file in sorted((bf / "groundTruth").glob("*.txt")):
        if vid_file.stem in excluded:
            continue
        y = gt(vid_file.name)
        m = motion[vid_file.stem].astype(np.float64)
        np.add.at(m_sum, y, m)
        np.add.at(m_n, y, 1)
        np.add.at(segs, segments(y)[0], 1)
    table.insert(1, "frames", m_n.astype(int))
    table.insert(2, "segments", segs.astype(int))
    table.insert(3, "motion", m_sum / np.maximum(m_n, 1))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "per_action.csv", index=False, float_format="%.2f")
    print(table.round(1).to_string())


if __name__ == "__main__":
    main()
