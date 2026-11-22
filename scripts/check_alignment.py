"""Are the DINOv2 features aligned in time with the labels?

The released I3D features match the label files frame for frame. For each video, the frame-to-frame
change of the DINOv2 features (distance between consecutive frames) is cross-correlated with the
change of the I3D features over lags of -45 to +45 frames (3 s). If the decoded videos line up with
the labels, the best lag sits at or near 0.

Usage: python -I scripts/check_alignment.py data cache/features/i3d cache/features/dinov2-vits14 [videos]
"""
import sys
from pathlib import Path

import numpy as np

MAX_LAG = 45


def change(f: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(f.astype(np.float32), axis=1), axis=0)
    return (d - d.mean()) / (d.std() + 1e-8)


def main() -> None:
    data, i3d, dino = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    labels = sorted((data / "ms-tcn" / "data" / "breakfast" / "groundTruth").glob("*.txt"))
    excluded = set((Path(__file__).resolve().parents[1] / "checks" / "excluded.txt").read_text().split())
    rng = np.random.default_rng(0)
    pick = [labels[i] for i in sorted(rng.choice(len(labels), size=min(n, len(labels)), replace=False))]
    lags, peaks, rows = [], [], []
    for lab in pick:
        a = change(np.load(i3d / f"{lab.stem}.npy"))
        b = change(np.load(dino / f"{lab.stem}.npy"))
        t = len(a)
        corr = [np.mean(a[max(0, -k):t - max(0, k)] * b[max(0, k):t - max(0, -k)]) for k in range(-MAX_LAG, MAX_LAG + 1)]
        k = int(np.argmax(corr)) - MAX_LAG  # b shifted by k frames matches a best
        lags.append(k)
        peaks.append(max(corr))
        rows.append((lab.stem, k, max(corr), lab.stem in excluded))
    lags = np.array(lags)
    print(f"I3D change against DINOv2 change, {len(lags)} videos: best lag median {np.median(lags):+.0f} "
          f"frames (negative: I3D later); |lag| <= 2 for {np.mean(np.abs(lags) <= 2):.0%}; "
          f"peak correlation median {np.median(peaks):.2f}")

    # Each feature's change against the label boundaries, pooled over the same videos: the lag at
    # which feature changes line up best with the frames where the label changes.
    for name, folder in (("DINOv2", dino), ("I3D", i3d)):
        total = np.zeros(2 * MAX_LAG + 1)
        for lab in pick:
            y = (lab.read_text().split("\n")[:-1])
            edges = np.zeros(len(y) - 1)
            edges[[i for i in range(len(y) - 1) if y[i + 1] != y[i]]] = 1
            edges = np.convolve(edges, np.ones(5) / 5, mode="same")
            if edges.std() == 0:
                continue
            e = (edges - edges.mean()) / edges.std()
            c = change(np.load(folder / f"{lab.stem}.npy"))
            t = len(c)
            total += [np.mean(e[max(0, -k):t - max(0, k)] * c[max(0, k):t - max(0, -k)])
                      for k in range(-MAX_LAG, MAX_LAG + 1)]
        best = int(np.argmax(total)) - MAX_LAG
        print(f"{name} change against label boundaries: best lag {best:+d} frames "
              f"(positive: the features change after the label does)")

    # Classification with the labels shifted: frames of split 1's test videos go to the nearest class
    # mean (cosine) of split 1's training frames, pairing the label of frame t with the features of
    # frame t + s. If the features are aligned, held-out accuracy peaks near s = 0.
    bf = data / "ms-tcn" / "data" / "breakfast"
    actions = {name: int(i) for i, name in (line.split() for line in (bf / "mapping.txt").read_text().splitlines() if line)}
    train = [v[:-4] for v in (bf / "splits" / "train.split1.bundle").read_text().split() if v[:-4] not in excluded]
    test = [v[:-4] for v in (bf / "splits" / "test.split1.bundle").read_text().split() if v[:-4] not in excluded]
    shifts = [-30, -20, -11, -5, 0, 5, 11, 20, 30]
    for name, folder, step in (("DINOv2", dino, 3), ("I3D", i3d, 6)):
        def pairs(stem, s):
            y = np.array([actions[a] for a in (bf / "groundTruth" / f"{stem}.txt").read_text().split("\n")[:-1]])
            f = np.load(folder / f"{stem}.npy").astype(np.float32)
            f /= np.linalg.norm(f, axis=0, keepdims=True) + 1e-8
            t = np.arange(max(0, -s), len(y) - max(0, s), step)
            return f[:, t + s], y[t]
        acc = []
        for s in shifts:
            sums = np.zeros((len(actions), f_dim := np.load(folder / f"{train[0]}.npy").shape[0]))
            for stem in train:
                f, y = pairs(stem, s)
                np.add.at(sums, y, f.T)
            means = sums / (np.linalg.norm(sums, axis=1, keepdims=True) + 1e-8)
            hit = n_all = 0
            for stem in test:
                f, y = pairs(stem, s)
                hit += int(np.sum(np.argmax(means @ f, axis=0) == y))
                n_all += len(y)
            acc.append(100 * hit / n_all)
        print(f"{name} nearest class mean, split 1 test, accuracy by shift: "
              + ", ".join(f"{s:+d}: {a:.1f}" for s, a in zip(shifts, acc)))


if __name__ == "__main__":
    main()
