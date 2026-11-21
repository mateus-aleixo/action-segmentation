"""Check a Breakfast feature folder: one (dim, label count) array per label file, all values finite.

Usage: python -I scripts/check_features.py data cache/features/i3d 2048
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    data, folder, dim = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    labels = sorted((data / "ms-tcn" / "data" / "breakfast" / "groundTruth").glob("*.txt"))
    missing, bad, dtypes = [], [], Counter()
    frames = 0
    for lab in labels:
        f = folder / f"{lab.stem}.npy"
        if not f.exists():
            missing.append(lab.stem)
            continue
        n = sum(1 for line in lab.open() if line.strip())
        a = np.load(f)
        dtypes[str(a.dtype)] += 1
        frames += a.shape[-1]
        if a.shape != (dim, n) or not np.isfinite(a).all():
            bad.append((lab.stem, a.shape, n))
    extra = sorted({p.stem for p in folder.glob("*.npy")} - {lab.stem for lab in labels})
    size = sum(p.stat().st_size for p in folder.glob("*.npy")) / 2**30
    print(f"{folder}: {len(labels)} label files, {len(missing)} missing, {len(bad)} bad, {len(extra)} extra, "
          f"dtypes {dict(dtypes)}, {frames:,} frames, {size:.2f} GiB")
    for b in bad[:10]:
        print("bad", b)
    for m in missing[:10]:
        print("missing", m)
    sys.exit(1 if missing or bad else 0)


if __name__ == "__main__":
    main()
