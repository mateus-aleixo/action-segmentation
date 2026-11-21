"""Step 6 input: how much each frame moves, from the video pixels alone.

For every label file, the video chosen in step 3 (see data/features/dinov2-vits14/manifest.csv)
is decoded again. Each frame is turned grey, and its motion is the mean absolute difference from
the frame before, on the 0 to 255 scale (frame 0 gets 0). The arrays are cut or padded to the
label count by the same rule as the features, and all of them go into data/motion/frame_diff.npz.

Usage: python -I scripts/frame_motion.py data [workers]
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def frame_diff(job):
    label, video, n = job
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(video)
    prev = None
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        out.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    m = np.array(out[:n], dtype=np.float32)
    if len(m) < n:
        m = np.concatenate([m, np.repeat(m[-1:], n - len(m))])
    return label, m.astype(np.float16)


def main() -> None:
    data = Path(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    man = pd.read_csv(data / "features" / "dinov2-vits14" / "manifest.csv")
    videos = data / "breakfast-videos" / "videos"
    jobs = [(r.label, str(videos / r.video), int(r.label_frames)) for r in man.itertuples()]
    out = data / "motion"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = {}
    with Pool(workers) as pool:
        for i, (label, m) in enumerate(pool.imap_unordered(frame_diff, jobs, chunksize=4), 1):
            result[label] = m
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} videos, {(time.time() - t0) / 60:.1f} min", flush=True)
    np.savez_compressed(out / "frame_diff.npz", **result)
    frames = sum(len(m) for m in result.values())
    print(f"done: {len(result)} videos, {frames:,} frames in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
