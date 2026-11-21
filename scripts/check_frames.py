"""Step 1 check: match every MS-TCN Breakfast label file to a video and compare frame counts.

Label names look like P03_cam01_P03_cereals (video P03/cam01/P03_cereals.avi). Stereo labels,
P03_stereo01_P03_cereals, have two videos, P03/stereo/P03_cereals_ch0.avi and _ch1.avi, which is
why there are 1,989 videos for 1,712 label files. Every video is decoded to count its frames,
since AVI headers can be wrong.

Usage: python -I scripts/check_frames.py data checks/frame_check.csv
"""
import csv
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2


def videos_for(label: str, videos: Path) -> list[Path]:
    pid, cam, rest = label.split("_", 2)
    if cam.startswith("stereo"):
        return [videos / pid / "stereo" / f"{rest}_ch{c}.avi" for c in (0, 1)]
    return [videos / pid / cam / f"{rest}.avi"]


def count_frames(path: str) -> tuple[str, int, int, float]:
    cap = cv2.VideoCapture(path)
    header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    decoded = 0
    while cap.grab():
        decoded += 1
    cap.release()
    return path, header, decoded, fps


def bucket(d: int) -> str:
    a = abs(d)
    return "0" if a == 0 else "1-2" if a <= 2 else "3-15" if a <= 15 else "16-150" if a <= 150 else ">150"


def main() -> None:
    data, out_csv = Path(sys.argv[1]), Path(sys.argv[2])
    gt = data / "ms-tcn" / "data" / "breakfast" / "groundTruth"
    videos = data / "breakfast-videos" / "videos"
    labels = {p.stem: sum(1 for line in p.open() if line.strip()) for p in sorted(gt.glob("*.txt"))}
    all_videos = {str(p) for p in videos.rglob("*.avi")}

    pairs = []
    missing = []
    for label, n in labels.items():
        found = [v for v in videos_for(label, videos) if str(v) in all_videos]
        if not found:
            missing.append(label)
        pairs += [(label, n, str(v)) for v in found]
    used = {v for _, _, v in pairs}

    workers = max(1, (os.cpu_count() or 4) - 2)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        counts = {p: (h, d, f) for p, h, d, f in pool.map(count_frames, sorted(used), chunksize=8)}

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    diffs = Counter()
    fps_seen = Counter()
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "video", "label_frames", "header_frames", "decoded_frames", "fps", "diff"])
        for label, n, v in pairs:
            h, d, f = counts[v]
            w.writerow([label, Path(v).relative_to(videos).as_posix(), n, h, d, round(f, 3), d - n])
            diffs[bucket(d - n)] += 1
            fps_seen[round(f, 2)] += 1

    print(f"labels {len(labels)}, videos {len(all_videos)}, matched pairs {len(pairs)}")
    print(f"labels with no video: {len(missing)} {missing[:10]}")
    print(f"videos not matched to a label: {len(all_videos - used)}")
    print("decoded minus label frames:", dict(sorted(diffs.items())))
    print("fps:", dict(fps_seen.most_common(5)))
    stereo = [(lab, counts[v][1]) for lab, _, v in pairs if "_stereo" in lab]
    by_label = {}
    for lab, d in stereo:
        by_label.setdefault(lab, []).append(d)
    unequal = [lab for lab, ds in by_label.items() if len(ds) == 2 and ds[0] != ds[1]]
    print(f"stereo labels {len(by_label)}, with one channel only {sum(len(d) == 1 for d in by_label.values())}, "
          f"channels of unequal length {len(unequal)}")
    print(f"written {out_csv}")


if __name__ == "__main__":
    main()
