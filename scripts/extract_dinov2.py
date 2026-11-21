"""Step 3: frozen DINOv2 frame features for every Breakfast label file, stored the way MS-TCN reads them.

For each MS-TCN label file (1,712), the video chosen by the rule in checks/README.md (stereo: ch0,
otherwise ch1) is decoded frame by frame at its native 15 fps. Each frame is resized to 224 x 294
pixels (the whole 320 x 240 view, 16 x 21 patches of 14 pixels; bilinear with antialiasing),
normalised with the ImageNet mean and std, and passed through DINOv2 from the cached torch.hub copy;
the class token is kept. The features are cut to the label count (frames after the last label are
dropped) or padded by repeating the last frame, then saved as float16 arrays of shape (dim, frames),
one .npy per label file, named like the label file. Each file is written under a temporary name and
renamed when complete, and finished files are skipped, so a stopped run resumes cleanly. At the end
manifest.csv is rebuilt for every label file from checks/frame_check.csv.

Usage: python -I scripts/extract_dinov2.py data vits14 [limit]
Environment: BATCH (frames per forward pass, default 256) and GPU_FRACTION (default 0.8), lowered
when the extraction shares the card with training runs.
"""
import csv
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SIZE = (224, 294)
BATCH = int(os.environ.get("BATCH", "256"))
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def video_for(label: str, videos: Path) -> Path | None:
    pid, cam, rest = label.split("_", 2)
    if cam.startswith("stereo"):
        for ch in (0, 1):
            p = videos / pid / "stereo" / f"{rest}_ch{ch}.avi"
            if p.exists():
                return p
        return None
    p = videos / pid / cam / f"{rest}.avi"
    return p if p.exists() else None


def frame_batches(path: Path, q: queue.Queue) -> None:
    """Decode a video in a thread and put RGB uint8 batches (B, H, W, 3) on the queue, then None."""
    cap = cv2.VideoCapture(str(path))
    batch = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(batch) == BATCH:
            q.put(np.stack(batch))
            batch = []
    if batch:
        q.put(np.stack(batch))
    cap.release()
    q.put(None)


@torch.no_grad()
def embed(model, frames: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(frames).to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    x = F.interpolate(x, size=SIZE, mode="bilinear", antialias=True, align_corners=False)
    x = (x - MEAN.to(device)) / STD.to(device)
    with torch.autocast("cuda", dtype=torch.float16):
        cls = model(x)
    return cls.float().cpu()


def write_manifest(out: Path, labels: list[Path], videos: Path) -> None:
    decoded = {}
    with (ROOT / "checks" / "frame_check.csv").open() as fh:
        for row in csv.DictReader(fh):
            decoded[(row["label"], row["video"])] = int(row["decoded_frames"])
    with (out / "manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "video", "label_frames", "video_frames", "frames_dropped", "frames_padded",
                    "feature_file"])
        for lab in labels:
            n_labels = sum(1 for line in lab.open() if line.strip())
            vid = video_for(lab.stem, videos)
            rel = vid.relative_to(videos).as_posix() if vid else ""
            n_video = decoded.get((lab.stem, rel), 0)
            w.writerow([lab.stem, rel, n_labels, n_video, max(0, n_video - n_labels),
                        max(0, n_labels - n_video) if n_video else 0, (out / f"{lab.stem}.npy").exists()])


def main() -> None:
    data, arch = Path(sys.argv[1]), sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    gt = data / "ms-tcn" / "data" / "breakfast" / "groundTruth"
    videos = data / "breakfast-videos" / "videos"
    out = data / "features" / f"dinov2-{arch}"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRACTION", "0.8")))
    if sys.platform == "win32":  # keep Windows awake while this runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    hub = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    model = torch.hub.load(str(hub), f"dinov2_{arch}", source="local").to(device).eval()

    labels = sorted(gt.glob("*.txt"))[:limit]
    todo = [lab for lab in labels if not (out / f"{lab.stem}.npy").exists()]
    print(f"{len(labels)} label files, {len(labels) - len(todo)} already done, {len(todo)} to extract", flush=True)
    t0 = time.time()
    frames_done = 0
    for i, lab in enumerate(todo, 1):
        vid = video_for(lab.stem, videos)
        if vid is None:
            print(f"no video for {lab.stem}", flush=True)
            continue
        n_labels = sum(1 for line in lab.open() if line.strip())
        q: queue.Queue = queue.Queue(maxsize=4)
        threading.Thread(target=frame_batches, args=(vid, q), daemon=True).start()
        feats = []
        while (batch := q.get()) is not None:
            feats.append(embed(model, batch, device))
        f = torch.cat(feats).numpy()
        if f.shape[0] >= n_labels:
            f = f[:n_labels]
        else:
            f = np.concatenate([f, np.repeat(f[-1:], n_labels - f.shape[0], axis=0)])
        tmp = out / f"{lab.stem}.partial"
        with tmp.open("wb") as fh:
            np.save(fh, f.T.astype(np.float16))
        os.replace(tmp, out / f"{lab.stem}.npy")
        frames_done += n_labels
        if i % 50 == 0 or i == len(todo):
            rate = frames_done / (time.time() - t0)
            print(f"{i}/{len(todo)} videos, {frames_done:,} frames, {rate:,.0f} frames/s", flush=True)

    write_manifest(out, labels, videos)
    print(f"done: {len(todo)} extracted, {frames_done:,} frames in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
