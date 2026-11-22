"""Train and test MS-TCN as the official main.py does, for several splits and seeds, and score each run.

Each run is seeded the way the official code seeds (random and torch, cudnn deterministic), trains
for 50 epochs with batch size 1, predicts the split's test videos with the last epoch's weights and
scores them with the official metrics. Results go to runs/<dataset>/<name>/split<k>/seed<s>/, and a
run whose metrics.json exists is skipped, so an interrupted grid can be started again.

Options the official code does not have:
  --exclude FILE      leaves the listed videos out of the training and test lists (Breakfast:
                      checks/excluded.txt, the 20 label files whose video does not match, see
                      checks/README.md)
  --motion-k K        puts each frame's features next to their difference from K frames earlier
                      (the first K frames take their difference from frame 0)
  --val-subjects A-B  trains on the split's training videos without subjects A to B and tests on
                      those subjects, so a setting can be chosen without the split's test videos
  --gpu-fraction F    caps this process's GPU memory, so several runs can share the 6 GB card

Usage (step 2, GTEA with the released I3D features):
  python -I scripts/train_mstcn.py --data data/ms-tcn/data --dataset gtea
      --features i3d --splits 1,2,3,4 --seeds 1538574472,0,1
Usage (step 4, Breakfast with DINOv2 ViT-S features):
  python -I scripts/train_mstcn.py --data data/ms-tcn/data
      --dataset breakfast --features dinov2-vits14 --features-dir cache/features/dinov2-vits14
      --exclude checks/excluded.txt --seeds 1538574472 --gpu-fraction 0.25
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mstcn.batch_gen import BatchGenerator  # noqa: E402
from mstcn.eval import evaluate  # noqa: E402
from mstcn.model import Trainer  # noqa: E402


def read_list(path: Path) -> list[str]:
    return path.read_text().split("\n")[:-1]


def subject(vid: str) -> int:
    return int(vid.split("_")[0][1:])  # Breakfast: P03_cam01_P03_cereals.txt -> 3


def make_loader(features_dir: Path, motion_k: int, center: bool = False):
    """Features as MS-TCN reads them; center subtracts the video's own mean feature (no labels used)."""
    def load_features(vid: str) -> np.ndarray:
        f = np.load(features_dir / (vid.split(".")[0] + ".npy")).astype(np.float32, copy=False)
        if center:
            f = f - f.mean(axis=1, keepdims=True)
        if motion_k:
            earlier = np.maximum(np.arange(f.shape[1]) - motion_k, 0)
            f = np.concatenate([f, f - f[:, earlier]], axis=0)
        return f
    return load_features


def run_one(args, data: Path, features_dir: Path, split: str, seed: int, device) -> dict | None:
    root = data / args.dataset
    gt_path = str(root / "groundTruth") + "/"
    actions = read_list(root / "mapping.txt")
    actions_dict = {a.split()[1]: int(a.split()[0]) for a in actions}
    train_list = read_list(root / "splits" / f"train.split{split}.bundle")
    test_list = read_list(root / "splits" / f"test.split{split}.bundle")
    excluded = set()
    if args.exclude:
        excluded = {line.strip() for line in Path(args.exclude).read_text().splitlines() if line.strip()}
    n_before = len(train_list) + len(test_list)
    train_list = [v for v in train_list if v.split(".")[0] not in excluded]
    test_list = [v for v in test_list if v.split(".")[0] not in excluded]
    n_excluded = n_before - len(train_list) - len(test_list)
    split_name = f"split{split}"
    if args.val_subjects:
        lo, hi = (int(x) for x in args.val_subjects.split("-"))
        test_list = [v for v in train_list if lo <= subject(v) <= hi]
        train_list = [v for v in train_list if not lo <= subject(v) <= hi]
        split_name = f"split{split}-val{lo}-{hi}"

    out = Path(args.out) / args.dataset / args.name / split_name / f"seed{seed}"
    if (out / "metrics.json").exists():
        print(f"{out} already done, skipped", flush=True)
        return None
    out.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    load_features = make_loader(features_dir, args.motion_k)
    dim = load_features(train_list[0]).shape[0]
    log_file = (out / "train.log").open("w")
    t0 = time.time()

    def log(line: str) -> None:
        log_file.write(f"{line}   ({time.time() - t0:.0f} s)\n")
        log_file.flush()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(4, 10, 64, dim, len(actions_dict))
    batch_gen = BatchGenerator(len(actions_dict), actions_dict, gt_path, load_features, args.sample_rate)
    batch_gen.read_data(train_list)
    trainer.train(str(out), batch_gen, num_epochs=args.epochs, batch_size=1, learning_rate=0.0005,
                  device=device, log=log)
    train_s = time.time() - t0
    log_file.close()

    recognitions = trainer.predict(str(out), load_features, test_list, args.epochs, actions_dict, device,
                                   args.sample_rate)
    metrics = evaluate(recognitions, gt_path, test_list)
    metrics.update({"dataset": args.dataset, "features": args.features, "name": args.name, "split": split,
                    "seed": seed, "feature_dim": dim, "motion_k": args.motion_k,
                    "val_subjects": args.val_subjects, "epochs": args.epochs,
                    "train_videos": len(train_list), "test_videos": len(test_list),
                    "excluded_videos": n_excluded, "train_seconds": round(train_s, 1),
                    "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0,
                    "torch": torch.__version__})
    np.savez_compressed(out / "predictions.npz",
                        **{vid: np.array([actions_dict[a] for a in rec], dtype=np.int16)
                           for vid, rec in recognitions.items()})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder holding <dataset>/groundTruth, splits, mapping.txt")
    ap.add_argument("--dataset", default="gtea")
    ap.add_argument("--features", default="i3d",
                    help="i3d reads <dataset>/features; another name reads <dataset>/features-<name>")
    ap.add_argument("--features-dir", default="", help="override the features folder")
    ap.add_argument("--name", default="", help="run folder name (default: features, plus -motion<k>)")
    ap.add_argument("--splits", default="1,2,3,4")
    ap.add_argument("--seeds", default="1538574472")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--sample-rate", type=int, default=1)
    ap.add_argument("--exclude", default="", help="file of video names to leave out, one per line")
    ap.add_argument("--motion-k", type=int, default=0)
    ap.add_argument("--val-subjects", default="", help="e.g. 16-28")
    ap.add_argument("--gpu-fraction", type=float, default=0.8)
    ap.add_argument("--out", default=str(ROOT / "runs"))
    args = ap.parse_args()
    if not args.name:
        args.name = args.features + (f"-motion{args.motion_k}" if args.motion_k else "")

    data = Path(args.data)
    if args.features_dir:
        features_dir = Path(args.features_dir)
    elif args.features == "i3d":
        features_dir = data / args.dataset / "features"
    else:
        features_dir = data / args.dataset / f"features-{args.features}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
        # The CPU side only builds small tensors; PyTorch's default thread pool spun about 5 cores
        # per run here and slowed the card, so one thread (the results do not depend on it).
        torch.set_num_threads(1)
    if sys.platform == "win32":  # keep Windows from sleeping while this runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    for seed in [int(s) for s in args.seeds.split(",")]:
        for split in args.splits.split(","):
            m = run_one(args, data, features_dir, split, seed, device)
            if m:
                print(f"{args.dataset} {args.name} split {split} seed {seed}: "
                      f"F1@10 {m['f1@10']:.1f}  F1@25 {m['f1@25']:.1f}  F1@50 {m['f1@50']:.1f}  "
                      f"Edit {m['edit']:.1f}  Acc {m['acc']:.1f}  ({m['train_seconds']:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
