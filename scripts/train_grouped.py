"""Train MS-TCN on several feature settings of the same split and seed at once, then score each.

The models are trained side by side (mstcn/grouped.py): the same videos in the official order for
the seed, each model with the official initial weights for the seed, the official loss and Adam.
Everything after training is as in train_mstcn.py: the last epoch's weights (saved per setting in
the official format), the official predict step and metrics. Results go to
runs/<dataset>/<name>/split<k>/seed<s>/, the layout train_mstcn.py uses; a setting whose
metrics.json exists is left out of the group, and a split and seed with all settings done is skipped.

A setting is NAME=FOLDER, NAME=FOLDER:K for the motion cue (features next to their difference from
K frames earlier), and NAME=FOLDER:K:center to subtract each video's own mean feature first.
--exclude and --val-subjects work as in train_mstcn.py.

Usage:
  python -I scripts/train_grouped.py --data data/ms-tcn/data
      --dataset breakfast --exclude checks/excluded.txt --splits 1 --seeds 1538574472
      --setting dinov2-vits14=cache/features/dinov2-vits14 --setting i3d=cache/features/i3d
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mstcn.eval import evaluate  # noqa: E402
from mstcn.grouped import FeatureSource, GroupedBatchGenerator, GroupedTrainer  # noqa: E402
from train_mstcn import read_list, subject  # noqa: E402


def parse_setting(text: str):
    name, spec = text.split("=", 1)
    parts = spec.split(":")
    center = parts[-1] == "center"
    if center:
        parts.pop()
    k = int(parts.pop()) if len(parts) > 1 and parts[-1].isdigit() else 0
    return name, Path(":".join(parts)), k, center


def run_group(args, settings, split: str, seed: int, device) -> None:
    root = Path(args.data) / args.dataset
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

    outs = {name: Path(args.out) / args.dataset / name / split_name / f"seed{seed}" for name, _, _ in settings}
    todo = [s for s in settings if not (outs[s[0]] / "metrics.json").exists()]
    if not todo:
        print(f"{split_name} seed {seed}: all settings done, skipped", flush=True)
        return
    names = [s[0] for s in todo]

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    source = FeatureSource([(folder, k, center) for _, folder, k, center in todo])
    raw = source.load(train_list[0])
    dims = [x.shape[1] for x in source.inputs(raw, source.frames(raw), device)]
    t0 = time.time()
    files = []
    for name in names:
        outs[name].mkdir(parents=True, exist_ok=True)
        files.append((outs[name] / "train.log").open("w"))

    def logger(fh):
        def log(line: str) -> None:
            fh.write(f"{line}   ({time.time() - t0:.0f} s)\n")
            fh.flush()
        return log

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trainer = GroupedTrainer(dims, len(actions_dict), seed)
    batch_gen = GroupedBatchGenerator(actions_dict, gt_path, source)
    batch_gen.read_data(train_list)
    trainer.train(batch_gen, args.epochs, 0.0005, device, [logger(fh) for fh in files])
    train_s = time.time() - t0
    for fh in files:
        fh.close()
    for g, name in enumerate(names):
        torch.save(trainer.model.official_state(g), outs[name] / f"epoch-{args.epochs}.model")

    recognitions = trainer.predict(source, test_list, actions_dict, device)
    for g, (name, folder, k, center) in enumerate(todo):
        metrics = evaluate(recognitions[g], gt_path, test_list)
        metrics.update({"dataset": args.dataset, "features": folder.name, "name": name, "split": split,
                        "seed": seed, "feature_dim": dims[g], "motion_k": k, "centered": center,
                        "val_subjects": args.val_subjects, "epochs": args.epochs,
                        "train_videos": len(train_list), "test_videos": len(test_list),
                        "excluded_videos": n_excluded, "train_seconds": round(train_s, 1),
                        "trained_alongside": names,
                        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0,
                        "torch": torch.__version__})
        np.savez_compressed(outs[name] / "predictions.npz",
                            **{vid: np.array([actions_dict[a] for a in rec], dtype=np.int16)
                               for vid, rec in recognitions[g].items()})
        (outs[name] / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"{args.dataset} {name} {split_name} seed {seed}: F1@10 {metrics['f1@10']:.1f}  "
              f"F1@25 {metrics['f1@25']:.1f}  F1@50 {metrics['f1@50']:.1f}  Edit {metrics['edit']:.1f}  "
              f"Acc {metrics['acc']:.1f}  ({train_s:.0f} s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", default="breakfast")
    ap.add_argument("--setting", action="append", required=True, help="NAME=FOLDER[:K][:center]")
    ap.add_argument("--splits", default="1,2,3,4")
    ap.add_argument("--seeds", default="1538574472")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--val-subjects", default="")
    ap.add_argument("--gpu-fraction", type=float, default=0.8)
    ap.add_argument("--out", default=str(ROOT / "runs"))
    args = ap.parse_args()
    settings = [parse_setting(s) for s in args.setting]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
        torch.set_num_threads(1)  # see train_mstcn.py
    if sys.platform == "win32":  # keep Windows awake while this runs (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    for seed in [int(s) for s in args.seeds.split(",")]:
        for split in args.splits.split(","):
            run_group(args, settings, split, seed, device)


if __name__ == "__main__":
    main()
