"""The step 4 and 5 grid on Modal GPUs: one container per split and seed, the same scripts as locally.

The labels, splits, mapping and the Breakfast I3D features are fetched inside Modal from the MS-TCN
Zenodo record (scripts/download_data.py); the DINOv2 features are uploaded from the local extraction
with `modal volume put`. Everything lives on the volume `action-segmentation`:
  /data/ms-tcn/data/breakfast/{groundTruth,splits,mapping.txt}
  /features/{i3d,dinov2-vits14,dinov2-vitb14}/<video>.npy
  /runs/breakfast/<setting>/split<k>/seed<s>/   (results, the same layout as runs/ locally)
Each job copies the features it needs to the container's own disk, runs scripts/train_grouped.py for
its split and seed, and copies its run folders back to the volume.

Usage:
  modal run scripts/modal_grid.py::fetch                     # labels and I3D features into the volume
  modal volume put action-segmentation data/features/dinov2-vits14 /features/dinov2-vits14
  modal run scripts/modal_grid.py::bench                     # one epoch of split 1, to time a job
  modal run --detach scripts/modal_grid.py::grid             # 4 splits x 3 seeds
  modal run --detach scripts/modal_grid.py::centred          # centred I3D and ViT-B, the same jobs
  modal volume get action-segmentation /runs runs-modal      # results back
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/proj"
SEEDS = [1538574472, 0, 1]
K = 4  # motion offset chosen on split 1's training people (see runs/queue history and the README)
# NAME: FOLDER[:K][:center], as train_grouped.py takes them
SETTINGS = {"i3d": "i3d",
            "dinov2-vits14": "dinov2-vits14",
            f"dinov2-vits14-motion{K}": f"dinov2-vits14:{K}",
            "dinov2-vitb14": "dinov2-vitb14",
            "dinov2-vits14-centred": "dinov2-vits14:0:center",
            "i3d-centred": "i3d:0:center",
            "dinov2-vitb14-centred": "dinov2-vitb14:0:center"}
GRID = list(SETTINGS)[:5]
# Added after the grid: centring was the largest gain for DINOv2-S, so it is tried on I3D and ViT-B.
CENTRED = ["i3d-centred", "dinov2-vitb14-centred"]


app = modal.App("action-segmentation")
vol = modal.Volume.from_name("action-segmentation", create_if_missing=True)
# torch from the CUDA 12.6 index, the same build as the laptop runs (PyPI's default is a CUDA 13 build).
# MS-TCN sees a new sequence length with almost every video, and cuDNN builds plans for each new
# length; the plan cache is raised above the default 10,000 entries so a split's ~60,000 plans stay.
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cu126")
         .pip_install("numpy==2.5.4", "requests==2.34.2", "remotezip==0.12.6", "huggingface-hub==2.4.0")
         .env({"TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT": "200000"})
         .add_local_dir(ROOT / "mstcn", f"{REMOTE}/mstcn", ignore=["__pycache__"])
         .add_local_dir(ROOT / "scripts", f"{REMOTE}/scripts", ignore=["__pycache__"])
         .add_local_dir(ROOT / "checks", f"{REMOTE}/checks"))


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=4096, timeout=3 * 3600)
def fetch_data() -> str:
    sys.path.insert(0, f"{REMOTE}/scripts")
    import download_data as dd
    dd.fetch_zenodo(Path("/tmp/ms-tcn"), prefixes=("data/breakfast/groundTruth/", "data/breakfast/splits/",
                                                    "data/breakfast/mapping.txt"))
    shutil.copytree("/tmp/ms-tcn", "/vol/data/ms-tcn", dirs_exist_ok=True)
    vol.commit()
    dd.fetch_zenodo(Path("/tmp/i3d"), prefixes=("data/breakfast/features/",), flat=True, workers=4)
    shutil.copytree("/tmp/i3d", "/vol/features/i3d", dirs_exist_ok=True)
    vol.commit()
    labels = len(list(Path("/vol/data/ms-tcn/data/breakfast/groundTruth").glob("*.txt")))
    i3d = list(Path("/vol/features/i3d").glob("*.npy"))
    return f"{labels} label files, {len(i3d)} I3D files, {sum(p.stat().st_size for p in i3d) / 2**30:.2f} GiB"


def copy_folder(src: str, dst: Path, workers: int = 32) -> None:
    """Copy a folder of many files from the volume with many threads: one at a time took about
    0.55 s a file (31 min for the I3D and DINOv2-S folders)."""
    from concurrent.futures import ThreadPoolExecutor
    dst.mkdir(parents=True, exist_ok=True)
    files = [p for p in Path(src).iterdir() if p.is_file()]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda p: shutil.copyfile(p, dst / p.name), files))


@app.function(image=image, volumes={"/vol": vol}, cpu=4, memory=4096, timeout=3600)
def copy_speed(workers: int = 32) -> str:
    out = []
    for name in ("dinov2-vits14", "i3d"):
        t0 = time.time()
        copy_folder(f"/vol/features/{name}", Path("/tmp/features") / name, workers)
        size = sum(p.stat().st_size for p in (Path("/tmp/features") / name).iterdir()) / 2**30
        out.append(f"{name}: {size:.2f} GiB in {time.time() - t0:.0f} s with {workers} threads")
    return "; ".join(out)


@app.function(image=image, gpu="L4", volumes={"/vol": vol}, cpu=4, memory=16384, timeout=int(2.5 * 3600))
def train(split: int, seed: int, epochs: int = 50, names: tuple = tuple(GRID)) -> str:
    t0 = time.time()
    local = Path("/tmp/features")
    for folder in dict.fromkeys(SETTINGS[n].split(":")[0] for n in names):
        copy_folder(f"/vol/features/{folder}", local / folder)
    copied = time.time() - t0
    cmd = [sys.executable, "-I", f"{REMOTE}/scripts/train_grouped.py", "--data", "/vol/data/ms-tcn/data",
           "--dataset", "breakfast", "--exclude", f"{REMOTE}/checks/excluded.txt", "--gpu-fraction", "0.9",
           "--splits", str(split), "--seeds", str(seed), "--epochs", str(epochs), "--out", "/tmp/runs"]
    for n in names:
        cmd += ["--setting", f"{n}={local}/{SETTINGS[n]}"]
    run = subprocess.run(cmd, capture_output=True, text=True)
    out = Path("/vol/runs" if epochs == 50 else f"/vol/bench-{epochs}ep")
    if Path("/tmp/runs").exists():
        shutil.copytree("/tmp/runs", out, dirs_exist_ok=True)
        vol.commit()
    tail = "\n".join([line for line in (run.stdout + run.stderr).splitlines() if "Warning" not in line][-8:])
    return (f"split {split} seed {seed}: exit {run.returncode}, features copied in {copied:.0f} s, "
            f"total {time.time() - t0:.0f} s\n{tail}")


@app.function(image=image, cpu=0.25, memory=512, timeout=6 * 3600)
def run_grid(names: tuple = tuple(GRID)) -> list[str]:
    """Runs the 12 jobs from inside Modal, so the grid survives the local client going away
    (a detached run only keeps the last function the client called alive)."""
    out = []
    jobs = [(split, seed, 50, names) for seed in SEEDS for split in (1, 2, 3, 4)]
    for result in train.starmap(jobs, return_exceptions=True):
        out.append(result if isinstance(result, str) else f"FAILED: {result!r}")
        print(out[-1], flush=True)
    return out


@app.local_entrypoint()
def fetch():
    print(fetch_data.remote())


@app.function(image=image, gpu="L4", volumes={"/vol": vol}, cpu=4, memory=16384, timeout=1800)
def profile_steps() -> str:
    """Time per training step on this container, with and without the I3D input."""
    import platform
    import random
    sys.path.insert(0, REMOTE)
    import torch
    from torch import optim

    from mstcn.grouped import FeatureSource, GroupedBatchGenerator, GroupedTrainer, group_losses
    cpu = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), "?")
    for name in ("i3d", "dinov2-vits14"):
        copy_folder(f"/vol/features/{name}", Path("/tmp/features") / name)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    bf = Path("/vol/data/ms-tcn/data/breakfast")
    actions = {a.split()[1]: int(a.split()[0]) for a in (bf / "mapping.txt").read_text().split("\n")[:-1]}
    vids = (bf / "splits" / "train.split1.bundle").read_text().split("\n")[:-1][:400]
    out = [f"CPU {cpu}, {platform.python_version()}, torch {torch.__version__}, cuDNN {torch.backends.cudnn.version()}"]
    s = "/tmp/features/dinov2-vits14"
    for label, sets in (("I3D + 3 x DINOv2-S, 400 videos", [("/tmp/features/i3d", 0), (s, 0), (s, 4), (s, 0, True)]),):
        random.seed(0)
        source = FeatureSource(sets)
        raw = source.load(vids[0])
        dims = [x.shape[1] for x in source.inputs(raw, source.frames(raw), torch.device("cuda"))]
        tr = GroupedTrainer(dims, 48, 0)
        model = tr.model.cuda().train()
        opt = optim.Adam(model.parameters(), lr=0.0005)
        bg = GroupedBatchGenerator(actions, str(bf / "groundTruth") + "/", source)
        for p in range(2):
            bg.list_of_examples, bg.index, bg.prefetch = list(vids), 0, None
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            while bg.has_next():
                xs, y, m = bg.next_batch(torch.device("cuda"))
                opt.zero_grad()
                group_losses(model(xs, m), y, m, 48).sum().backward()
                opt.step()
            torch.cuda.synchronize()
            out.append(f"{label}, pass {p + 1}: {1000 * (time.perf_counter() - t0) / len(vids):.0f} ms a step")
    return "\n".join(out)


@app.local_entrypoint()
def copytest():
    print(copy_speed.remote())


@app.local_entrypoint()
def profile():
    print(profile_steps.remote())


@app.local_entrypoint()
def bench():
    print(train.remote(1, 0, epochs=1, names=tuple(n for n in GRID if n != "dinov2-vitb14")))


@app.local_entrypoint()
def grid():
    for line in run_grid.remote():
        print(line, flush=True)


@app.local_entrypoint()
def centred():
    for line in run_grid.remote(tuple(CENTRED)):
        print(line, flush=True)
