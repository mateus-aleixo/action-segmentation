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


def settings(features: str, vitb: bool = True) -> list[str]:
    out = [f"i3d={features}/i3d",
           f"dinov2-vits14={features}/dinov2-vits14",
           f"dinov2-vits14-motion{K}={features}/dinov2-vits14:{K}",
           f"dinov2-vits14-centred={features}/dinov2-vits14:0:center"]
    if vitb:
        out.insert(3, f"dinov2-vitb14={features}/dinov2-vitb14")
    return out


app = modal.App("action-segmentation")
vol = modal.Volume.from_name("action-segmentation", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.14.1", "numpy==2.5.4", "requests==2.34.2", "remotezip==0.12.6",
                      "huggingface-hub==2.4.0")
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


@app.function(image=image, gpu="L4", volumes={"/vol": vol}, cpu=4, memory=16384, timeout=int(2.5 * 3600))
def train(split: int, seed: int, epochs: int = 50, vitb: bool = True) -> str:
    t0 = time.time()
    local = Path("/tmp/features")
    for name in ("i3d", "dinov2-vits14") + (("dinov2-vitb14",) if vitb else ()):
        shutil.copytree(f"/vol/features/{name}", local / name)
    copied = time.time() - t0
    cmd = [sys.executable, "-I", f"{REMOTE}/scripts/train_grouped.py", "--data", "/vol/data/ms-tcn/data",
           "--dataset", "breakfast", "--exclude", f"{REMOTE}/checks/excluded.txt", "--gpu-fraction", "0.9",
           "--splits", str(split), "--seeds", str(seed), "--epochs", str(epochs), "--out", "/tmp/runs"]
    for s in settings(str(local), vitb):
        cmd += ["--setting", s]
    run = subprocess.run(cmd, capture_output=True, text=True)
    out = Path("/vol/runs" if epochs == 50 else f"/vol/bench-{epochs}ep")
    if Path("/tmp/runs").exists():
        shutil.copytree("/tmp/runs", out, dirs_exist_ok=True)
        vol.commit()
    tail = "\n".join([line for line in (run.stdout + run.stderr).splitlines() if "Warning" not in line][-8:])
    return (f"split {split} seed {seed}: exit {run.returncode}, features copied in {copied:.0f} s, "
            f"total {time.time() - t0:.0f} s\n{tail}")


@app.function(image=image, cpu=0.25, memory=512, timeout=6 * 3600)
def run_grid(vitb: bool = True) -> list[str]:
    """Runs the 12 jobs from inside Modal, so the grid survives the local client going away
    (a detached run only keeps the last function the client called alive)."""
    out = []
    jobs = [(split, seed, 50, vitb) for seed in SEEDS for split in (1, 2, 3, 4)]
    for result in train.starmap(jobs, return_exceptions=True):
        out.append(result if isinstance(result, str) else f"FAILED: {result!r}")
        print(out[-1], flush=True)
    return out


@app.local_entrypoint()
def fetch():
    print(fetch_data.remote())


@app.local_entrypoint()
def bench():
    print(train.remote(1, 0, epochs=1, vitb=False))


@app.local_entrypoint()
def grid():
    for line in run_grid.remote():
        print(line, flush=True)
