"""Run jobs from a queue file, a few at a time, while memory allows.

Each non-empty line of the queue file that does not start with # is one job: optional NAME=VALUE
environment settings, then a script in scripts/ and its arguments (a line that starts with an
option is for train_mstcn.py). The file is read again whenever a slot frees, so jobs can be added
while it runs. Training scripts skip runs whose metrics.json exists, so a finished job ends at once.
A line STOP waits for the running jobs and ends the runner, to make room for a decision before the
next jobs. The number of parallel jobs comes from the first line of <queue>.slots if that file
exists (else --slots), read again before each start. A job starts only when Windows has at least
--min-commit-gb of commit left: every CUDA process here commits about 3.5 GB, and running out of
commit killed four runs and a download on 2026-11-21. Logs go to runs/_logs/. The runner stops
when every job has run.

Usage: python -I scripts/run_queue.py runs/queue.txt --slots 1
"""
import os
import argparse
import ctypes
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def commit_left_gb() -> float:
    s = MemoryStatus()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPageFile / 2**30


def read_jobs(queue: Path) -> list[str]:
    return [line.strip() for line in queue.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")]


def slots(queue: Path, default: int) -> int:
    f = queue.with_suffix(".slots")
    if f.exists():
        try:
            return int(f.read_text().split()[0])
        except (ValueError, IndexError):
            pass
    return default


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--min-commit-gb", type=float, default=4.5)
    args = ap.parse_args()
    queue = Path(args.queue)
    logs = ROOT / "runs" / "_logs"
    logs.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    started: set[str] = set()
    running: dict[str, subprocess.Popen] = {}
    failed = 0
    while True:
        for job, p in list(running.items()):
            if p.poll() is not None:
                del running[job]
                log = logs / f"{hashlib.sha1(job.encode()).hexdigest()[:10]}.log"
                last = [line for line in log.read_text(errors="replace").splitlines() if line.strip()][-1:]
                if p.returncode:
                    failed += 1
                print(f"{stamp()} exit {p.returncode}: {job}\n    {last[0] if last else ''}", flush=True)
        pending = [j for j in read_jobs(queue) if j not in started]
        if not pending and not running:
            break
        if pending and pending[0] == "STOP":
            if not running:
                print(f"{stamp()} STOP reached", flush=True)
                break
        elif pending and len(running) < slots(queue, args.slots):
            if commit_left_gb() >= args.min_commit_gb:
                job = pending[0]
                log = logs / f"{hashlib.sha1(job.encode()).hexdigest()[:10]}.log"
                tokens = job.split()
                env = dict(os.environ)
                while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
                    key, value = tokens.pop(0).split("=", 1)
                    env[key] = value
                script = tokens.pop(0) if tokens and tokens[0].endswith(".py") else "train_mstcn.py"
                with log.open("w") as fh:
                    fh.write(f"# {job}\n")
                    fh.flush()
                    running[job] = subprocess.Popen(
                        [python, "-I", str(ROOT / "scripts" / script), *tokens],
                        cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
                started.add(job)
                print(f"{stamp()} start ({len(running)} running, {commit_left_gb():.1f} GB commit left): "
                      f"{job}  -> {log.name}", flush=True)
                time.sleep(30)  # let its CUDA context settle before the next memory check
                continue
        time.sleep(10)
    print(f"{stamp()} queue done, {len(started)} jobs, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
