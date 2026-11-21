"""Step 1 downloads, resumable.

From the MS-TCN Zenodo zip (record 3625992): the Breakfast labels, splits and mapping, and
everything for GTEA (I3D features included). The zip is 34 GB, so only the byte ranges of the
wanted members are fetched, and each contiguous run of members comes in one request: one request
per member hit Zenodo's rate limit (HTTP 429) after about 450 files on 2026-11-21. Every member is
checked against the CRC in the zip's central directory.

Then the Breakfast videos and their split files from Hugging Face (CVML-TueAI/Breakfast-Actions:
videos/ and data/; the web-dataset tars hold the same videos and are skipped), retried with a
pause when the Hub rate-limits.

Step 4 adds --breakfast-i3d: only the Breakfast I3D features (1,712 files, 29 GB), written flat
into the given folder, for the same-code I3D re-run.

Usage: python -I scripts/download_data.py data
       python -I scripts/download_data.py --breakfast-i3d cache/features/i3d
"""
import struct
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub import snapshot_download
from remotezip import RemoteZip

HF_REPO = "CVML-TueAI/Breakfast-Actions"
ZENODO_ZIP = "https://zenodo.org/records/3625992/files/data.zip?download=1"
ZIP_PREFIXES = (
    "data/breakfast/groundTruth/",
    "data/breakfast/splits/",
    "data/breakfast/mapping.txt",
    "data/gtea/",
)
MAX_RUN_BYTES = 64 * 1024 * 1024
LOCAL_HEADER = b"PK\x03\x04"


def get_range(session: requests.Session, start: int, end: int) -> bytearray:
    """Bytes start..end (inclusive) of the zip, streamed into one buffer, waiting out HTTP 429 and
    dropped connections."""
    size = end - start + 1
    for attempt in range(12):
        try:
            with session.get(ZENODO_ZIP, headers={"Range": f"bytes={start}-{end}"}, timeout=120,
                             stream=True) as r:
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", "60")) + 5
                    print(f"zenodo 429, waiting {wait} s (attempt {attempt + 1})", flush=True)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                if r.status_code != 206:
                    raise RuntimeError(f"bad range answer: HTTP {r.status_code}")
                buf = bytearray(size)
                pos = 0
                for chunk in r.iter_content(1 << 20):
                    if pos + len(chunk) > size:
                        raise RuntimeError("range answer longer than asked")
                    buf[pos:pos + len(chunk)] = chunk
                    pos += len(chunk)
                if pos != size:
                    raise requests.ConnectionError(f"short range answer: {pos} of {size} bytes")
                return buf
        except requests.RequestException as e:
            print(f"zenodo request failed ({type(e).__name__}), retrying in 60 s", flush=True)
            time.sleep(60)
    raise RuntimeError("zenodo kept failing")


def safe_target(out: Path, name: str) -> Path:
    target = (out / name).resolve()
    if out.resolve() not in target.parents:
        raise RuntimeError(f"member escapes the output folder: {name}")
    return target


def extract_member(buf: bytes, base: int, m, out: Path, flat: bool = False) -> None:
    off = m.header_offset - base
    if buf[off:off + 4] != LOCAL_HEADER:
        raise RuntimeError(f"no local header for {m.filename}")
    name_len, extra_len = struct.unpack("<HH", buf[off + 26:off + 30])
    start = off + 30 + name_len + extra_len
    comp = buf[start:start + m.compress_size]
    if m.compress_type == 8:
        data = zlib.decompress(comp, -15)
    elif m.compress_type == 0:
        data = comp
    else:
        raise RuntimeError(f"compression {m.compress_type} not handled: {m.filename}")
    if len(data) != m.file_size or zlib.crc32(data) != m.CRC:
        raise RuntimeError(f"size or CRC mismatch: {m.filename}")
    target = safe_target(out, Path(m.filename).name if flat else m.filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_bytes(data)
    partial.replace(target)


def runs_of(members, next_offset, max_bytes=MAX_RUN_BYTES):
    """Group members (sorted by offset) into contiguous runs under max_bytes."""
    run = []
    for m in members:
        if run and (next_offset[run[-1].header_offset] != m.header_offset
                    or next_offset[m.header_offset] - run[0].header_offset > max_bytes):
            yield run
            run = []
        run.append(m)
    if run:
        yield run


def fetch_zenodo(out: Path, prefixes=ZIP_PREFIXES, flat: bool = False, max_bytes=MAX_RUN_BYTES,
                 workers: int = 1) -> None:
    for attempt in range(12):
        try:
            with RemoteZip(ZENODO_ZIP) as z:  # reads only the central directory
                everything = sorted(z.infolist(), key=lambda m: m.header_offset)
                central_start = z.start_dir
            break
        except Exception as e:
            if "429" not in str(e) or attempt == 11:
                raise
            print("zenodo 429 on the central directory, waiting 90 s", flush=True)
            time.sleep(90)
    offsets = [m.header_offset for m in everything] + [central_start]
    next_offset = {offsets[i]: offsets[i + 1] for i in range(len(everything))}
    wanted = [m for m in everything
              if not m.is_dir() and m.filename.startswith(prefixes)]

    def local(m) -> Path:
        return out / (Path(m.filename).name if flat else m.filename)

    todo = [m for m in wanted if not (local(m).exists() and local(m).stat().st_size == m.file_size)]
    print(f"zenodo: {len(wanted)} wanted, {len(todo)} still to fetch", flush=True)
    sessions = threading.local()
    lock = threading.Lock()
    progress = {"done": 0, "bytes": 0}
    t0 = time.time()

    def fetch_run(run) -> None:
        if not hasattr(sessions, "s"):
            sessions.s = requests.Session()
        start = run[0].header_offset
        end = next_offset[run[-1].header_offset] - 1
        buf = get_range(sessions.s, start, end)
        for m in run:
            extract_member(buf, start, m, out, flat)
        with lock:
            progress["done"] += len(run)
            progress["bytes"] += end - start + 1
            print(f"zenodo {progress['done']}/{len(todo)} "
                  f"({progress['bytes'] / 1e6 / (time.time() - t0):.1f} MB/s)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in [pool.submit(fetch_run, run) for run in runs_of(todo, next_offset, max_bytes)]:
            f.result()
    print(f"zenodo done: {len(wanted)} files in {out}", flush=True)


def fetch_hf(dest: Path) -> None:
    out = dest / "breakfast-videos"
    for attempt in range(8):
        try:
            snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=out,
                              allow_patterns=["videos/*", "data/*", "README.md"], max_workers=4)
            break
        except Exception as e:  # rate limit or a dropped connection: resume after a pause
            print(f"hf attempt {attempt + 1} failed: {str(e)[:200]}; waiting 300 s", flush=True)
            time.sleep(300)
    n = len(list((out / "videos").glob("*.avi")))
    print(f"hf done: {n} videos in {out / 'videos'}", flush=True)


def main() -> None:
    t0 = time.time()
    if sys.argv[1] == "--breakfast-i3d":
        out = Path(sys.argv[2])
        out.mkdir(parents=True, exist_ok=True)
        fetch_zenodo(out, prefixes=("data/breakfast/features/",), flat=True, workers=3)
    else:
        dest = Path(sys.argv[1])
        dest.mkdir(parents=True, exist_ok=True)
        fetch_zenodo(dest / "ms-tcn")
        fetch_hf(dest)
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
