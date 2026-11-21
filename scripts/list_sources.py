"""List what the two data sources hold, before downloading anything."""
from collections import Counter

from huggingface_hub import HfApi
from remotezip import RemoteZip

HF_REPO = "CVML-TueAI/Breakfast-Actions"
ZENODO_ZIP = "https://zenodo.org/records/3625992/files/data.zip?download=1"


def list_hf():
    info = HfApi().dataset_info(HF_REPO, files_metadata=True)
    counts, sizes = Counter(), Counter()
    for s in info.siblings:
        top = s.rfilename.split("/")[0] if "/" in s.rfilename else "(root)"
        ext = s.rfilename.rsplit(".", 1)[-1] if "." in s.rfilename else ""
        key = f"{top} .{ext}"
        counts[key] += 1
        sizes[key] += s.size or 0
    print(f"HF {HF_REPO}: {len(info.siblings)} files, {sum(sizes.values()) / 1e9:.2f} GB")
    for key in sorted(counts):
        print(f"  {key:40s} {counts[key]:6d} files {sizes[key] / 1e9:8.3f} GB")
    print("  root files:", [s.rfilename for s in info.siblings if "/" not in s.rfilename][:20])
    print("  first paths:", [s.rfilename for s in info.siblings[:5]])


def list_zenodo():
    with RemoteZip(ZENODO_ZIP) as z:
        members = z.infolist()
        counts, sizes = Counter(), Counter()
        for m in members:
            parts = m.filename.split("/")
            key = "/".join(parts[:3]) if len(parts) > 3 else m.filename
            counts[key] += 1
            sizes[key] += m.file_size
        print(f"Zenodo zip: {len(members)} members, {sum(sizes.values()) / 1e9:.2f} GB uncompressed")
        for key in sorted(counts):
            print(f"  {key:50s} {counts[key]:6d} {sizes[key] / 1e6:10.1f} MB")


if __name__ == "__main__":
    list_hf()
    list_zenodo()
