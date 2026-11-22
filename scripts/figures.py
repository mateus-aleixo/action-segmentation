"""README figures.

timelines: ground truth and each setting's prediction for three test videos of one activity. The
videos are the ones at the 25th, 50th and 75th percentile of the reference setting's F1@50 among
that activity's test videos of the split, so they are typical, not picked.
per-action: per-action F1@50 differences against how much each action moves (results/per_action.csv).

Usage:
  python -I scripts/figures.py timelines runs/breakfast split1 1538574472 coffee dinov2-vits14 \
      "i3d=I3D" "dinov2-vits14=DINOv2 ViT-S" "dinov2-vits14-motion4=DINOv2 ViT-S + motion"
  python -I scripts/figures.py per-action "i3d=I3D" "dinov2-vits14=DINOv2 ViT-S" \
      "dinov2-vits14-motion4=with motion"
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mstcn.eval import f_score  # noqa: E402

DATA = ROOT / "data" / "ms-tcn" / "data" / "breakfast"
OUT = ROOT / "docs" / "figures"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Any two actions can touch in a timeline, so the colours must stay apart in every pair, not only
# neighbours; of the palette's 4-colour sets this one passes every pair check (colour-blind
# included). So timelines show activities with at most 4 actions, and the ground-truth strip also
# carries the action names.
TIMELINE = ["#2a78d6", "#1baf7a", "#eda100", "#e34948"]
TEXT_ON = {"#eda100": INK}
BACKGROUND, OTHER = "#e1e0d9", "#52514e"
FPS = 15

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "text.color": INK,
                     "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def mapping():
    pairs = [line.split() for line in (DATA / "mapping.txt").read_text().splitlines() if line.strip()]
    return {int(i): name for i, name in pairs}


def segments(labels):
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, change]
    return labels[starts], starts, np.r_[change, len(labels)]


def timelines(runs: Path, split: str, seed: str, activity: str, ref: str, rows: list[str]) -> None:
    names = mapping()
    index = {v: k for k, v in names.items()}
    rows = [r.split("=", 1) for r in rows]
    preds = {n: np.load(runs / n / split / f"seed{seed}" / "predictions.npz") for n, _ in rows}
    vids = sorted(v for v in preds[ref].files if v.endswith(f"_{activity}.txt"))

    def gt(v):
        return np.array([index[a] for a in (DATA / "groundTruth" / v).read_text().split("\n")[:-1]])

    scores = []
    for v in vids:
        tp, fp, fn = f_score([names[i] for i in preds[ref][v]], [names[i] for i in gt(v)], 0.5)
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    order = np.argsort(scores, kind="stable")
    picks = [vids[order[int(round(q * (len(vids) - 1)))]] for q in (0.25, 0.5, 0.75)]

    actions = sorted({a for v in vids for a in np.unique(gt(v))} - {index["SIL"]})
    assert len(actions) <= len(TIMELINE), f"{activity} has {len(actions)} actions; pick one with at most 4"
    colors = {a: TIMELINE[i] for i, a in enumerate(actions)}
    colors[index["SIL"]] = BACKGROUND

    fig, axes = plt.subplots(len(picks), 1, figsize=(8, 1.0 + 1.05 * len(picks) * (len(rows) + 1) / 4),
                             gridspec_kw={"hspace": 0.55})
    used_other = False
    for ax, v, q in zip(np.atleast_1d(axes), picks, (25, 50, 75)):
        strips = [("Ground truth", gt(v))] + [(label, preds[n][v].astype(int)) for n, label in rows]
        total = len(gt(v))
        for r, (label, lab) in enumerate(strips):
            y = len(strips) - 1 - r
            for a, s, e in zip(*segments(lab)):
                c = colors.get(a, OTHER)
                used_other |= a not in colors
                ax.barh(y, (e - s) / FPS, left=s / FPS, height=0.72, color=c, linewidth=0)
                if r == 0 and a in colors and a != index["SIL"] and (e - s) / total > 0.11:
                    ax.text((s + e) / 2 / FPS, y, names[a].replace("_", " "), ha="center", va="center",
                            fontsize=7, color=TEXT_ON.get(c, "#ffffff"))
        ax.set_yticks(range(len(strips)), [s[0] for s in strips][::-1])
        ax.tick_params(axis="y", length=0, labelcolor=INK2)
        ax.set_xlim(0, len(gt(v)) / FPS)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title(f"{v[:-4]}  ({q}th percentile of {dict(rows)[ref]} F1@50 among {len(vids)} test videos)",
                     loc="left", fontsize=9, color=INK2)
    np.atleast_1d(axes)[-1].set_xlabel("seconds")
    handles = [Patch(color=colors[a], label=names[a].replace("_", " ")) for a in actions]
    handles.append(Patch(color=BACKGROUND, label="SIL (background)"))
    if used_other:
        handles.append(Patch(color=OTHER, label="an action of another activity"))
    fig.legend(handles=handles, loc="upper center", ncol=min(6, len(handles)), frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.2)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"timelines_{activity}.png", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT / f'timelines_{activity}.png'}: {', '.join(p[:-4] for p in picks)}")


def label_points(ax, points) -> None:
    """Label (x, y, text) points in order, each at the first of four offsets where it overlaps no
    earlier label and stays inside the axes; a label that fits nowhere is left out."""
    renderer = ax.figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    placed = []
    for x, y, text in points:
        for dx, dy, ha, va in ((4, 2, "left", "bottom"), (4, -2, "left", "top"),
                               (-4, 2, "right", "bottom"), (-4, -2, "right", "top")):
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=7.5, color=INK2)
            box = t.get_window_extent(renderer).expanded(1.04, 1.15)
            if (not any(box.overlaps(p) for p in placed) and frame.x0 <= box.x0 and box.x1 <= frame.x1
                    and frame.y0 <= box.y0 and box.y1 <= frame.y1):
                placed.append(box)
                break
            t.remove()


def per_action(rows: list[str], min_segments: int = 50) -> None:
    """Two panels: the second setting minus the first, and the third minus the second, per action.
    The background class and actions with fewer than min_segments segments in the data are left
    out (the rare ones score 0 in every setting)."""
    (ref, ref_label), (a, a_label), (b, b_label) = (r.split("=", 1) for r in rows)
    t = pd.read_csv(ROOT / "results" / "per_action.csv")
    t = t[(t["action"] != "SIL") & (t["segments"] >= min_segments)]
    panels = [(f"{a} f1@50", f"{ref} f1@50", f"{a_label} minus {ref_label}"),
              (f"{b} f1@50", f"{a} f1@50", f"{b_label} minus {a_label}")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    for ax, (hi, lo, title) in zip(axes, panels):
        d = t[["action", "motion", "segments"]].assign(diff=t[hi] - t[lo]).dropna()
        rho = d["motion"].rank().corr(d["diff"].rank())
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.scatter(d["motion"], d["diff"], s=22, color=SERIES[0], edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.set_title(f"{title}\nSpearman ρ = {rho:.2f}, {len(d)} actions", loc="left", fontsize=9, color=INK)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("motion: mean grey-level change between frames")
    axes[0].set_ylabel("F1@50 difference (points)")
    fig.tight_layout()
    for ax, (hi, lo, _) in zip(axes, panels):
        d = t[["action", "motion"]].assign(diff=t[hi] - t[lo]).dropna()
        ends = pd.concat([d.nsmallest(3, "diff"), d.nlargest(3, "diff")])
        label_points(ax, [(r["motion"], r["diff"], r["action"].replace("_", " ")) for _, r in ends.iterrows()])
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "per_action.png", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT / 'per_action.png'}")


if __name__ == "__main__":
    if sys.argv[1] == "timelines":
        timelines(Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7:])
    else:
        per_action(sys.argv[2:5])
