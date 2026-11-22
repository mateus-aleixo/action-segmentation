# How it was run

## MS-TCN

The official code (github.com/yabufarha/ms-tcn, commit 33ed91c), ported to Python 3 in
[mstcn](../mstcn): 4 stages of 10 dilated layers with 64 filters, cross-entropy plus 0.15 times
the clamped smoothing loss, Adam at 0.0005, 50 epochs, one video per batch, and the official
seeding and evaluation. Only the last epoch's weights are kept, since the official predict step
loads those. On GTEA with the released I3D features (4 splits by 3 seeds) the port scores
F1@{10,25,50} 86.2, 83.7 and 70.4, Edit 81.0 and accuracy 76.7, against the paper's 85.8, 83.4,
69.8, 79.0 and 76.3 (Table 9, without fine-tuning).

## Data

- Labels, splits, the class mapping and the I3D features come from the MS-TCN Zenodo record
  (3625992). The 30 GB zip is never downloaded whole: [scripts/download_data.py](../scripts/download_data.py)
  reads its directory and fetches the needed members with range requests.
- Videos come from the Hugging Face copy of Breakfast (CVML-TueAI/Breakfast-Actions). Every
  video's decoded frame count was compared with its number of labels
  ([checks/frame_check.csv](../checks/frame_check.csv)). The labels cover a video's first frames
  and the trailing ones are dropped; the 20 label files whose video is more than 15 frames (one
  second) off are left out of every setting ([checks/excluded.txt](../checks/excluded.txt)),
  which leaves 1,692 of 1,712 videos. The paper's row uses all 1,712.
- DINOv2 features line up with the labels and with I3D's: [checks/README.md](../checks/README.md).

## Features

- DINOv2, frozen, from torch.hub: the class token of every frame at 224 × 294 pixels (the whole
  320 × 240 frame, 16 × 21 patches), with ImageNet normalisation, stored in fp16
  ([scripts/extract_dinov2.py](../scripts/extract_dinov2.py)). ViT-S/14 gives 384 values a frame
  and ViT-B/14 768. ViT-S ran at about 770 frames a second on an RTX 3060 laptop GPU.
- Motion cue: each frame's features next to their difference from the frame k earlier (the
  first k frames use the first frame). k was chosen on split 1's training people only, training
  on P29 to P54 and validating on P16 to P28 with the official seed, by a rule fixed before any
  result: the highest mean of F1@10, 25 and 50, ties to the smaller k. The means were 28.5
  without motion, 27.1 for k = 2, 31.3 for k = 4 and 27.4 for k = 8 (runs/breakfast/val-*). On
  the test splits k = 4 added 0.5 points.
- Centred: each video's mean feature is subtracted from all its frames. It was added after the
  validation runs, where ViT-S got 88% of training frames right and 32% of held-out ones, and was
  not in the plan. Centred I3D and ViT-B were run after the main grid, once centring had turned
  out to be the largest gain for ViT-S.

## Training several settings at once

MS-TCN's kernels are small, so a laptop GPU spends most of each step launching them. The settings
of one split and seed therefore train side by side in one process
([mstcn/grouped.py](../mstcn/grouped.py)): each model is a slice of grouped convolutions, with its
own first layer for its input size, the official initial weights for the seed and the official
video order. The official losses are summed and Adam updates every weight on its own, so each
model trains as it would alone apart from its dropout masks.
[tests/test_grouped.py](../tests/test_grouped.py) checks outputs, losses, gradients, one Adam step,
the initial weights and the batch order against the official code in fp32, and that the inputs
built on the GPU equal the CPU loader's. Four models took 74 ms a step on the RTX 3060, about
what one took.

## Compute

The 24 jobs (12 splits and seeds, twice: the five settings of the main grid, then centred I3D and
ViT-B) ran on Modal's L4 GPUs, one container each ([scripts/modal_grid.py](../scripts/modal_grid.py)):
17.9 container hours in all, about USD 22, covered by Modal's free monthly credit. The containers use the
same CUDA 12.6 build of PyTorch as the laptop, with cuDNN's plan cache raised to 200,000 entries,
since nearly every video brings a new sequence length.

## Paired differences

F1@50, paired by split and seed: the difference averaged over the four splits for each seed, then
the mean ± standard deviation over the three seeds ([results/differences.csv](../results/differences.csv)).

| comparison | F1@50 |
|---|---|
| DINOv2 ViT-S minus I3D | −16.0 ± 0.7 |
| ViT-S with motion minus ViT-S | +0.5 ± 0.3 |
| ViT-B minus ViT-S | +5.6 ± 1.0 |
| ViT-S centred minus ViT-S | +7.8 ± 1.9 |
| ViT-B centred minus ViT-B | +5.1 ± 1.4 |
| I3D centred minus I3D | +1.2 ± 1.0 |
| ViT-B centred minus I3D | −5.3 ± 2.9 |
| ViT-B centred minus I3D centred | −6.4 ± 1.8 |

Frames right in the training videos at the last epoch, same averaging
([results/summary.csv](../results/summary.csv)): I3D 92.6%, centred 95.3%; ViT-S 83.1%, with
motion 84.5%, centred 93.2%; ViT-B 86.4%, centred 94.7%.

## Frames given another breakfast's action

A predicted frame counts here when its action never occurs in the ground truth of its video's
activity (milk, coffee and so on), over all test frames of the 12 runs
([scripts/other_activity.py](../scripts/other_activity.py),
[results/other_activity.csv](../results/other_activity.csv)): I3D 14.1%, centred 13.8%; ViT-S
33.6%, with motion 34.3%, centred 26.7%; ViT-B 27.7%, centred 21.9%. Centring lowers this share
for both DINOv2 models.

## Variance between videos

[scripts/variance_split.py](../scripts/variance_split.py) splits each feature set's variance over
all frames of the 1,692 videos (summed over dimensions) into the part the per-video means carry
and the rest. The per-video means carry 32.6% for I3D, 75.3% for ViT-S and 72.2% for ViT-B
([results/variance_split.csv](../results/variance_split.csv)). That part is what centring removes.

## Per action

[scripts/per_action.py](../scripts/per_action.py) scores each action with the official F1 rule over
the test videos of all 12 runs ([results/per_action.csv](../results/per_action.csv)); the counts
per action add up to the official totals. Motion per action is the mean grey-level change between
consecutive frames over that action's frames ([scripts/frame_motion.py](../scripts/frame_motion.py)).
It measures pixel change, so fetching a cup scores high and stirring scores low. The
[figure](figures/per_action.png) leaves out the background class and the 10 actions with fewer
than 50 segments in the data. Over the other 37, what ViT-S loses against I3D does not follow
motion (Spearman ρ = −0.04), and neither does what the motion cue adds (ρ = −0.20). The largest
losses are cutting an orange (−38 points), squeezing it (−38) and stirring milk (−32).

## Tables and figures

    python -I scripts/results_table.py runs/breakfast i3d i3d-centred dinov2-vits14 dinov2-vits14-motion4 dinov2-vits14-centred dinov2-vitb14 dinov2-vitb14-centred --ref=i3d --ref=dinov2-vits14 --ref=dinov2-vitb14 --ref=i3d-centred
    python -I scripts/per_action.py data runs/breakfast i3d i3d-centred dinov2-vits14 dinov2-vits14-motion4 dinov2-vits14-centred dinov2-vitb14 dinov2-vitb14-centred
    python -I scripts/other_activity.py data runs/breakfast i3d i3d-centred dinov2-vits14 dinov2-vits14-motion4 dinov2-vits14-centred dinov2-vitb14 dinov2-vitb14-centred
    python -I scripts/variance_split.py data/ms-tcn/data/breakfast/groundTruth checks/excluded.txt i3d=data/ms-tcn/data/breakfast/features dinov2-vits14=data/features/dinov2-vits14 dinov2-vitb14=data/features/dinov2-vitb14
    python -I scripts/figures.py timelines runs/breakfast split1 1538574472 milk dinov2-vits14 "i3d=I3D" "dinov2-vits14=DINOv2 ViT-S" "dinov2-vitb14-centred=DINOv2 ViT-B, centred"
    python -I scripts/figures.py per-action "i3d=I3D" "dinov2-vits14=DINOv2 ViT-S" "dinov2-vits14-motion4=DINOv2 ViT-S with motion"
