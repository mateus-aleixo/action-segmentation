# Step 1: videos against labels

`scripts/check_frames.py` matched every MS-TCN Breakfast label file to its video and decoded every
video to count frames (`frame_check.csv`, run 2026-11-21).

- 1,712 label files, 1,989 videos, 1,984 label and video pairs. Every label file has a video. The 5
  videos with no label file are not used.
- The stereo camera has two videos per label file (`_ch0` and `_ch1`). Of the 304 stereo label files,
  32 have one channel only and 29 have channels of different lengths. Within 2 frames of the labels:
  280 `ch0` videos, 238 `ch1` videos.
- All videos are 15 fps. The decoded video minus the label count is +2 frames for the median video
  (quartiles +2 and +2); 1,839 pairs are 1 to 2 frames apart, 123 are 3 to 15 apart, 22 are more.
  One video is 11 frames shorter than its labels.

## The rule used from step 3 on

1. Stereo: the `ch0` video when it exists, otherwise `ch1`.
2. The labels describe the first N frames of the video; frames after the last label are dropped.
3. A video shorter than its labels has its last frame's features repeated up to the label count.
4. The 20 label files whose chosen video differs from its labels by more than 15 frames (one second)
   are left out of both feature settings (DINOv2 and the same-code I3D re-run), so the comparison
   uses the same 1,692 videos. They are listed in `excluded.txt` and in `frame_check.csv` (column
   `diff`). The paper's numbers use all 1,712, so they stay in the results table only as a check on
   the I3D re-run.

## Are the DINOv2 features aligned with the labels?

`scripts/check_alignment.py`, run 2026-11-21 on 300 random videos and on split 1:

- Frame-to-frame change of the features against the frames where the label changes (pooled over
  the 300 videos): DINOv2 changes 2 frames after the label does, the released I3D features 1 frame
  after. Neither is off by more than a fraction of a second.
- The two change signals against each other: I3D trails DINOv2 by 11 frames in the median video,
  about half of I3D's 21-frame window.
- Nearest class mean (cosine to the mean feature of each action over split 1's training frames),
  frame by frame with no temporal model, on split 1's test frames: 11.0% right with DINOv2 ViT-S
  and 20.1% with I3D. Shifting the labels by up to 30 frames either way moves either number by less
  than one point, so this one says nothing about alignment.
