# Segmentation metrics ported to Python 3 from the official MS-TCN eval.py (github.com/yabufarha/ms-tcn,
# commit 33ed91c, MIT license with the Commons Clause, see LICENSE-ms-tcn; that file is adapted from
# github.com/colincsl/TemporalConvolutionalNetworks). Same rules, including bg_class=["background"]
# for every dataset, as the official evaluation runs it.
import numpy as np


def get_labels_start_end_time(frame_wise_labels, bg_class=("background",)):
    labels, starts, ends = [], [], []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row, n_col = len(p), len(y)
    D = np.zeros([m_row + 1, n_col + 1], float)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i
    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)
    if norm:
        return (1 - D[-1, -1] / max(m_row, n_col)) * 100
    return D[-1, -1]


def edit_score(recognized, ground_truth, norm=True, bg_class=("background",)):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=("background",)):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
    tp, fp = 0, 0
    hits = np.zeros(len(y_label))
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * np.array([p_label[j] == y_label[x] for x in range(len(y_label))])
        idx = np.array(IoU).argmax()
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def evaluate(recognitions, gt_path, vid_list):
    """Acc, Edit and F1@{10,25,50} over a test list, exactly as the official main() sums them."""
    overlap = [0.1, 0.25, 0.5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    correct = total = 0
    edit = 0.0
    for vid in vid_list:
        with open(gt_path + vid, "r") as fh:
            gt_content = fh.read().split("\n")[0:-1]
        recog_content = recognitions[vid]
        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1
        edit += edit_score(recog_content, gt_content)
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
    result = {"acc": 100 * float(correct) / total, "edit": edit / len(vid_list)}
    for s, name in enumerate(["f1@10", "f1@25", "f1@50"]):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
        f1 = 2.0 * (precision * recall) / (precision + recall)
        result[name] = float(np.nan_to_num(f1) * 100)
    return result
