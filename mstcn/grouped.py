# Several MS-TCN models trained side by side on the same videos, for speed: on this laptop MS-TCN's
# kernels are so small that a run is bound by kernel launches, and two runs in parallel processes
# were slower than one after the other. Here G models (one per feature setting) share every launch.
#
# Each model is the official MultiStageModel (model.py). Its layers are slices of grouped
# convolutions (groups=G): a grouped convolution computes each group's output channels from that
# group's input channels only, so the models never mix. The first layer of each model is its own
# convolution, because the feature sizes differ. The loss of each model is the official one; the
# total is their sum, and since the models share no parameter each model gets exactly its own
# gradient. Adam works element by element, so one Adam over the grouped tensors equals one Adam per
# model. Each model starts from the weights the official code draws for the run's seed, and the
# videos come in the official order for that seed. Only the dropout masks differ from a single run,
# because all models draw them from one random stream.
import queue
import random
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .model import MultiStageModel


class GroupedLayer(nn.Module):
    def __init__(self, dilation, channels, groups):
        super().__init__()
        self.conv_dilated = nn.Conv1d(channels * groups, channels * groups, 3, padding=dilation,
                                      dilation=dilation, groups=groups)
        self.conv_1x1 = nn.Conv1d(channels * groups, channels * groups, 1, groups=groups)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class GroupedStage(nn.Module):
    def __init__(self, num_layers, num_f_maps, num_classes, groups, dims=None):
        super().__init__()
        if dims is None:  # stages 2 to 4 read the previous stage's class probabilities
            self.conv_in = nn.Conv1d(num_classes * groups, num_f_maps * groups, 1, groups=groups)
        else:  # stage 1 reads each model's own features
            self.firsts = nn.ModuleList([nn.Conv1d(d, num_f_maps, 1) for d in dims])
        self.layers = nn.ModuleList([GroupedLayer(2 ** i, num_f_maps, groups) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps * groups, num_classes * groups, 1, groups=groups)

    def forward(self, x, mask):
        if isinstance(x, list):
            out = torch.cat([conv(xi) for conv, xi in zip(self.firsts, x)], dim=1)
        else:
            out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out, mask)
        return self.conv_out(out) * mask[:, 0:1, :]


class GroupedModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dims, num_classes):
        super().__init__()
        self.groups, self.num_classes = len(dims), num_classes
        self.num_stages, self.num_layers, self.num_f_maps, self.dims = num_stages, num_layers, num_f_maps, dims
        self.stage1 = GroupedStage(num_layers, num_f_maps, num_classes, self.groups, dims)
        self.stages = nn.ModuleList([GroupedStage(num_layers, num_f_maps, num_classes, self.groups)
                                     for _ in range(num_stages - 1)])

    def forward(self, xs, mask):
        b, t = mask.shape[0], mask.shape[2]
        out = self.stage1(xs, mask)
        outputs = [out]
        for s in self.stages:
            probs = F.softmax(out.view(b, self.groups, self.num_classes, t), dim=2).view(b, -1, t)
            out = s(probs * mask[:, 0:1, :], mask)
            outputs.append(out)
        return torch.stack(outputs)  # (stages, batch, groups * classes, frames)

    def _pairs(self, g):
        """(grouped tensor slice, official parameter name) for model g."""
        f, c = self.num_f_maps, self.num_classes
        pairs = [(self.stage1.firsts[g].weight, "stage1.conv_1x1.weight", None),
                 (self.stage1.firsts[g].bias, "stage1.conv_1x1.bias", None)]
        for name, stage in [("stage1", self.stage1)] + [(f"stages.{i}", s) for i, s in enumerate(self.stages)]:
            if name != "stage1":
                pairs += [(stage.conv_in.weight, f"{name}.conv_1x1.weight", (g * f, (g + 1) * f)),
                          (stage.conv_in.bias, f"{name}.conv_1x1.bias", (g * f, (g + 1) * f))]
            for i, layer in enumerate(stage.layers):
                for part in ("conv_dilated", "conv_1x1"):
                    conv = getattr(layer, part)
                    pairs += [(conv.weight, f"{name}.layers.{i}.{part}.weight", (g * f, (g + 1) * f)),
                              (conv.bias, f"{name}.layers.{i}.{part}.bias", (g * f, (g + 1) * f))]
            pairs += [(stage.conv_out.weight, f"{name}.conv_out.weight", (g * c, (g + 1) * c)),
                      (stage.conv_out.bias, f"{name}.conv_out.bias", (g * c, (g + 1) * c))]
        return pairs

    @torch.no_grad()
    def load_official(self, g, state):
        for p, name, rows in self._pairs(g):
            (p if rows is None else p[rows[0]:rows[1]]).copy_(state[name])

    @torch.no_grad()
    def official_state(self, g):
        return {name: (p if rows is None else p[rows[0]:rows[1]]).detach().clone()
                for p, name, rows in self._pairs(g)}


def group_losses(predictions, target, mask, num_classes):
    """The official loss of each model, as a (groups,) tensor. predictions: (stages, 1, G * C, T)."""
    s, b, _, t = predictions.shape
    p = predictions.view(s, b, -1, num_classes, t)  # (stages, batch, groups, classes, frames)
    g = p.shape[2]
    logits = p.permute(0, 1, 2, 4, 3).reshape(s, b, g, t, num_classes)
    tgt = target.view(1, b, 1, t).expand(s, b, g, t)
    ce = F.cross_entropy(logits.reshape(-1, num_classes), tgt.reshape(-1), ignore_index=-100, reduction="none")
    ce = ce.view(s, b, g, t)
    valid = (tgt != -100).float()
    ce = (ce * valid).sum(dim=(1, 3)) / valid.sum(dim=(1, 3))  # (stages, groups): mean over frames
    mse = (F.log_softmax(p[..., 1:], dim=3) - F.log_softmax(p.detach()[..., :-1], dim=3)) ** 2
    mse = torch.clamp(mse, min=0, max=16) * mask[None, :, None, :, 1:]
    mse = mse.mean(dim=(1, 3, 4))  # (stages, groups): the official torch.mean over (batch, classes, frames)
    return (ce + 0.15 * mse).sum(dim=0)


class FeatureSource:
    """The inputs of each model for a video. A setting is (folder, k) or (folder, k, center): the
    features stored in folder (dim x frames, read as float32), with center minus the video's own mean
    feature, and with k > 0 next to their difference from k frames earlier (the first k frames take
    their difference from frame 0), as train_mstcn.make_loader builds them. Each folder is read once
    per video and the inputs are built on the card; the values match the CPU path's (checked in
    tests/test_grouped.py)."""

    def __init__(self, settings):
        self.settings = [(Path(s[0]), int(s[1]), bool(s[2]) if len(s) > 2 else False) for s in settings]
        self.folders = list(dict.fromkeys(s[0] for s in self.settings))

    def load(self, vid):
        """The stored arrays of one video, in pinned memory for a fast copy to the card."""
        stem = vid.split(".")[0]
        return {f: torch.from_numpy(np.load(f / (stem + ".npy"))).pin_memory() for f in self.folders}

    def frames(self, raw):
        return min(a.shape[1] for a in raw.values())

    def inputs(self, raw, n, device):
        on_card = {f: a.to(device, non_blocking=True).float() for f, a in raw.items()}
        xs = []
        for folder, k, center in self.settings:
            x = on_card[folder]
            if center:
                x = x - x.mean(dim=1, keepdim=True)
            if k:
                earlier = (torch.arange(x.shape[1], device=device) - k).clamp(min=0)
                x = torch.cat([x, x - x[:, earlier]], dim=0)
            xs.append(x[:, :n].unsqueeze(0))
        return xs


class GroupedBatchGenerator:
    """The official batch generator's order and targets, with every model's features for the video.
    A thread reads the next videos' files while the card trains on the current one."""

    def __init__(self, actions_dict, gt_path, source):
        self.actions_dict, self.gt_path, self.source = actions_dict, gt_path, source
        self.list_of_examples, self.index, self.labels = [], 0, {}
        self.prefetch = None

    def reset(self):
        self.index = 0
        self.prefetch = None
        random.shuffle(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def read_data(self, vid_list):
        self.list_of_examples = list(vid_list)
        random.shuffle(self.list_of_examples)

    def target(self, vid):
        if vid not in self.labels:
            with open(self.gt_path + vid) as fh:
                content = fh.read().split("\n")[:-1]
            self.labels[vid] = np.array([self.actions_dict[c] for c in content], dtype=np.int64)
        return self.labels[vid]

    def _reader(self, order, q):
        try:
            for vid in order:
                q.put((vid, self.source.load(vid), self.target(vid)))
        except Exception as e:  # hand the error to the training loop instead of leaving it waiting
            q.put(e)

    def next_batch(self, device):
        if self.prefetch is None:  # first batch of the epoch: start reading this epoch's order
            self.prefetch = queue.Queue(maxsize=3)
            threading.Thread(target=self._reader, args=(list(self.list_of_examples), self.prefetch),
                             daemon=True).start()
        item = self.prefetch.get()
        if isinstance(item, Exception):
            raise item
        vid, raw, labels = item
        assert vid == self.list_of_examples[self.index]
        self.index += 1
        n = min(self.source.frames(raw), len(labels))
        xs = self.source.inputs(raw, n, device)
        target = torch.from_numpy(labels[:n]).unsqueeze(0).to(device)
        mask = torch.ones(1, len(self.actions_dict), n, device=device)
        return xs, target, mask


class GroupedTrainer:
    def __init__(self, dims, num_classes, seed, num_stages=4, num_layers=10, num_f_maps=64):
        """Model g starts from the weights the official Trainer draws after torch.manual_seed(seed)."""
        self.num_classes = num_classes
        self.model = GroupedModel(num_stages, num_layers, num_f_maps, dims, num_classes)
        for g, d in enumerate(dims):
            torch.manual_seed(seed)
            self.model.load_official(g, MultiStageModel(num_stages, num_layers, num_f_maps, d, num_classes).state_dict())
        torch.manual_seed(seed)

    def train(self, batch_gen, num_epochs, learning_rate, device, logs):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        g = self.model.groups
        for epoch in range(num_epochs):
            epoch_loss = torch.zeros(g, device=device)
            correct = torch.zeros(g, device=device)
            total = torch.zeros((), device=device)
            while batch_gen.has_next():
                xs, target, mask = batch_gen.next_batch(device)
                optimizer.zero_grad()
                predictions = self.model(xs, mask)
                losses = group_losses(predictions, target, mask, self.num_classes)
                losses.sum().backward()
                optimizer.step()
                epoch_loss += losses.detach()
                t = target.shape[1]
                predicted = predictions[-1].detach().view(1, g, self.num_classes, t).argmax(dim=2)
                correct += ((predicted == target[:, None, :]).float() * mask[:, 0:1, :]).sum(dim=(0, 2))
                total += mask[:, 0, :].sum()
            batch_gen.reset()
            n = len(batch_gen.list_of_examples)
            el, acc = (epoch_loss / n).tolist(), (correct / total).tolist()
            for k in range(g):
                logs[k]("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, el[k], acc[k]))

    @torch.no_grad()
    def predict(self, source, vid_list, actions_dict, device):
        """Frame-level label lists per video for each model, as the official predict step makes them."""
        index_to_action = {v: k for k, v in actions_dict.items()}
        self.model.eval()
        self.model.to(device)
        g = self.model.groups
        out = [{} for _ in range(g)]
        for vid in vid_list:
            raw = source.load(vid)
            t = source.frames(raw)
            xs = source.inputs(raw, t, device)
            predictions = self.model(xs, torch.ones(1, self.num_classes, t, device=device))
            predicted = predictions[-1].view(1, g, self.num_classes, t).argmax(dim=2)[0].tolist()
            for k in range(g):
                out[k][vid] = [index_to_action[p] for p in predicted[k]]
        return out
