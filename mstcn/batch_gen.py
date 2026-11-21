# Batch generator ported to Python 3 from the official MS-TCN code (github.com/yabufarha/ms-tcn,
# commit 33ed91c, MIT license with the Commons Clause, see LICENSE-ms-tcn). Same shuffling, padding and
# masking. Change: features come from a load_features(video) callable, so other feature sets fit in.
import random

import numpy as np
import torch


class BatchGenerator:
    def __init__(self, num_classes, actions_dict, gt_path, load_features, sample_rate):
        self.list_of_examples = []
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.load_features = load_features
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        return self.index < len(self.list_of_examples)

    def read_data(self, vid_list):
        self.list_of_examples = list(vid_list)
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            features = self.load_features(vid)
            with open(self.gt_path + vid, "r") as fh:
                content = fh.read().split("\n")[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences),
                                         dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
