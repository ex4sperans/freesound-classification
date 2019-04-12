import random
from copy import deepcopy

import numpy as np
from torch.utils.data.dataloader import default_collate


def make_collate_fn(padding_values):

    def _collate_fn(batch):

        for name, padding_value in padding_values.items():

            lengths = [len(sample[name]) for sample in batch]
            max_length = max(lengths)

            for n, size in enumerate(lengths):
                p = max_length - size
                if p:
                    pad_width = [(0, p)] + [(0, 0)] * (batch[n][name].ndim - 1)
                    if padding_value == "edge":
                        batch[n][name] = np.pad(
                            batch[n][name], pad_width,
                            mode="edge")
                    else:
                        batch[n][name] = np.pad(
                            batch[n][name], pad_width,
                            mode="constant", constant_values=padding_value)

        return default_collate(batch)

    return _collate_fn



class BucketingSampler:

    def __init__(self, dataset, max_batch_elems, buckets):

        self.buckets = buckets
        self.dataset = dataset
        self.max_batch_elems = max_batch_elems

        self._create_batches()

    def _create_batches(self):

        self.n_bins = len(self.buckets)
        binned_sizes = np.digitize(self.dataset.lengths, self.buckets)

        batches = []

        for bin_idx in range(1, self.n_bins):
            ids = np.nonzero(binned_sizes == bin_idx)[0]
            random.shuffle(ids)

            current_len = 0
            batch = []

            for id in ids:
                if current_len < self.max_batch_elems:
                    batch.append(id)
                    current_len += self.dataset.lengths[id]
                else:
                    batches.append(batch)
                    current_len = self.dataset.lengths[id]
                    batch = [id]

            if batch:
                batches.append(batch)

        random.shuffle(batches)

        self.n_batches = len(batches)
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return self.n_batches