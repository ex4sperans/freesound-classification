import os
import glob
import pickle
import random
import json

import torch
from tqdm import tqdm
import torch.utils.data as data
import numpy as np
import pandas as pd


class SoundDataset(data.Dataset):

    def __init__(
        self, audio_files, labels=None,
        transform=None, is_noisy=None, clean_transform=None):

        self.transform = transform
        self.clean_transform = clean_transform
        self.audio_files = audio_files
        self.labels = labels
        self.is_noisy = is_noisy or np.zeros(len(self.audio_files))

    def __getitem__(self, index):

        sample = dict(
            filename=self.audio_files[index],
            is_noisy=self.is_noisy[index]
        )

        if self.labels is not None:
            sample["raw_labels"] = self.labels[index]

        if self.transform is not None:
            sample = self.transform(dataset=self, **sample)

        return sample

    def random_clean_sample(self):

        index = random.randint(0, len(self) - 1)

        sample = dict(
            filename=self.audio_files[index],
            is_noisy=self.is_noisy[index]
        )

        if self.labels is not None:
            sample["raw_labels"] = self.labels[index]

        if self.clean_transform is not None:
            sample = self.clean_transform(dataset=self, **sample)

        return sample

    def __len__(self):
        return len(self.audio_files)
