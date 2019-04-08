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

    def __init__(self, audio_files, labels=None, transform=None):

        self.transform = transform
        self.audio_files = audio_files
        self.labels = labels

    def __getitem__(self, index):

        sample = dict(filename=self.audio_files[index])

        if self.labels is not None:
            sample["raw_labels"] = self.labels[index]

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def __len__(self):
        return len(self.audio_files)
