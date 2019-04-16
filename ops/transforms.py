import random
import math
from functools import partial
import json

import librosa
import numpy as np
import torch

from ops.audio import read_audio, compute_stft, trim_audio, mix_audio_and_labels


SAMPLE_RATE = 44100


class Augmentation:
    """A base class for data augmentation transforms"""
    pass


class MapLabels:

    def __init__(self, class_map, drop_raw=True):

        self.class_map = class_map

    def __call__(self, dataset, **inputs):

        labels = np.zeros(len(self.class_map), dtype=np.float32)
        for c in inputs["raw_labels"]:
            labels[self.class_map[c]] = 1.0

        transformed = dict(inputs)
        transformed["labels"] = labels
        transformed.pop("raw_labels")

        return transformed


class MixUp(Augmentation):

    def __init__(self, p):

        self.p = p

    def __call__(self, dataset, **inputs):

        transformed = dict(inputs)

        if np.random.uniform() < self.p:
            first_audio, first_labels = inputs["audio"], inputs["labels"]
            random_sample = dataset.random_clean_sample()
            new_audio, new_labels = mix_audio_and_labels(
                first_audio, random_sample["audio"],
                first_labels, random_sample["labels"]
            )

            transformed["audio"] = new_audio
            transformed["labels"] = new_labels

        return transformed


class LoadAudio:

    def __init__(self):

        pass

    def __call__(self, dataset, **inputs):

        audio, sr = read_audio(inputs["filename"])

        transformed = dict(inputs)
        transformed["audio"] = audio
        transformed["sr"] = sr

        return transformed


class STFT:

    eps = 1e-4

    def __init__(self, n_fft, hop_size):

        self.n_fft = n_fft
        self.hop_size = hop_size

    def __call__(self, dataset, **inputs):

        stft = compute_stft(
            inputs["audio"],
            window_size=self.n_fft, hop_size=self.hop_size,
            eps=self.eps)

        transformed = dict(inputs)
        transformed["stft"] = np.transpose(stft)

        return transformed


class AudioFeatures:

    eps = 1e-4

    def __init__(self, descriptor, verbose=True):

        name, *args = descriptor.split("_")

        self.feature_type = name

        if name == "stft":

            n_fft, hop_size = args
            self.n_fft = int(n_fft)
            self.hop_size = int(hop_size)

            self.n_features = self.n_fft // 2 + 1
            self.padding_value = math.log(self.eps)

            if verbose:
                print(
                    "\nUsing STFT features with params:\n",
                    "n_fft: {}, hop_size: {}".format(
                        n_fft, hop_size
                    )
                )

        elif name == "mel":

            n_fft, hop_size, n_mel = args
            self.n_fft = int(n_fft)
            self.hop_size = int(hop_size)
            self.n_mel = int(n_mel)

            self.n_features = self.n_mel
            self.padding_value = math.log(self.eps)

            self.filterbank = librosa.filters.mel(
                sr=SAMPLE_RATE, n_fft=self.n_fft, n_mels=self.n_mel,
                fmin=5, fmax=None
            ).astype(np.float32)

            if verbose:
                print(
                    "\nUsing mel features with params:\n",
                    "n_fft: {}, hop_size: {}, n_mel: {}".format(
                        n_fft, hop_size, n_mel
                    )
                )

    def __call__(self, dataset, **inputs):

        transformed = dict(inputs)

        if self.feature_type == "stft":

            stft = compute_stft(
                inputs["audio"],
                window_size=self.n_fft, hop_size=self.hop_size,
                eps=self.eps, log=True
            )

            transformed["signal"] = np.transpose(stft)

        elif self.feature_type == "mel":

            stft = compute_stft(
                inputs["audio"],
                window_size=self.n_fft, hop_size=self.hop_size,
                eps=self.eps, log=False
            )

            mel = self.filterbank.dot(stft)
            mel = np.log(mel + self.eps)

            transformed["signal"] = np.transpose(mel)

        return transformed


class SampleSegment(Augmentation):

    def __init__(self, ratio=(0.3, 0.9), p=1.0):

        self.min, self.max = ratio
        self.p = 1.0

    def __call__(self, dataset, **inputs):

        transformed = dict(inputs)

        if np.random.uniform() < self.p:
            original_size = inputs["audio"].size
            target_size = int(np.random.uniform(self.min, self.max) * original_size)
            start = np.random.randint(original_size - target_size - 1)
            transformed["audio"] = inputs["audio"][start:start+target_size]

        return transformed


class OneOf:

    def __init__(self, transforms):

        self.transforms = transforms

    def __call__(self, dataset, **inputs):

        transform = random.choice(self.transforms)
        return transform(**inputs)


class DropFields:

    def __init__(self, fields):

        self.to_drop = fields

    def __call__(self, dataset, **inputs):

        transformed = dict()

        for name, input in inputs.items():
            if not name in self.to_drop:
                transformed[name] = input

        return transformed


class RenameFields:

    def __init__(self, mapping):

        self.mapping = mapping

    def __call__(self, dataset, **inputs):

        transformed = dict(inputs)

        for old, new in self.mapping.items():
            transformed[new] = transformed.pop(old)

        return transformed


class Compose:

    def __init__(self, transforms):
        self.transforms = transforms

    def switch_off_augmentations(self):
        for t in self.transforms:
            if isinstance(t, Augmentation):
                t.p = 0.0

    def __call__(self, dataset=None, **inputs):
        for t in self.transforms:
            inputs = t(dataset=dataset, **inputs)

        return inputs