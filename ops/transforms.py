import random
import math
from functools import partial
import json

import pysndfx
import librosa
import numpy as np
import torch

from ops.audio import (
    read_audio, compute_stft, trim_audio, mix_audio_and_labels,
    shuffle_audio, cutout
)


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


class FlipAudio(Augmentation):

    def __init__(self, p):

        self.p = p

    def __call__(self, dataset, **inputs):

        transformed = dict(inputs)

        if np.random.uniform() < self.p:
            transformed["audio"] = np.flipud(inputs["audio"])

        return transformed


class AudioAugmentation(Augmentation):

    def __init__(self, p):

        self.p = p

    def __call__(self, dataset, **inputs):

        transformed = dict(inputs)

        if np.random.uniform() < self.p:
            effects_chain = (
                pysndfx.AudioEffectsChain()
                .reverb(
                    reverberance=random.randrange(50),
                    room_scale=random.randrange(50),
                    stereo_depth=random.randrange(50)
                )
                .pitch(shift=random.randrange(-300, 300))
                .overdrive(gain=random.randrange(2, 10))
                .speed(random.uniform(0.9, 1.1))
            )
            transformed["audio"] = effects_chain(inputs["audio"])

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
            self.padding_value = 0.0

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
            self.padding_value = 0.0

            if verbose:
                print(
                    "\nUsing mel features with params:\n",
                    "n_fft: {}, hop_size: {}, n_mel: {}".format(
                        n_fft, hop_size, n_mel
                    )
                )

        elif name == "raw":

            self.n_features = 1
            self.padding_value = 0.0

            if verbose:
                print(
                    "\nUsing raw waveform features."
                )


    def __call__(self, dataset, **inputs):

        transformed = dict(inputs)

        if self.feature_type == "stft":

            # stft = compute_stft(
            #     inputs["audio"],
            #     window_size=self.n_fft, hop_size=self.hop_size,
            #     eps=self.eps, log=True
            # )

            transformed["signal"] = np.expand_dims(inputs["audio"], -1)

        elif self.feature_type == "mel":

            stft = compute_stft(
                inputs["audio"],
                window_size=self.n_fft, hop_size=self.hop_size,
                eps=self.eps, log=False
            )

            transformed["signal"] = np.expand_dims(inputs["audio"], -1)

        elif self.feature_type == "raw":
            transformed["signal"] = np.expand_dims(inputs["audio"], -1)

        return transformed


class SampleSegment(Augmentation):

    def __init__(self, ratio=(0.3, 0.9), p=1.0):

        self.min, self.max = ratio
        self.p = p

    def __call__(self, dataset, **inputs):

        transformed = dict(inputs)

        if np.random.uniform() < self.p:
            original_size = inputs["audio"].size
            target_size = int(np.random.uniform(self.min, self.max) * original_size)
            start = np.random.randint(original_size - target_size - 1)
            transformed["audio"] = inputs["audio"][start:start+target_size]

        return transformed


class ShuffleAudio(Augmentation):

    def __init__(self, chunk_length=0.5, p=0.5):

        self.chunk_length = chunk_length
        self.p = p

    def __call__(self, dataset, **inputs):

        transformed = dict(inputs)

        if np.random.uniform() < self.p:
            transformed["audio"] = shuffle_audio(
                transformed["audio"], self.chunk_length, sr=transformed["sr"])

        return transformed


class CutOut(Augmentation):

    def __init__(self, area=0.25, p=0.5):

        self.area = area
        self.p = p

    def __call__(self, dataset, **inputs):

        transformed = dict(inputs)

        if np.random.uniform() < self.p:
            transformed["audio"] = cutout(
                transformed["audio"], self.area)

        return transformed


class SampleLongAudio:

    def __init__(self, max_length):

        self.max_length = max_length

    def __call__(self, dataset, **inputs):

        transformed = dict(inputs)

        if (inputs["audio"].size / inputs["sr"]) > self.max_length:

            max_length = self.max_length * inputs["sr"]

            start = np.random.randint(0, inputs["audio"].size - max_length)
            transformed["audio"] = inputs["audio"][start:start+max_length]

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


class Identity:

    def __call__(self, dataset=None, **inputs):

        return inputs