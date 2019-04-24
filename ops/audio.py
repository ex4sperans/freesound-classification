import random

import numpy as np
import librosa
import scipy.signal


def compute_stft(audio, window_size, hop_size, log=True, eps=1e-4):
    f, t, s = scipy.signal.stft(
        audio, nperseg=window_size, noverlap=hop_size)

    s = np.abs(s)

    if log:
        s = np.log(s + eps)

    return s


def trim_audio(audio):
    audio, interval = librosa.effects.trim(audio, top_db=60)
    return audio


def read_audio(file):
    audio, sr = librosa.load(file, sr=None)
    return audio, sr


def mix_audio_and_labels(first_audio, second_audio, first_labels, second_labels):

    new_labels = np.clip(first_labels + second_labels, 0, 1)

    a = np.random.uniform(0.4, 0.6)

    shorter, longer = first_audio, second_audio

    if shorter.size == longer.size:
        return (shorter + longer) / 2, new_labels

    if first_audio.size > second_audio.size:
        shorter, longer = longer, shorter

    start = random.randint(0, longer.size - 1 - shorter.size)
    end = start + shorter.size

    longer *= a
    longer[start:end] =+ shorter * (1 - a)

    return longer, new_labels
