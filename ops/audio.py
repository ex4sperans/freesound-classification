import random

import numpy as np
import librosa
import scipy.signal

from sklearn.utils import gen_even_slices


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


def shuffle_audio(audio, chunk_length=0.5, sr=None):

    n_chunks = int((audio.size / sr) / chunk_length)

    if n_chunks in (0, 1):
        return audio

    slices = list(gen_even_slices(audio.size, n_chunks))
    random.shuffle(slices)

    shuffled = np.concatenate([audio[s] for s in slices])

    return shuffled


def cutout(audio, area=0.25):

    area = int(audio.size * area)

    start = random.randrange(audio.size)
    end = start + area

    audio[start:end] = 0

    return audio
