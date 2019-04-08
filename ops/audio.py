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
