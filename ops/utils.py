import json

import umap
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import label_ranking_average_precision_score, accuracy_score
from matplotlib import pyplot as plt
import librosa


# Calculate the overall lwlrap using sklearn.metrics function.

def lwlrap(truth, scores):
  """Calculate the overall lwlrap using sklearn.metrics.lrap."""
  # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
  sample_weight = np.sum(truth > 0, axis=1)
  nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
  overall_lwlrap = label_ranking_average_precision_score(
      truth[nonzero_weight_sample_indices, :] > 0,
      scores[nonzero_weight_sample_indices, :],
      sample_weight=sample_weight[nonzero_weight_sample_indices])
  return overall_lwlrap


def load_json(file):
    with open(file, "r") as f:
        return json.load(f)


def get_class_names_from_classmap(classmap):
    r = dict((v, k) for k, v in classmap.items())
    return [r[label] for label in sorted(classmap.values())]


def plot_projection(vectors, labels, frames_per_example=3, newline=False):

    representations = []
    classes = []
    for sample, label in zip(vectors, labels):
        if sum(label) > 1:
            continue
        choices = np.random.choice(
            np.arange(len(sample)), replace=False,
            size=min(frames_per_example, len(sample)))
        representations.extend(sample[choices])
        classes.extend([label.tolist().index(1)] * len(choices))

    representations = np.array(representations)

    # fit a simple model to estimate the quality of the learned representations
    X_train, X_valid, y_train, y_valid = train_test_split(
        representations, classes, shuffle=False, test_size=0.2)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    score = accuracy_score(y_valid, model.predict(X_valid))
    if newline:
        print()
    print("Classification accuracy: {:.4f}".format(score))

    # plot projection
    embeddings = TSNE().fit_transform(representations)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=classes, s=10)

    fig.canvas.draw()

    image = np.array(fig.canvas.renderer._renderer)

    plt.close()

    return image


def make_mel_filterbanks(descriptor, sr=44100):

    name, *args = descriptor.split("_")

    n_fft, hop_size, n_mel = args
    n_fft = int(n_fft)
    hop_size = int(hop_size)
    n_mel = int(n_mel)

    filterbank = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mel,
        fmin=5, fmax=None
    ).astype(np.float32)

    return filterbank


def is_mel(descriptor):
    return descriptor.startswith("mel")