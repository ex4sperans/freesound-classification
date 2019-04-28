import json

import umap
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
from matplotlib import pyplot as plt


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


def plot_umap(vectors, labels, frames_per_example=3):

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

    embeddings = umap.UMAP().fit_transform(representations)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.scatter(embeddings[:, 0], embeddings[:, 1], c=classes, s=10)

    fig.canvas.draw()

    image = np.array(fig.canvas.renderer._renderer)

    plt.close()

    return image
