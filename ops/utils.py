import json

import numpy as np
from sklearn.metrics import label_ranking_average_precision_score

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