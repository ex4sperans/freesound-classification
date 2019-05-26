import os
import gc
import argparse
import json
import math
from functools import partial

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--noisy_df", required=True, type=str,
    help="path to noisy dataframe"
)
parser.add_argument(
    "--noisy_predictions_df", required=True, type=str,
    help="path to noisy predictions"
)
parser.add_argument(
    "--output_df", required=True, type=str,
    help="where to save relabeled dataframe"
)
parser.add_argument(
    "--mode", required=True, type=str,
    help="relabeling strategy"
)

args = parser.parse_args()

noisy_df = pd.read_csv(args.noisy_df)
noisy_predictions_df = pd.read_csv(args.noisy_predictions_df)

noisy_df.sort_values(by="fname", inplace=True)
noisy_predictions_df.sort_values(by="fname", inplace=True)

mode, *params = args.mode.split("_")

class_cols = noisy_predictions_df.columns.drop("fname").values
classname_to_idx = dict((c, i) for i, c in enumerate(class_cols))
noisy_labels = np.zeros((len(noisy_df), len(class_cols)), dtype=np.float32)
for k in range(noisy_df.labels.values.size):
    for label in str(noisy_df.labels.values[k]).split(","):
        noisy_labels[k, classname_to_idx[label]] = 1

if mode == "fullmatch":

    def find_threshold(probs, expected_classes_per_sample):

        thresholds = np.linspace(0, 1, 10000)
        classes_per_sample = np.zeros_like(thresholds)

        for k in range(thresholds.size):
            c = (probs > thresholds[k]).sum(-1).mean()
            classes_per_sample[k] = c

        k = np.argmin(np.abs(classes_per_sample - expected_classes_per_sample))

        return thresholds[k]

    def nonzero(x):
        return np.nonzero(x)[0]

    expected_classes_per_sample, = params
    expected_classes_per_sample = float(expected_classes_per_sample)

    probs = noisy_predictions_df[class_cols].values
    threshold = find_threshold(probs, expected_classes_per_sample)
    print(threshold)
    binary = probs > threshold

    match = (binary == noisy_labels).all(-1)

    relabeled = noisy_df[match]

print("Relabeled df shape:", relabeled.shape)

relabeled.to_csv(args.output_df, index=False)
