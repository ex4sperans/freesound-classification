from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np


def train_validation_data(ids, labels, n_folds, seed):

    for train, valid in KFold(
        n_folds, shuffle=True, random_state=seed).split(ids, labels):
        yield train, valid


def train_validation_data_stratified(
    ids, labels, classmap, n_folds, seed):

    binary_labels = np.zeros(
        (len(labels), len(classmap)), dtype=np.float32)
    for k, item in enumerate(labels.values):
        for label in item.split(","):
            binary_labels[k, classmap[label]] = 1

    for train, valid in MultilabelStratifiedKFold(
        n_folds, shuffle=True, random_state=seed).split(ids, binary_labels):
        yield train, valid