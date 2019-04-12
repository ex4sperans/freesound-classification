from sklearn.model_selection import KFold
import numpy as np


def train_validation_data(ids, labels, n_folds, seed):

    for train, valid in KFold(
        n_folds, shuffle=True, random_state=seed).split(ids, labels):
        yield train, valid