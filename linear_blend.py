import argparse
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.optimize
from scipy.stats import rankdata
from mag.utils import blue, green, bold

from ops.utils import lwlrap


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--experiments", type=str, required=True, nargs="+",
    help="experiments to blend"
)
parser.add_argument(
    "--train_df", type=str, required=True,
    help="path to train df"
)
parser.add_argument(
    "--rankdata", action="store_true", default=False,
    help="whether to use ranks instead of raw scores"
)
parser.add_argument(
    "--output_df", type=str, required=True,
    help="where to save test submission"
)

args = parser.parse_args()


n = len(args.experiments)


def load_predictions(experiment):

    prediction_files = (
        "experiments" / Path(experiment) / "predictions").glob("val_preds*")
    dfs = [pd.read_csv(f) for f in prediction_files]
    df = pd.concat(dfs).reset_index(drop=True)
    df = df.sort_values(by="fname")
    df = df[sorted(df.columns.tolist())]
    return df


def to_ranks(values):
    return np.array([rankdata(r) for r in values])


predictions = [load_predictions(exp) for exp in args.experiments]
class_cols = predictions[0].columns.drop("fname")
prediction_values = [p[class_cols].values for p in predictions]
if args.rankdata:
    prediction_values = [to_ranks(p) for p in prediction_values]

train_df = pd.read_csv(args.train_df)


def make_actual_labels(train_df):

    classname_to_idx = dict((c, i) for i, c in enumerate(class_cols))
    actual_labels = np.zeros((len(train_df), len(class_cols)), dtype=np.float32)
    for k in range(train_df.labels.values.size):
        for label in str(train_df.labels.values[k]).split(","):
            actual_labels[k, classname_to_idx[label]] = 1

    return actual_labels


actual_labels = make_actual_labels(train_df)


def constraints():
    A = np.ones(n)
    yield scipy.optimize.LinearConstraint(A=A, lb=0.01, ub=0.99)
    for k in range(n):
        A = np.zeros(n)
        A[k] = 1
        yield scipy.optimize.LinearConstraint(A=A, lb=0, ub=1)


def initial():
    return np.ones(n) / n

def target(alphas, *args):
    prediction = np.sum([a * p for a, p in zip(alphas, prediction_values)], axis=0)
    return -lwlrap(actual_labels, prediction)


alphas = scipy.optimize.minimize(
    target,
    initial(),
    constraints=list(constraints()),
    method="COBYLA").x

print()
for experiment, alpha in zip(args.experiments, alphas):
    print("{}: {}".format(green(bold(experiment)), blue(bold(alpha))))

print()
print("Final lwlrap:", bold(green(-target(alphas))))


def load_test_predictions(experiment):

    prediction_files = (
        "experiments" / Path(experiment) / "predictions").glob("test_preds*")
    dfs = [pd.read_csv(f) for f in prediction_files]
    dfs = [df.sort_values(by="fname") for df in dfs]
    return dfs


test_preds = []

for alpha, exp in zip(alphas, args.experiments):
    experiment_test_predictions = load_test_predictions(experiment)
    for p in experiment_test_predictions:
        if args.rankdata:
            test_preds.append(to_ranks(p[class_cols].values) * alpha)
        else:
            test_preds.append(p[class_cols].values * alpha)

test_preds = np.sum(test_preds, 0)

sub = pd.DataFrame(test_preds, columns=class_cols)
sub["fname"] = p.fname

sub.to_csv(args.output_df, index=False)