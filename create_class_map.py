import json
import argparse

import pandas as pd


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--train_df", required=True, type=str,
    help="path to train dataframe"
)
parser.add_argument(
    "--output_file", type=str, required=True,
    help="where to save classmap"
)

args = parser.parse_args()


df = pd.read_csv(args.train_df)

all_labels = set()
for item in df.labels:
    all_labels.update(item.split(","))


classmap = dict((v, k) for k, v in enumerate(sorted(all_labels)))

with open(args.output_file, "w") as file:
    json.dump(classmap, file, indent=4, sort_keys=True)