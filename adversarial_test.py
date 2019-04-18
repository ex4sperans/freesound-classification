import os
import gc
import argparse
import json
import math
from functools import partial

import tqdm
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from datasets.sound_dataset import SoundDataset
from networks.classifiers import HierarchicalCNNClassificationModel
from ops.folds import train_validation_data
from ops.transforms import (
    Compose, DropFields, LoadAudio,
    AudioFeatures, MapLabels, RenameFields,
    MixUp, SampleSegment, SampleLongAudio)
from ops.utils import load_json, get_class_names_from_classmap, lwlrap
from ops.padding import make_collate_fn
from networks.classifiers import ResnetBlock

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--train_df", required=True, type=str,
    help="path to train dataframe"
)
parser.add_argument(
    "--train_data_dir", required=True, type=str,
    help="path to train data"
)
parser.add_argument(
    "--test_data_dir", required=True, type=str,
    help="path to test data"
)
parser.add_argument(
    "--test_df", required=True, type=str,
    help="path to train dataframe"
)
parser.add_argument(
    "--val_size", required=True, type=float,
    help="size of the validation set"
)
parser.add_argument(
    "--device", type=str, required=True,
    help="whether to train on cuda or cpu",
    choices=("cuda", "cpu")
)
parser.add_argument(
    "--batch_size", type=int, default=64,
    help="minibatch size"
)
parser.add_argument(
    "--epochs", type=int, default=100,
    help="number of epochs"
)
parser.add_argument(
    "--lr", default=0.01, type=float,
    help="starting learning rate"
)
parser.add_argument(
    "--max_samples", type=int,
    help="maximum number of samples to use"
)

args = parser.parse_args()

train_df = pd.read_csv(args.train_df)
test_df = pd.read_csv(args.test_df)

if args.max_samples:
    train_df = train_df.sample(args.max_samples).reset_index(drop=True)
    test_df = test_df.sample(args.max_samples).reset_index(drop=True)

fnames = np.concatenate([
    [os.path.join(args.train_data_dir, fname) for fname in train_df.fname.values],
    [os.path.join(args.test_data_dir, fname) for fname in test_df.fname.values]
])
labels = np.concatenate([np.ones(len(train_df)), np.zeros(len(test_df))])

train_fnames, val_fnames, train_labels, val_labels = train_test_split(
    fnames, labels, test_size=args.val_size, shuffle=True)

audio_transform = AudioFeatures("stft_256_64")


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(audio_transform.n_features),
            torch.nn.Conv1d(audio_transform.n_features, 32, kernel_size=1),
            ResnetBlock(32),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(32, 64, kernel_size=1),
            ResnetBlock(64),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(64, 128, kernel_size=1),
            ResnetBlock(128)
        )

        self.pool = torch.nn.AdaptiveMaxPool1d(1)

        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(128),
            torch.nn.Linear(128, 1)
        )

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)

        return torch.sigmoid(x)


train_loader = torch.utils.data.DataLoader(
    SoundDataset(
        audio_files=train_fnames,
        labels=train_labels,
        transform=Compose([
            LoadAudio(),
            audio_transform,
            RenameFields({"raw_labels": "labels"}),
            DropFields(("audio", "filename", "sr")),
        ]),
        clean_transform=Compose([
            LoadAudio(),
        ])
    ),
    shuffle=True,
    drop_last=True,
    batch_size=args.batch_size,
    collate_fn=make_collate_fn({"signal": audio_transform.padding_value}),
)

validation_loader = torch.utils.data.DataLoader(
    SoundDataset(
        audio_files=val_fnames,
        labels=val_labels,
        transform=Compose([
            LoadAudio(),
            audio_transform,
            RenameFields({"raw_labels": "labels"}),
            DropFields(("audio", "filename", "sr")),
        ]),
        clean_transform=Compose([
            LoadAudio(),
        ])
    ),
    shuffle=False,
    drop_last=False,
    batch_size=args.batch_size,
    collate_fn=make_collate_fn({"signal": audio_transform.padding_value}),
)

model = Model().to(args.device)
optimizer = torch.optim.Adam(model.parameters(), args.lr)


for epoch in range(args.epochs):

    print(
        "\n" + " " * 10 + "****** Epoch {epoch} ******\n"
        .format(epoch=epoch)
    )

    with tqdm.tqdm(total=len(train_loader), ncols=80) as pb:

        for sample in train_loader:

            signal, labels = (
                sample["signal"].to(args.device),
                sample["labels"].to(args.device).float()
            )

            probs = model(signal).squeeze(-1)

            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy(probs, labels)
            loss.backward()
            optimizer.step()

            pb.update()
            pb.set_description("Loss: {:.4f}".format(loss.item()))

    val_probs = []
    val_labels = []

    for sample in validation_loader:

        signal, labels = (
            sample["signal"].to(args.device),
            sample["labels"].to(args.device).float()
        )

        probs = model(signal).squeeze(-1)

        val_probs.extend(probs.data.cpu().numpy())
        val_labels.extend(labels.data.cpu().numpy())

    auc = roc_auc_score(val_labels, val_probs)

    print("\nEpoch: {}, AUC: {}".format(epoch, auc))
