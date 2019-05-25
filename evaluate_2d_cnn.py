import os
import gc
import argparse
import json
import math
from functools import partial

import pandas as pd
import numpy as np
import torch
from mag.experiment import Experiment
from mag.utils import green, bold
import mag

from datasets.sound_dataset import SoundDataset
from networks.classifiers import TwoDimensionalCNNClassificationModel
from ops.folds import train_validation_data_stratified
from ops.transforms import (
    Compose, DropFields, LoadAudio,
    AudioFeatures, MapLabels, RenameFields,
    MixUp, SampleSegment, SampleLongAudio,
    AudioAugmentation, FlipAudio, ShuffleAudio)
from ops.utils import load_json, get_class_names_from_classmap, lwlrap
from ops.padding import make_collate_fn

mag.use_custom_separator("-")

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--experiment", type=str, required=True,
    help="path to an experiment"
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
    "--noisy_train_df", type=str,
    help="path to noisy train dataframe (optional)"
)
parser.add_argument(
    "--noisy_train_data_dir", type=str,
    help="path to noisy train data (optional)"
)
parser.add_argument(
    "--classmap", required=True, type=str,
    help="path to class map json"
)
parser.add_argument(
    "--batch_size", type=int, default=32,
    help="batch size used for prediction"
)
parser.add_argument(
    "--max_audio_length", type=int, default=10,
    help="max audio length in seconds. For longer clips are sampled"
)
parser.add_argument(
    "--n_tta", type=int, default=1,
    help="number of tta"
)
parser.add_argument(
    "--device", type=str, required=True,
    help="whether to train on cuda or cpu",
    choices=("cuda", "cpu")
)
parser.add_argument(
    "--num_conv_blocks", type=int, default=5,
    help="number of conv blocks"
)
parser.add_argument(
    "--num_workers", type=int, default=4,
    help="number of workers for data loader",
)

args = parser.parse_args()

class_map = load_json(args.classmap)

train_df = pd.read_csv(args.train_df)

with Experiment(resume_from=args.experiment) as experiment:

    config = experiment.config

    audio_transform = AudioFeatures(config.data.features)

    splits = list(train_validation_data_stratified(
            train_df.fname, train_df.labels, class_map,
            config.data._n_folds, config.data._kfold_seed))

    all_labels = np.zeros(
        shape=(len(train_df), len(class_map)), dtype=np.float32)
    all_predictions = np.zeros(
        shape=(len(train_df), len(class_map)), dtype=np.float32)

    for fold in range(config.data._n_folds):

        print("\n\n   -----  Fold {}\n".format(fold))

        train, valid = splits[fold]

        loader_kwargs = (
            {"num_workers": args.num_workers, "pin_memory": True}
            if torch.cuda.is_available() else {})

        valid_loader = torch.utils.data.DataLoader(
            SoundDataset(
                audio_files=[
                    os.path.join(args.train_data_dir, fname)
                    for fname in train_df.fname.values[valid]],
                labels=[item.split(",") for item in train_df.labels.values[valid]],
                transform=Compose([
                    LoadAudio(),
                    MapLabels(class_map=class_map),
                    SampleLongAudio(args.max_audio_length),
                    ShuffleAudio(chunks_range=(12, 20), p=1.0),
                    audio_transform,
                    DropFields(("audio", "filename", "sr")),
                ]),
                clean_transform=Compose([
                    LoadAudio(),
                    MapLabels(class_map=class_map),
                ]),


            ),
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=make_collate_fn({"signal": audio_transform.padding_value}),
            **loader_kwargs
        )

        model = TwoDimensionalCNNClassificationModel(
                experiment, device=args.device)
        model.load_best_model(fold)
        model.eval()

        val_preds = model.predict(valid_loader, n_tta=args.n_tta)
        val_labels = np.array([item["labels"] for item in valid_loader.dataset])

        all_labels[valid] = val_labels
        all_predictions[valid] = val_preds

        metric = lwlrap(val_labels, val_preds)
        print("Fold metric:", metric)

    metric = lwlrap(all_labels, all_predictions)

    print("\nOverall metric:", green(bold(metric)))


