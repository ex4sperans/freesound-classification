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
import mag

from datasets.sound_dataset import SoundDataset
from networks.classifiers import HierarchicalCNNClassificationModel
from ops.folds import train_validation_data
from ops.transforms import (
    Compose, DropFields, LoadAudio, STFT, MapLabels, RenameFields)
from ops.utils import load_json
from ops.padding import make_collate_fn

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

mag.use_custom_separator("-")

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
    "--classmap", required=True, type=str,
    help="path to class map json"
)
parser.add_argument(
    "--log_interval", default=10, type=int,
    help="how frequently to log batch metrics"
    "in terms of processed batches"
)
parser.add_argument(
    "--batch_size", type=int, default=64,
    help="minibatch size"
)
parser.add_argument(
    "--lr", default=0.01, type=float,
    help="starting learning rate"
)
parser.add_argument(
    "--max_samples", type=int,
    help="maximum number of samples to use"
)
parser.add_argument(
    "--epochs", default=100, type=int,
    help="number of epochs to train"
)
parser.add_argument(
    "--scheduler", type=str, default="steplr_1_0.5",
    help="scheduler type",
)
parser.add_argument(
    "--accumulation_steps", type=int, default=1,
    help="number of gradient accumulation steps",
)
parser.add_argument(
    "--save_every", type=int, default=1,
    help="how frequently to save a model",
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
    "--start_deep_supervision_on", type=int, default=2,
    help="from which layer to start aggregating features for classification"
)
parser.add_argument(
    "--conv_base_depth", type=int, default=64,
    help="base depth for conv layers"
)
parser.add_argument(
    "--growth_rate", type=float, default=2,
    help="how quickly to increase the number of units as a function of layer"
)
parser.add_argument(
    "--weight_decay", type=float, default=1e-5,
    help="weight decay"
)
parser.add_argument(
    "--dropout", type=float, default=0.0,
    help="output dropout"
)
parser.add_argument(
    "--n_fft", type=int, default=512,
    help="number of fft bins"
)
parser.add_argument(
    "--hop_size", type=int, default=128,
    help="hop_size for stft-based features"
)
parser.add_argument(
    "--optimizer", type=str, required=True,
    help="which optimizer to use",
    choices=("adam", "momentum")
)
parser.add_argument(
    "--folds", type=int, required=True, nargs="+",
    help="which folds to use"
)
parser.add_argument(
    "--n_folds", type=int, default=4,
    help="number of folds"
)
parser.add_argument(
    "--kfold_seed", type=int, default=42,
    help="kfold seed"
)
parser.add_argument(
    "--num_workers", type=int, default=4,
    help="number of workers for data loader",
)
parser.add_argument(
    "--label", type=str, default="hierarchical_cnn_classifier",
    help="optional label",
)
args = parser.parse_args()

class_map = load_json(args.classmap)


with Experiment({
    "network": {
        "num_conv_blocks": args.num_conv_blocks,
        "start_deep_supervision_on": args.start_deep_supervision_on,
        "conv_base_depth": args.conv_base_depth,
        "growth_rate": args.growth_rate,
        "dropout": args.dropout
    },
    "data": {
        "_n_folds": args.n_folds,
        "_kfold_seed": args.kfold_seed,
        "n_fft": args.n_fft,
        "hop_size": args.hop_size,
        "_input_dim": args.n_fft // 2 + 1,
        "_n_classes": len(class_map)
    },
    "train": {
        "accumulation_steps": args.accumulation_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "scheduler": args.scheduler,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "_save_every": args.save_every,
        "weight_decay": args.weight_decay
    },
    "label": args.label
}) as experiment:

    config = experiment.config
    print()
    print("     ////// CONFIG //////")
    print(experiment.config)

    train_df = pd.read_csv(args.train_df)

    if args.max_samples:
        train_df = train_df.sample(args.max_samples).reset_index(drop=True)

    splits = list(train_validation_data(
        train_df.fname, train_df.labels,
        config.data._n_folds, config.data._kfold_seed))

    for fold in args.folds:

        train, valid = splits[fold]

        loader_kwargs = (
            {"num_workers": args.num_workers, "pin_memory": True}
            if torch.cuda.is_available() else {})

        experiment.register_directory("checkpoints")
        experiment.register_directory("predictions")

        train_loader = torch.utils.data.DataLoader(
            SoundDataset(
                audio_files=[
                    os.path.join(args.train_data_dir + fname)
                    for fname in train_df.fname[train]],
                labels=[item.split(",") for item in train_df.labels[train]],
                transform=Compose([
                    LoadAudio(),
                    MapLabels(class_map=class_map),
                    STFT(n_fft=args.n_fft, hop_size=args.hop_size),
                    DropFields(("audio", "filename", "sr")),
                    RenameFields({"stft": "signal"})
                ])
            ),
            shuffle=True,
            drop_last=True,
            batch_size=config.train.batch_size,
            collate_fn=make_collate_fn({"signal": math.log(STFT.eps)}),
            **loader_kwargs
        )

        valid_loader = torch.utils.data.DataLoader(
            SoundDataset(
                audio_files=[
                    os.path.join(args.train_data_dir + fname)
                    for fname in train_df.fname[valid]],
                labels=[item.split(",") for item in train_df.labels[valid]],
                transform=Compose([
                    LoadAudio(),
                    MapLabels(class_map=class_map),
                    STFT(n_fft=args.n_fft, hop_size=args.hop_size),
                    DropFields(("audio", "filename", "sr")),
                    RenameFields({"stft": "signal"})
                ])
            ),
            shuffle=False,
            batch_size=config.train.batch_size,
            collate_fn=make_collate_fn({"signal": math.log(STFT.eps)}),
            **loader_kwargs
        )

        model = HierarchicalCNNClassificationModel(experiment, device=args.device)

        scores = model.fit_validate(
            train_loader, valid_loader,
            epochs=experiment.config.train.epochs, fold=fold,
            log_interval=args.log_interval
        )

        best_metric = max(scores)
        experiment.register_result("fold{}.metric".format(fold), best_metric)

        torch.save(
            model.state_dict(),
            os.path.join(
                experiment.checkpoints,
                "fold_{}".format(fold),
                "final_model.pth")
        )

        if args.device == "cuda":
            torch.cuda.empty_cache()

    if all(
        "fold{}".format(k) in experiment.results.to_dict()
        for k in range(config.data._n_folds)):

        average_metric = np.mean([
            experiment.results.to_dict()["fold{}".format(k)]["metric"]
            for k in range(config.data._n_folds)
        ])

        experiment.register_result("avg_metric", average_metric)