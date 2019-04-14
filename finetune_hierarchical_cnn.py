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
from sklearn.model_selection import train_test_split

from datasets.sound_dataset import SoundDataset
from networks.classifiers import HierarchicalCNNClassificationModel
from ops.folds import train_validation_data
from ops.transforms import (
    Compose, DropFields, LoadAudio,
    STFT, MapLabels, RenameFields, MixUp)
from ops.utils import load_json, get_class_names_from_classmap, lwlrap
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
    "--test_data_dir", required=True, type=str,
    help="path to test data"
)
parser.add_argument(
    "--sample_submission", required=True, type=str,
    help="path sample submission"
)
parser.add_argument(
    "--pretrained_model", required=True, type=str,
    help="path to old experiment"
)
parser.add_argument(
    "--pretrained_fold", required=True, type=int,
    help="pretrained fold"
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
    "--holdout_size", type=float, default=0.0,
    help="size of holdout set"
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
    "--weight_decay", type=float, default=1e-5,
    help="weight decay"
)
parser.add_argument(
    "--dropout", type=float, default=0.0,
    help="internal dropout"
)
parser.add_argument(
    "--output_dropout", type=float, default=0.0,
    help="output dropout"
)
parser.add_argument(
    "--p_mixup", type=float, default=0.0,
    help="probability of the mixup augmentation"
)
parser.add_argument(
    "--switch_off_augmentations_on", type=int, default=20,
    help="on which epoch to remove augmentations"
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
    "--label", type=str, default="finetuned_hierarchical_cnn_classifier",
    help="optional label",
)
args = parser.parse_args()

class_map = load_json(args.classmap)

pretrained = Experiment(resume_from=args.pretrained_model)

with Experiment({
    "network": {
        "num_conv_blocks": pretrained.config.network.num_conv_blocks,
        "start_deep_supervision_on": pretrained.config.network.start_deep_supervision_on,
        "conv_base_depth": pretrained.config.network.conv_base_depth,
        "growth_rate": pretrained.config.network.growth_rate,
        "dropout": args.dropout,
        "output_dropout": args.output_dropout,
    },
    "data": {
        "_n_folds": args.n_folds,
        "_kfold_seed": args.kfold_seed,
        "n_fft": pretrained.config.data.n_fft,
        "hop_size": pretrained.config.data.hop_size,
        "_input_dim": pretrained.config.data.n_fft // 2 + 1,
        "_n_classes": len(class_map),
        "_holdout_size": args.holdout_size,
        "p_mixup": args.p_mixup
    },
    "train": {
        "accumulation_steps": args.accumulation_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "scheduler": args.scheduler,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "_save_every": args.save_every,
        "weight_decay": args.weight_decay,
        "switch_off_augmentations_on": args.switch_off_augmentations_on,
        "_pretrained_experiment": args.pretrained_model,
        "_pretrained_fold": args.pretrained_fold,
    },
    "label": args.label
}) as experiment:

    config = experiment.config
    print()
    print("     ////// CONFIG //////")
    print(experiment.config)

    train_df = pd.read_csv(args.train_df)
    test_df = pd.read_csv(args.sample_submission)

    if args.max_samples:
        train_df = train_df.sample(args.max_samples).reset_index(drop=True)
        test_df = test_df.sample(
            min(args.max_samples, len(test_df))).reset_index(drop=True)

    if args.holdout_size:
        keep, holdout = train_test_split(
            np.arange(len(train_df)), test_size=args.holdout_size,
            random_state=args.kfold_seed)
        holdout_df = train_df.iloc[holdout].reset_index(drop=True)
        train_df = train_df.iloc[keep].reset_index(drop=True)

    splits = list(train_validation_data(
        train_df.fname, train_df.labels,
        config.data._n_folds, config.data._kfold_seed))

    for fold in args.folds:

        print("\n\n   -----  Fold {}\n".format(fold))

        train, valid = splits[fold]

        loader_kwargs = (
            {"num_workers": args.num_workers, "pin_memory": True}
            if torch.cuda.is_available() else {})

        experiment.register_directory("checkpoints")
        experiment.register_directory("predictions")

        train_loader = torch.utils.data.DataLoader(
            SoundDataset(
                audio_files=[
                    os.path.join(args.train_data_dir, fname)
                    for fname in train_df.fname.values[train]],
                labels=[item.split(",") for item in train_df.labels.values[train]],
                transform=Compose([
                    LoadAudio(),
                    MapLabels(class_map=class_map),
                    MixUp(p=args.p_mixup),
                    STFT(n_fft=config.data.n_fft, hop_size=config.data.hop_size),
                    DropFields(("audio", "filename", "sr")),
                    RenameFields({"stft": "signal"})
                ]),
                clean_transform=Compose([
                    LoadAudio(),
                    MapLabels(class_map=class_map),
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
                    os.path.join(args.train_data_dir, fname)
                    for fname in train_df.fname.values[valid]],
                labels=[item.split(",") for item in train_df.labels.values[valid]],
                transform=Compose([
                    LoadAudio(),
                    MapLabels(class_map=class_map),
                    STFT(n_fft=config.data.n_fft, hop_size=config.data.hop_size),
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
        # load pretrained model
        model.load_state_dict(
            torch.load(
                os.path.join(
                    pretrained.checkpoints,
                    "fold_{}".format(args.pretrained_fold),
                    "best_model.pth"
                )
            )
        )

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

        # predictions
        model.load_best_model(fold)

        # validation

        val_preds = model.predict(valid_loader)
        val_predictions_df = pd.DataFrame(
            val_preds, columns=get_class_names_from_classmap(class_map))
        val_predictions_df["fname"] = train_df.fname[valid].values
        val_predictions_df.to_csv(
            os.path.join(
                experiment.predictions,
                "val_preds_fold_{}.csv".format(fold)
            ),
            index=False
        )
        del val_predictions_df

        # test
        test_loader = torch.utils.data.DataLoader(
            SoundDataset(
                audio_files=[
                    os.path.join(args.test_data_dir, fname)
                    for fname in test_df.fname.values],
                transform=Compose([
                    LoadAudio(),
                    STFT(n_fft=config.data.n_fft, hop_size=config.data.hop_size),
                    DropFields(("audio", "filename", "sr")),
                    RenameFields({"stft": "signal"})
                ])
            ),
            shuffle=False,
            batch_size=config.train.batch_size,
            collate_fn=make_collate_fn({"signal": math.log(STFT.eps)}),
            **loader_kwargs
        )

        test_preds = model.predict(test_loader)
        test_predictions_df = pd.DataFrame(
            test_preds, columns=get_class_names_from_classmap(class_map))
        test_predictions_df["fname"] = test_df.fname
        test_predictions_df.to_csv(
            os.path.join(
                experiment.predictions,
                "test_preds_fold_{}.csv".format(fold)
            ),
            index=False
        )
        del test_predictions_df

        # holdout
        if args.holdout_size:
            holdout_loader = torch.utils.data.DataLoader(
                SoundDataset(
                    audio_files=[
                        os.path.join(args.train_data_dir, fname)
                        for fname in holdout_df.fname.values],
                    labels=[item.split(",") for item in holdout_df.labels.values],
                    transform=Compose([
                        LoadAudio(),
                        MapLabels(class_map),
                        STFT(n_fft=config.data.n_fft, hop_size=config.data.hop_size),
                        DropFields(("audio", "filename", "sr")),
                        RenameFields({"stft": "signal"})
                    ])
                ),
                shuffle=False,
                batch_size=config.train.batch_size,
                collate_fn=make_collate_fn({"signal": math.log(STFT.eps)}),
                **loader_kwargs
            )

            holdout_metric = model.evaluate(holdout_loader)
            experiment.register_result(
                "fold{}.holdout_metric".format(fold), holdout_metric)

            print("\nHoldout metric: {:.4f}".format(holdout_metric))

        if args.device == "cuda":
            torch.cuda.empty_cache()

    # global metric

    if all(
        "fold{}".format(k) in experiment.results.to_dict()
        for k in range(config.data._n_folds)):

        val_df_files = [
            os.path.join(
                experiment.predictions,
                "val_preds_fold_{}.csv".format(fold)
            )
            for fold in range(config.data._n_folds)
        ]

        val_predictions_df = pd.concat([
            pd.read_csv(file) for file in val_df_files]).reset_index(drop=True)

        labels = np.asarray([
            item["labels"] for item in SoundDataset(
                audio_files=train_df.fname.tolist(),
                labels=[item.split(",") for item in train_df.labels.values],
                transform=MapLabels(class_map)
            )
        ])

        val_labels_df = pd.DataFrame(
            labels, columns=get_class_names_from_classmap(class_map))
        val_labels_df["fname"] = train_df.fname

        assert set(val_predictions_df.fname) == set(val_labels_df.fname)

        val_predictions_df.sort_values(by="fname", inplace=True)
        val_labels_df.sort_values(by="fname", inplace=True)

        metric = lwlrap(
            val_labels_df.drop("fname", axis=1).values,
            val_predictions_df.drop("fname", axis=1).values
        )

        experiment.register_result("metric", metric)

    # submission

    test_df_files = [
        os.path.join(
            experiment.predictions,
            "test_preds_fold_{}.csv".format(fold)
        )
        for fold in range(config.data._n_folds)
    ]

    if all(os.path.isfile for file in test_df_files):
        test_dfs = [pd.read_csv(file) for file in test_df_files]
        submission_df = pd.DataFrame({"fname": test_dfs[0].fname.values})
        for c in get_class_names_from_classmap(class_map):
            submission_df[c] = np.mean([d[c].values for d in test_dfs], axis=0)
        submission_df.to_csv(
            os.path.join(experiment.predictions, "submission.csv"), index=False)