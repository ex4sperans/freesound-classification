import os
import math
import itertools
from collections import defaultdict, OrderedDict, deque

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from tensorboardX import SummaryWriter
from torch.nn.functional import binary_cross_entropy_with_logits

from ops.training import OPTIMIZERS, make_scheduler, make_step
from networks.losses import binary_cross_entropy, focal_loss, lsep_loss
from ops.utils import plot_projection


class APCModel(nn.Module):

    def __init__(self, experiment, device="cuda"):
        super().__init__()

        self.device = device

        self.experiment = experiment
        self.config = experiment.config

        self.input_norm = nn.LayerNorm(
            (self.config.data._input_dim,), elementwise_affine=False)

        self.rnn = nn.LSTM(
            self.config.data._input_dim, self.config.network.rnn_size,
            num_layers=self.config.network.rnn_layers,
            batch_first=True
        )

        self.output_norm = nn.LayerNorm((self.config.network.rnn_size,))

        self.prediction_transforms = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(
                    self.config.network.rnn_size,
                    self.config.data._input_dim)
            )
            for steps in range(self.config.network.prediction_steps)
        ])

        self.to(self.device)

    def forward(self, signal):

        # signal = signal.permute(0, 2, 1)
        signal = self.input_norm(signal)
        # signal = signal.permute(0, 2, 1)

        output, state = self.rnn(signal)
        output = self.output_norm(output)

        losses = []
        predictions = []

        for step, affine in enumerate(self.prediction_transforms, start=1):

            shifted_output = output[:, :-step, :]
            shifted_signal = signal.detach()[:, step:, :]

            prediction = affine(shifted_output)
            predictions.append(prediction)

            loss = torch.abs(shifted_signal - prediction)
            loss = loss.sum(-1)
            loss = loss.mean()

            losses.append(loss)

        r = dict(
            losses=losses,
            output=output,
            predictions=predictions
        )

        return r

    def add_scalar_summaries(
        self, losses, writer, global_step):

        # scalars
        for k, loss in enumerate(losses, start=1):
            writer.add_scalar("loss_{k}".format(k=k), loss, global_step)

    def add_image_summaries(
        self, signal, output, predictions, global_step, writer, to_plot=8):

        if len(signal) > to_plot:
            signal = signal[:to_plot]
            output = output[:to_plot]
            predictions = [p[:to_plot] for p in predictions]

        # signal
        image_grid = torchvision.utils.make_grid(
            signal.data.cpu().unsqueeze(1),
            normalize=True, scale_each=True
        )
        writer.add_image("signal", image_grid, global_step)
        # output
        image_grid = torchvision.utils.make_grid(
            output.data.cpu().unsqueeze(1),
            normalize=True, scale_each=True
        )
        writer.add_image("output", image_grid, global_step)

        for k, p in enumerate(predictions, start=1):
            image_grid = torchvision.utils.make_grid(
                p.data.cpu().unsqueeze(1),
                normalize=True, scale_each=True
            )
            writer.add_image(
                "prediction_{k}".format(k=k), image_grid, global_step)

    def add_projection_summary(self, image, global_step, writer, name="projection"):
        writer.add_image(name, image.transpose(2, 0, 1), global_step)

    def train_epoch(self, train_loader,
                    epoch, log_interval, write_summary=True):

        self.train()

        print(
            "\n" + " " * 10 + "****** Epoch {epoch} ******\n"
            .format(epoch=epoch)
        )

        history = deque(maxlen=30)

        self.optimizer.zero_grad()
        accumulated_loss = 0

        with tqdm(total=len(train_loader), ncols=80) as pb:

            for batch_idx, sample in enumerate(train_loader):

                self.global_step += 1

                make_step(self.scheduler, step=self.global_step)

                signal, labels = (
                    sample["signal"].to(self.device),
                    sample["labels"].to(self.device)
                )

                outputs = self(signal)

                losses = outputs["losses"]

                loss = (
                    sum(losses)
                ) / self.config.train.accumulation_steps

                loss.backward()
                accumulated_loss += loss

                if batch_idx % self.config.train.accumulation_steps == 0:
                    self.optimizer.step()
                    accumulated_loss = 0
                    self.optimizer.zero_grad()

                history.append(loss.item())

                pb.update()
                pb.set_description(
                    "Loss: {:.4f}".format(
                        np.mean(history)))

                if batch_idx % log_interval == 0:
                    self.add_scalar_summaries(
                        [loss.item() for loss in losses],
                        self.train_writer, self.global_step)

                if batch_idx == 0:
                    self.add_image_summaries(
                        signal,
                        outputs["output"],
                        outputs["predictions"],
                        self.global_step, self.train_writer)

    def evaluate(self, loader, verbose=False, write_summary=False, epoch=None):

        self.eval()

        valid_losses = [0 for _ in range(self.config.network.prediction_steps)]

        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, sample in enumerate(loader):

                signal, labels = (
                    sample["signal"].to(self.device),
                    sample["labels"].to(self.device)
                )

                outputs = self(signal)

                losses = outputs["losses"]

                multiplier = len(signal) / len(loader.dataset)

                for k, loss in enumerate(losses):
                    valid_losses[k] += loss.item() * multiplier

                all_outputs.extend(
                    outputs["output"].permute(0, 2, 1).data.cpu().numpy())
                all_labels.extend(labels.data.cpu().numpy())

        valid_loss = sum(valid_losses)

        all_labels = np.array(all_labels)

        if write_summary:
            self.add_scalar_summaries(
                valid_losses,
                writer=self.valid_writer, global_step=self.global_step
            )
            if epoch % self.config.train._proj_interval == 0:
                self.add_projection_summary(
                    plot_projection(
                        all_outputs, all_labels, frames_per_example=5, newline=True),
                    writer=self.valid_writer, global_step=self.global_step,
                    name="projection_output")

        if verbose:
            print("\nValidation loss: {:.4f}".format(valid_loss))

        return -valid_loss

    def validation(self, valid_loader, epoch):
        return self.evaluate(
            valid_loader,
            verbose=True, write_summary=True, epoch=epoch)

    def predict(self, loader):

        self.eval()

        all_class_probs = []

        with torch.no_grad():
            for sample in loader:

                signal = sample["signal"].to(self.device)

                outputs = self(signal)

                class_logits = outputs["class_logits"].squeeze()

                class_probs = torch.sigmoid(class_logits).data.cpu().numpy()
                all_class_probs.extend(class_probs)

        all_class_probs = np.asarray(all_class_probs)

        return all_class_probs

    def fit_validate(self, train_loader, valid_loader, epochs, fold,
                     log_interval=25):


        self.experiment.register_directory("summaries")
        self.train_writer = SummaryWriter(
            log_dir=os.path.join(
                self.experiment.summaries,
                "fold_{}".format(fold),
                "train"
            )
        )
        self.valid_writer = SummaryWriter(
            log_dir=os.path.join(
                self.experiment.summaries,
                "fold_{}".format(fold),
                "valid"
            )
        )

        os.makedirs(
            os.path.join(
                self.experiment.checkpoints,
                "fold_{}".format(fold)),
            exist_ok=True
        )

        self.global_step = 0
        self.make_optimizer(max_steps=len(train_loader) * epochs)

        scores = []
        best_score = 0

        for epoch in range(epochs):

            make_step(self.scheduler, epoch=epoch)

            if epoch == self.config.train.switch_off_augmentations_on:
                train_loader.dataset.transform.switch_off_augmentations()

            self.train_epoch(
                train_loader, epoch,
                log_interval, write_summary=True
            )
            validation_score = self.validation(valid_loader, epoch)
            scores.append(validation_score)

            if epoch % self.config.train._save_every == 0:
                print("\nSaving model on epoch", epoch)
                torch.save(
                    self.state_dict(),
                    os.path.join(
                        self.experiment.checkpoints,
                        "fold_{}".format(fold),
                        "model_on_epoch_{}.pth".format(epoch)
                    )
                )

            if validation_score > best_score:
                torch.save(
                    self.state_dict(),
                    os.path.join(
                        self.experiment.checkpoints,
                        "fold_{}".format(fold),
                        "best_model.pth"
                    )
                )
                best_score = validation_score

        return scores

    def make_optimizer(self, max_steps):

        optimizer = OPTIMIZERS[self.config.train.optimizer]
        optimizer = optimizer(
            self.parameters(),
            self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay
        )
        self.optimizer = optimizer
        self.scheduler = make_scheduler(
            self.config.train.scheduler, max_steps=max_steps)(optimizer)

    def load_best_model(self, fold):

        self.load_state_dict(
            torch.load(
                os.path.join(
                    self.experiment.checkpoints,
                    "fold_{}".format(fold),
                    "best_model.pth"
                )
            )
        )

