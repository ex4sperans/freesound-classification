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



class CPCModel(nn.Module):

    def __init__(self, experiment, device="cuda"):
        super().__init__()

        self.device = device

        self.experiment = experiment
        self.config = experiment.config

        encoder_layers = []

        for k in range(self.config.network.n_encoder_layers):
            input_size = self.config.data._input_dim if not k else depth
            depth = int(
                self.config.network.growth_rate ** k
                * self.config.network.conv_base_depth)
            modules = [nn.BatchNorm1d(input_size)] if not k else []
            modules.extend([
                nn.Conv1d(
                    input_size,
                    depth,
                    kernel_size=3,
                    padding=0,
                    stride=2
                ),
                nn.PReLU(depth)
            ])
            encoder_layers.extend(modules)

        encoder_layers.append(nn.BatchNorm1d(depth))

        self.encoder = nn.Sequential(*encoder_layers)

        self.context_network = nn.GRU(
            depth, self.config.network.context_size,
            num_layers=1,
            batch_first=True
        )

        self.coupling_transforms = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    self.config.network.context_size, depth, kernel_size=1)
            )
            for steps in range(self.config.network.prediction_steps)
        ])

        self.to(self.device)

    def forward(self, signal):

        signal = signal.permute(0, 2, 1)
        # z is (n, depth, steps)
        z = self.encoder(signal)
        # c is (n, context_size, steps)
        c, state = self.context_network(z.permute(0, 2, 1))
        c = c.permute(0, 2, 1)

        losses = []

        for step, affine in enumerate(self.coupling_transforms, start=1):

            a = affine(c)

            # logits is (n, steps, steps)
            logits = torch.bmm(z.permute(0, 2, 1), a)

            labels = torch.eye(logits.size(2) - step, device=z.device)
            labels = torch.nn.functional.pad(labels, (0, step, step, 0))
            labels = labels.unsqueeze(0).expand_as(logits)

            loss = binary_cross_entropy_with_logits(logits, labels)
            losses.append(loss)

        r = dict(
            losses=losses,
            z=z,
            c=c
        )

        return r

    def add_scalar_summaries(
        self, losses, writer, global_step):

        # scalars
        for k, loss in enumerate(losses, start=1):
            writer.add_scalar("loss_{k}".format(k=k), loss, global_step)

    def add_image_summaries(self, signal, c, z, global_step, writer, to_plot=8):

        if len(c) > to_plot:
            signal = signal[:to_plot]
            c = c[:to_plot]
            z = z[:to_plot]

        # signal
        image_grid = torchvision.utils.make_grid(
            signal.data.cpu().unsqueeze(1),
            normalize=True, scale_each=True
        )
        writer.add_image("signal", image_grid, global_step)
        # z
        image_grid = torchvision.utils.make_grid(
            z.data.cpu().unsqueeze(1),
            normalize=True, scale_each=True
        )
        writer.add_image("z", image_grid, global_step)
        # c
        image_grid = torchvision.utils.make_grid(
            c.data.cpu().unsqueeze(1),
            normalize=True, scale_each=True
        )
        writer.add_image("c", image_grid, global_step)

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
                        outputs["c"].permute(0, 2, 1),
                        outputs["z"].permute(0, 2, 1),
                        self.global_step, self.train_writer)

    def evaluate(self, loader, verbose=False, write_summary=False, epoch=None):

        self.eval()

        valid_losses = [0 for _ in range(self.config.network.prediction_steps)]

        all_c = []
        all_z = []
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

                all_c.extend(
                    outputs["c"].permute(0, 2, 1).data.cpu().numpy())
                all_z.extend(
                    outputs["z"].permute(0, 2, 1).data.cpu().numpy())
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
                        all_c, all_labels, frames_per_example=5, newline=True),
                    writer=self.valid_writer, global_step=self.global_step,
                    name="projection_c")
                self.add_projection_summary(
                    plot_projection(all_z, all_labels, frames_per_example=5),
                    writer=self.valid_writer, global_step=self.global_step,
                    name="projection_z")

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

