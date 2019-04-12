import os
import math
import itertools
from collections import defaultdict, OrderedDict, deque

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.utils
from tensorboardX import SummaryWriter

from ops.training import OPTIMIZERS, make_scheduler, make_step
from networks.losses import binary_cross_entropy, focal_loss
from ops.utils import lwlrap


class ResnetBlock(nn.Module):

    def __init__(self, depth):
        super().__init__()

        self.conv1 = nn.Conv1d(depth, depth, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(depth)
        self.conv2 = nn.Conv1d(depth, depth, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(depth)
        self.conv3 = nn.Conv1d(depth, depth, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(depth)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class HierarchicalCNNClassificationModel(nn.Module):

    def __init__(self, experiment, device="cuda"):
        super().__init__()

        self.device = device

        self.experiment = experiment
        self.config = experiment.config

        self.conv_modules = torch.nn.ModuleList()

        total_depth = 0

        for k in range(self.config.network.num_conv_blocks):

            input_size = self.config.data._input_dim if not k else depth
            depth = int(
                self.config.network.growth_rate ** k
                * self.config.network.conv_base_depth)

            if k >= self.config.network.start_deep_supervision_on:
                total_depth += depth

            modules = [nn.BatchNorm1d(input_size)] if not k else []
            modules.extend([
                nn.Conv1d(
                    input_size,
                    depth,
                    kernel_size=3,
                    padding=0
                ),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.BatchNorm1d(depth),
                nn.ReLU(inplace=True),
                ResnetBlock(depth)
            ])

            self.conv_modules.append(nn.Sequential(*modules))

        self.global_maxpool = nn.AdaptiveMaxPool1d(1)

        self.output_transform = nn.Sequential(
            nn.BatchNorm1d(total_depth),
            nn.Dropout(p=self.config.network.dropout),
            nn.Linear(total_depth, self.config.data._n_classes)
        )

        self.to(self.device)

    def forward(self, signal):

        signal = signal.permute(0, 2, 1)

        features = []

        h = signal
        for k, module in enumerate(self.conv_modules):
            h = module(h)
            if k >= self.config.network.start_deep_supervision_on:
                features.append(self.global_maxpool(h).squeeze(-1))

        features = torch.cat(features, -1)

        class_logits = self.output_transform(features)

        r = dict(
            class_logits=class_logits
        )

        return r

    def add_scalar_summaries(
        self, loss, metric, writer, global_step):

        # scalars
        writer.add_scalar("loss", loss, global_step)
        writer.add_scalar("metric", metric, global_step)

    def add_image_summaries(self, signal, global_step, writer, to_plot=8):

        if len(signal) > to_plot:
            signal = signal[:to_plot]

        # image
        image_grid = torchvision.utils.make_grid(
            signal.data.cpu().unsqueeze(1),
            normalize=True, scale_each=True
        )
        writer.add_image("signal", image_grid, global_step)

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
                    sample["labels"].to(self.device).float()
                )

                outputs = self(signal)

                class_logits = outputs["class_logits"].squeeze()

                loss = (
                    focal_loss(
                        class_logits,
                        labels
                    )
                ) / self.config.train.accumulation_steps

                loss.backward()
                accumulated_loss += loss

                if batch_idx % self.config.train.accumulation_steps == 0:
                    self.optimizer.step()
                    accumulated_loss = 0
                    self.optimizer.zero_grad()

                probs = torch.sigmoid(class_logits).data.cpu().numpy()
                labels = labels.data.cpu().numpy()

                metric = lwlrap(labels, probs)
                history.append(metric)

                pb.update()
                pb.set_description(
                    "Loss: {:.4f}, Metric: {:.4f}".format(
                        loss.item(), np.mean(history)))

                if batch_idx % log_interval == 0:
                    self.add_scalar_summaries(
                        loss.item(), metric, self.train_writer, self.global_step)

                if batch_idx == 0:
                    self.add_image_summaries(
                        signal, self.global_step, self.train_writer)

    def validation(self, valid_loader, epoch):

        self.eval()

        valid_loss = 0

        all_class_probs = []
        all_labels = []

        write_on = np.random.randint(len(valid_loader))

        with torch.no_grad():
            for batch_idx, sample in enumerate(valid_loader):

                signal, labels = (
                    sample["signal"].to(self.device),
                    sample["labels"].to(self.device).float()
                )

                outputs = self(signal)

                class_logits = outputs["class_logits"].squeeze()

                loss = (
                    focal_loss(
                        class_logits,
                        labels,
                    )
                ).item()

                multiplier = len(labels) / len(valid_loader.dataset)

                valid_loss += loss * multiplier

                class_probs = torch.sigmoid(class_logits).data.cpu().numpy()
                labels = labels.data.cpu().numpy()

                all_class_probs.extend(class_probs)
                all_labels.extend(labels)

            all_class_probs = np.asarray(all_class_probs)
            all_labels = np.asarray(all_labels)

            metric = lwlrap(all_labels, all_class_probs)

            self.add_scalar_summaries(
                valid_loss,
                metric,
                writer=self.valid_writer, global_step=self.global_step
            )

            print("\nValidation loss: {:.4f}".format(valid_loss))
            print("Validation metric: {:.4f}".format(metric))

            return metric

    def predict(self, loader):

        self.device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        self.eval()

        all_class_probs = []

        with torch.no_grad():
            for sample in loader:

                signal = sample["signal"].to(self.device)

                outputs = self(signal)

                class_logits = outputs["class_logits"].squeeze()

                class_probs = torch.sigmoid(class_logits).data.cpu().numpy()
                all_class_probs.extend(class_probs)

        all_class_probs = np.asarray(all_class_probs).tolist()

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
        self.make_optimizer()

        scores = []

        for epoch in range(epochs):

            make_step(self.scheduler, epoch=epoch)

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

        return scores

    def make_optimizer(self):

        optimizer = OPTIMIZERS[self.config.train.optimizer]
        optimizer = optimizer(
            self.parameters(),
            self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay
        )
        self.optimizer = optimizer
        self.scheduler = make_scheduler(self.config.train.scheduler)(optimizer)

