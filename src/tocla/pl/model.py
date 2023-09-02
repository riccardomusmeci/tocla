from typing import Dict, List, Literal, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Accuracy, CalibrationError, F1Score, Precision, Recall

from ..model.utils import disable_running_stats, enable_running_stats
from ..optimizer.sam import SAM
from ..utils import get_device


class ClassificationModelModule(pl.LightningModule):
    """PyTorchLightning module that combines a model with a loss function, an
    optimizer, a learning rate scheduler, and evaluation metrics.

    Args:
        model (nn.Module): model to train.
        loss (_Loss): loss function to use during training.
        optimizer (Optimizer): optimizer to use during training.
        lr_scheduler (_LRScheduler): learning rate scheduler to use during training.
        num_classes (int): number of classes in the classification task.
        task (Literal["binary", "multiclass", "multilabel"]): type of classification task.
        average (Literal["micro", "macro", "weighted", None], optional): type of averaging to use for the evaluation metrics. Defaults to macro.
        bn_to_zero_on_sam (bool, optional): whether to set the running mean and variance of BatchNorm layers to zero during the forward pass when using the SAM optimizer. Defaults to True.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: _Loss,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        num_classes: int,
        task: Literal["binary", "multiclass"],
        average: Literal["micro", "macro", "weighted", None] = "macro",
        bn_to_zero_on_sam: bool = True,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.loss = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.val_outputs = []  # type: ignore
        self.task = task
        self.num_classes = num_classes
        self.average = average

        device = get_device()

        if isinstance(self.optimizer, SAM):
            self.with_sam = True
            self.automatic_optimization = False
        else:
            self.with_sam = False
        self.bn_to_zero_on_sam = bn_to_zero_on_sam

        assert self.task in [
            "binary",
            "multiclass",
        ], "Task must be either binary or multiclass. Multilabel not supported yet."

        self.metrics = {
            "acc": Accuracy(task=self.task, num_classes=self.num_classes, average=self.average).to(device),  # type: ignore
            "f1": F1Score(task=self.task, num_classes=self.num_classes, average=self.average).to(device),  # type: ignore
            "prec": Precision(task=self.task, num_classes=self.num_classes, average=self.average).to(device),  # type: ignore
            "rec": Recall(task=self.task, num_classes=self.num_classes, average=self.average).to(device),  # type: ignore
            "cal_err": CalibrationError(task=self.task, num_classes=self.num_classes, average=self.average).to(device),  # type: ignore
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: logits
        """
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:  # type: ignore
        """Training step.

        Args:
            batch: batch
            batch_idx: batch idx


        Returns:
            torch.Tensor: train loss
        """

        x, target = batch

        # SAM - enabling BN only on the first forward pass
        if self.with_sam and self.bn_to_zero_on_sam:
            enable_running_stats(self.model)

        logits = self(x)
        loss = self.loss(logits, target)

        # SAM support
        if self.with_sam:
            optimizer = self.optimizers()
            self.manual_backward(loss, retain_graph=True)
            optimizer.first_step(zero_grad=True)  # type: ignore

            # SAM - disabling BN on the second forward pass
            if self.bn_to_zero_on_sam:
                disable_running_stats(self.model)

            logits = self(x)
            loss_2 = self.loss(logits, target)
            self.manual_backward(loss_2)
            optimizer.second_step(zero_grad=True)  # type: ignore

        self.log("loss_train", loss, sync_dist=True, prog_bar=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore
        """Validation step.

        Args:
            batch: batch
            batch_idx: batch index
        """
        x, target = batch
        logits = self(x)
        loss = self.loss(logits, target)

        if self.task == "binary":
            preds = torch.sigmoid(logits)
        else:
            preds = logits

        # update metrics
        for m in self.metrics:
            self.metrics[m].update(preds=preds, target=target)
        return {"val_step_loss": loss}

    def on_validation_batch_end(self, outputs, batch, batch_idx):  # type: ignore
        """Validation batch end.

        Args:
            outputs: model outputs
            batch: batch
            batch_idx: batch index
        """
        self.val_outputs.append(outputs)

    def on_validation_epoch_end(self):  # type: ignore
        """Validation epoch end."""
        # logging loss
        avg_loss = torch.stack([x["val_step_loss"] for x in self.val_outputs]).mean()
        self.log("loss_val", avg_loss, sync_dist=True, prog_bar=True)

        # logging metrics
        epoch_metrics = {m: self.metrics[m].compute() for m in self.metrics.keys()}
        for m in self.metrics.keys():
            self.metrics[m].reset()
        for m, val in epoch_metrics.items():
            self.log(f"{m}", val, sync_dist=True, prog_bar=True)

    def configure_optimizers(self) -> Union[List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]:
        """Configure optimizer and lr scheduler.

        Returns:
            Union[List[Optimizer], Tuple[List[Optimizer], List[_LRScheduler]]]: optimizer and lr scheduler
        """
        if self.lr_scheduler is None:
            return [self.optimizer]  # type: ignore
        else:
            return [self.optimizer], [self.lr_scheduler]
