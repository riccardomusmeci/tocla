import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import Accuracy, F1Score, Precision, Recall, CalibrationError

class Model(pl.LightningModule):
    
    def __init__(
        self, 
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler = None,
    ) -> None:
        super().__init__()
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        if torch.has_cuda: device = "cuda"
        if torch.has_mps: device = "mps"
        
        self.metrics = {
            "acc": Accuracy().to(device),
            "f1": F1Score().to(device),
            "prec": Precision().to(device),
            "recall": Recall().to(device),
            "cal_err": CalibrationError().to(device)
        }
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        
        x, target = batch
        logits = self(x)
        loss = self.criterion(logits, target)
        
        self.log("loss/train", loss, sync_dist=True, prog_bar=True)
        self.log("lr", self.lr_scheduler.get_last_lr()[0], prog_bar=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        
        x, target = batch
        logits = self(x)
        loss = self.criterion(logits, target)
        
        for m in self.metrics:
            self.metrics[m].update(
                preds=logits,
                target=target
            )
        
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        vals = {
            m: self.metrics[m].compute() for m in self.metrics
        }
        for m in self.metrics: self.metrics[m].reset()
        # just for lightning compatibility
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("loss/val", avg_loss, sync_dist=True, prog_bar=True)
        for m, val in vals.items():
            self.log(f"{m}/val", val, sync_dist=True, prog_bar=True)
        
    def training_epoch_end(self, outputs):
        pass
        
    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [self.lr_scheduler]
        
        
        
        