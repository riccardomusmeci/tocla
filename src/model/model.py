import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Optimizer
from src.optimizer.optimizer import SAM
from torch.optim.lr_scheduler import _LRScheduler
from src.model.utils import enable_running_stats, disable_running_stats
from torchmetrics import Accuracy, F1Score, Precision, Recall, CalibrationError

torch.autograd.set_detect_anomaly(True)

class Model(pl.LightningModule):
    
    def __init__(
        self, 
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler = None,
        bn_to_zero_on_sam: bool = True
    ) -> None:
        super().__init__()
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        if isinstance(optimizer, SAM):
            self.with_sam = True
            self.automatic_optimization = False
        else:
            self.with_sam = False
        self.bn_to_zero_on_sam = bn_to_zero_on_sam
        
        if torch.cuda.is_available(): device = "cuda"
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
        
        # SAM - enabling BN only on the first forward pass
        if self.with_sam and self.bn_to_zero_on_sam:
            enable_running_stats(self.model)
            
        logits = self(x)
        loss = self.criterion(logits, target)
        
        # SAM support
        if self.with_sam:
            optimizer = self.optimizers()
            self.manual_backward(loss, retain_graph=True)
            optimizer.first_step(zero_grad=True)
            
             # SAM - disabling BN on the second forward pass
            if self.bn_to_zero_on_sam:
                 disable_running_stats(self.model)
            
            logits = self(x)
            loss_2 = self.criterion(logits, target)
            self.manual_backward(loss_2)
            optimizer.second_step(zero_grad=True)
        
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
    
    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [self.lr_scheduler]
        
        
        
        