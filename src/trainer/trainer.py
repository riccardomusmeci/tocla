import torch
from tqdm import tqdm
import torch.nn as nn
from src.utils import Device
from torch.optim import Optimizer
from src.optimizer.sam import SAM
from src.utils import ModelCheckpoint
from typing import Tuple, Dict, Union
from torch.utils.data import DataLoader
from src.utils import timeit, TimeMonitor
from torch.optim.lr_scheduler import _LRScheduler
from src.model.utils import enable_running_stats, disable_running_stats
from torchmetrics import Accuracy, F1Score, Precision, Recall, CalibrationError

class Trainer:
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        max_epochs: int,
        model_checkpoint: ModelCheckpoint,
        check_val_every_n_epoch: int = 1,
        check_train_every_n_iter: int = 10,
        gradient_clip_val: Union[int, float] = None,
        gradient_clip_algorithm: str = "norm",
        device: str = "mps",
    ) -> None:
        """Trainer

        Args:
            model (nn.Module): model nn.Module
            train_dataloader (DataLoader): train
            val_dataloader (DataLoader): validation dataloader
            criterion (nn.Module): loss criterion
            optimizer (Optimizer): optimizer
            scheduler (_LRScheduler): scheduler
            max_epochs (int): max number of epochs
            model_checkpoint (ModelCheckpoint): model checkpoint class instance
            check_val_every_n_epoch (int, optional): how often running validation. Defaults to 1.
            device (str, optional): device. Defaults to "mps".
        """
        
        assert device in ["mps", "cuda"], "Device must be either mps or cuda, not {device}."
        assert gradient_clip_algorithm in ["val", "norm"], "Gradient clip algorithm must be one of val or norm, not {gradient_clip_algorithm}"
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.check_train_every_n_iter = check_train_every_n_iter
        self.device = device
        self.model_checkpoint = model_checkpoint
        self.epoch_val_metrics = None
        
        # Gradient clip
        self.gradient_clip_algorithm = torch.nn.utils.clip_grad.clip_grad_norm_ if gradient_clip_algorithm=="norm" \
            else torch.nn.utils.clip_grad.clip_grad_value_
        self.gradient_clip_val = gradient_clip_val
        
        # [SAM] checking Optimizer type
        if isinstance(self.optimizer, SAM):
            self.with_sam = True
            self.bn_to_zero = self.optimizer.bn_to_zero
        else:
            self.with_sam = False
            self.bn_to_zero = False
                
        self.metrics = {
            "acc": Accuracy().to(self.device),
            "f1": F1Score().to(self.device),
            "prec": Precision().to(self.device),
            "rec": Recall().to(self.device),
            "cal_err": CalibrationError().to(self.device)
        }
        
        self.model.to(self.device)
        self.criterion.to(self.device)
        
    @timeit
    def training_step(
        self, 
        batch: Tuple, 
        batch_idx: int
    ) -> float:
        """training step

        Args:
            batch (Tuple): batch from dataloader
            batch_idx (int): batch index

        Returns:
            float: loss
        """
 
        x, target = batch
        
        if not self.with_sam:
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, target)
            if self.gradient_clip_val is not None: self.gradient_clip_algorithm(self.model.parameters(), self.gradient_clip_val)
            loss.backward()
            self.optimizer.step()
        else:
            # First SAM step
            if self.bn_to_zero:
                enable_running_stats(self.model)
            logits = self.model(x)
            loss = self.criterion(logits, target)
            if self.gradient_clip_val is not None: self.gradient_clip_algorithm(self.model.parameters(), self.gradient_clip_val)
            loss.mean().backward()
            self.optimizer.first_step(zero_grad=True)
            
            # Second SAM step
            if self.bn_to_zero:
                disable_running_stats(self.model)
            logits_2 = self.model(x)
            loss_2 = self.criterion(logits_2, target)
            if self.gradient_clip_val is not None: self.gradient_clip_algorithm(self.model.parameters(), self.gradient_clip_val)
            loss_2.mean().backward()
            self.optimizer.second_step(zero_grad=True)
            
        return loss.item()
        
    def training_epoch(
        self,
        epoch: int
    ) -> float:
        """training epoch

        Args:
            epoch (int): training epoch

        Returns:
            float: average training loss
        """
        
        time_monitor = TimeMonitor(len(self.train_dataloader))

        # setting model to train mode
        self.model.train()
        
        train_loss = 0
        for batch_idx, batch in enumerate(self.train_dataloader, 0):
            # training step
            step_loss, t_step = self.training_step(
                batch=(el.to(self.device) for el in batch),
                batch_idx=batch_idx
            )
            train_loss += step_loss
            # updating logging metrics
            time_monitor.update(t_step)
            # checking training metrics
            if (batch_idx+1)%self.check_train_every_n_iter==0:
                val_str = "" if self.epoch_val_metrics is None else "loss/val={:.4f}-f1/val={:.4f}-".format(
                                                                                self.epoch_val_metrics['loss_val'],
                                                                                self.epoch_val_metrics['f1_val']
                                                                            )
                print("> epoch=[{}/{}]-batch=[{}/{}]-loss/train={:.4f}-lr={}-{}time={}<{}".format(
                    epoch,
                    self.max_epochs,
                    batch_idx,
                    len(self.train_dataloader),
                    train_loss/(self.check_train_every_n_iter),
                    self.scheduler.get_last_lr()[0],
                    val_str,
                    time_monitor.elapsed(),
                    time_monitor.estimate()
                ))
                train_loss = 0
            
        self.scheduler.step()
                  
    @torch.no_grad()
    @timeit
    def validation_step(
        self, 
        batch: Tuple,
        batch_idx: int,
        sanity: bool = False
    ) -> float:
        """validation step

        Args:
            batch (Tuple): batch from dataloader
            batch_idx (int): batch index

        Returns:
            float: loss
        """
        
        x, target = batch
        logits = self.model(x)
        loss = self.criterion(logits, target)
        
        if sanity: return loss.item()
        
        for m in self.metrics:
            self.metrics[m].update(
                preds=torch.softmax(logits, dim=1),
                target=target
            )
        return loss.item()
    
    @torch.no_grad()
    def validation_epoch(
        self,
        epoch: int,
    ):
        """validation epoch

        Returns:
            float: average validation loss
        """
        
        # setting model to eval
        self.model.eval()
        
        print("> validation step")
        val_loss = 0
        for batch_idx, batch in tqdm(enumerate(self.val_dataloader, 0), total=len(self.val_dataloader)):
            # validation step
            step_loss, t_step = self.validation_step(
                batch=(el.to(self.device) for el in batch),
                batch_idx=batch_idx
            )
            val_loss += step_loss
        
        self.epoch_val_metrics = {
            f"{m}_val": self.metrics[m].compute() for m in self.metrics
        }
        self.epoch_val_metrics["loss_val"] = val_loss/len(self.val_dataloader)
        
        print(f"> epoch={epoch}-loss/val={self.epoch_val_metrics['loss_val']} - metrics:")
        for k, v in self.epoch_val_metrics.items():
            print(f"\t> {k}={v}")
            
    @torch.no_grad()        
    def _sanity_check(self):
        """performs sanity check before training model

        Raises:
            e: exception that broke sanity check
        """
        print(f"> Running Sanity Check")
        try:
            for batch_idx, batch in tqdm(enumerate(self.val_dataloader, 0), total=2, desc="Sanity Check"):
                # validation step
                _, _ = self.validation_step(
                    batch=(el.to(self.device) for el in batch),
                    batch_idx=batch_idx,
                    sanity=True
                )
                if batch_idx == 2:
                    break
        except Exception as e:
            raise e
        print(f"> Sanity Check - OK")
    
    def reset_metrics(self):
         for m in self.metrics: 
            self.metrics[m].reset()
            
    def fit(self):
        """fit method
        """
        self._sanity_check()
        print(f"> Starting training for {self.max_epochs} epochs on device {Device()}.")
        for epoch in range(self.max_epochs):
            print("-"*80)
            self.training_epoch(epoch=epoch)
            if (epoch+1) % self.check_val_every_n_epoch == 0:
                self.validation_epoch(epoch=epoch)
                self.reset_metrics()
                self.model_checkpoint.step(
                    epoch=epoch,
                    metrics=self.epoch_val_metrics,
                    state_dict=self.model.state_dict()
                )
                if self.model_checkpoint.patience_over:
                    print(f"> Patience over at epoch {epoch}. Ending training.")
                    print("-"*80)
                    break
        print(f"> Training over {epoch+1} epochs.")