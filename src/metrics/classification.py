import torch
from typing import List, Dict
from .loss import LossMetric
from torchmetrics import Accuracy, F1Score, Precision, Recall, CalibrationError

class ClassificationMetrics:
    
    FACTORY = {
        "f1": F1Score,
        "accuracy": Accuracy,
        "precision": Precision,
        "recall": Recall,
        "calibration_error": CalibrationError
    }
    
    def __init__(
        self,
        train_loss: LossMetric,
        val_loss: LossMetric,
        metrics: List[str] = ["f1", "accuracy", "precision", "recall", "calibration_error"],
        device: str = "mps"
    ) -> None:
        """Class to monitor classification metrics

        Args:
            metrics (List[str], optional): list of metrics to monitor. Defaults to ["f1", "accuracy", "precision", "recall", "calibration_error"].
            device (str, optional): device. Defaults to "mps".
        """
        
        self._available_metrics = None
        # checking metrics name
        for m in metrics:
            if m not in self.available_metrics:
                print(f"[WARNING] Metric {m} not found among {self.available_metrics}. It won't be used.")
        
        self.metrics = {
            "".join([s[:3] for s in m.split("_")]): self.FACTORY[m]().to(device) for m in metrics if m in self.available_metrics
        }
        self.train_loss = train_loss
        self.val_loss = val_loss
        
        self._status = {
            m: None for m in self.metrics
        }
        self._empty = True
                
    @property
    def available_metrics(self):
        if self._available_metrics is None:
            self._available_metrics = list(self.FACTORY.keys())
        return self._available_metrics

    def update(
        self, 
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> None:
        """updates metrics

        Args:
            logits (torch.Tensor): model logits (softmax applied here)
            target (torch.Tensor): batch target
        """
        self._empty = False
        for m in self.metrics:
            self.metrics[m].update(
                preds=torch.softmax(logits, dim=1),
                target=target
            )
                
    def reset(self):
        """reset metrics
        """
        for m in self.metrics:
            self.metrics[m].reset()
        self._empty = True
    
    @property
    def status(self) -> Dict[str, float]:
        """returns metrics status and reset metrics 

        Returns:
            Dict[str, float]: dict with metric name and metric value
        """
        
        if not self._empty:
            self._status = { m: self.metrics[m].compute().item() for m in self. metrics }
            self.reset()
        
        self._status["loss_train"] = self.train_loss.avg_loss
        self._status["loss_val"] = self.val_loss.avg_loss

        return self._status
        
            
    
    

            
    
            
        
        

    