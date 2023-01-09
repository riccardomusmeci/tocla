from typing import Dict, List

class MetricsMonitor:
    
    def __init__(
        self,
        train_metrics: List[str] = ["loss_train", "loss_val", "lr", "f1_val"],
        val_metrics: List[str] = ["acc_val", "f1_val", "prec_val", "recall_val", "calerr_val", "loss_val"],
    ) -> None:
        """metrics monitor class
        """
        # Training and Validation metrics
        self.train_metrics = {
            k: None for k in train_metrics
        }
        
        self.val_metrics = {
            k: None for k in val_metrics
        }
        # metrics in common between val and train
        self.common_metrics = set(train_metrics).intersection(set((val_metrics)))
        
        # Training and validation time info
        self.train_times = {
            "elapsed": "00:00",
            "total_estimate": "00:00"
        }
        
        self.val_times = {
            "elapsed": "00:00",
            "total_estimate": "00:00"
        }
        
    def _update_train(
        self,
        loss: float,
        lr: float,
        elapsed: float,
        total_estimate: float
    ):
        """updates training metrics

        Args:
            loss (float): training loss
            lr (float): learning rate
            elapsed (float): training step elapsed time
            total_estimate (float): training epoch total time estimate
        """
        # check if first time updating training metrics
        
        if self.train_metrics["loss_train"] is None:
            self.train_metrics["loss_train"] = loss
        else:
            self.train_metrics["loss_train"] += loss
            self.train_metrics["loss_train"] 
            
        # update common metrics metrics
        self.train_metrics["f1_val"] = self.val_metrics["f1_val"]
        self.train_metrics["loss_val"] = self.val_metrics["loss_val"]
        self.train_metrics["lr"] = lr
        
        # udpate time
        self.train_times["elapsed"] = elapsed
        self.train_times["total_estimate"] = total_estimate
        
    def _update_val(
        self,
        loss: float = None,
        elapsed: float = None,
        total_estimate: float = None,
        metrics_vals: Dict[str, float] = None,
    ):
        """updates validation metrics

        Args:
            loss (float, optional): validation loss. Defaults to None.
            elapsed (float, optional): validation step elapsed time. Defaults to None
            total_estimate (float, optional): validation epoch total time estimate. Defaults to None
            metrics_vals (Dict[str, float], optional): metrics from Trainer. Defaults to None.
        """
        
        if loss is not None:
            if self.val_metrics["loss_val"] is None:
                self.val_metrics["loss_val"] = loss
            else:
                self.val_metrics["loss_val"] += loss
                self.val_metrics["loss_val"] /= 2
        
        if metrics_vals is not None:
            for k, v in metrics_vals.items():
                self.val_metrics[f"{k}_val"] = v
        
        if elapsed is not None: self.val_times["elapsed"] = elapsed
        if total_estimate is not None: self.val_times["total_estimate"] = total_estimate
        
    def update(
        self,
        mode: str,
        **kwargs
    ):
        """updates metrics monitor

        Args:
            mode (str): either train or val
        """
        assert mode in ["train", "val"], f"Only train or val mode supported, not {mode}."
        if mode == "train":
            self._update_train(**kwargs)
        if mode == "val":
            self._update_val(**kwargs)
            
    