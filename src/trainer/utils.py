import os
from typing import List
import pytorch_lightning
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, ModelSummary

def callbacks(
    output_dir: str,
    filename: str = "epoch={epoch}-step={step}-val_loss={loss/val:.3f}",
    monitor: str = "loss/val",
    mode: str = "min",
    save_top_k: int = 20,
    patience: int = 20
) -> List[pytorch_lightning.Callback]:
    """returns a list of pytorch-lightning callbacks

    Args:
        output_dir (str): where to save callbacks' output
        filename (str, optional): ModelCheckpoint filename. Defaults to "epoch={epoch}-step={step}-val_loss={loss/val:.3f}".
        monitor (str, optional): ModelCheckpoint/EarlyStopping metric to monitor. Defaults to "loss/val".
        mode (str, optional):  ModelCheckpoint/EarlyStopping mode to monitor metric. Defaults to "min".
        save_top_k (int, optional): ModelCheckpoint number of checkpoints to save. Defaults to 20.
        patience (int, optional): EarlyStopping number of epochs patience. Defaults to 20.

    Returns:
        List[pytorch_lightning.Callback]: list of pytorch-lightning Callback
    """
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=output_dir, 
            filename=filename,
            monitor=monitor,
            verbose=True,
            mode=mode,
            save_top_k=save_top_k,
            auto_insert_metric_name=False
        )
    )
    
    callbacks.append(
         LearningRateMonitor(
             logging_interval="epoch",
             log_momentum=True
        )
    )
    
    callbacks.append(ModelSummary(max_depth=1))
    
    callbacks.append(
        EarlyStopping(
            monitor=monitor,
            min_delta=0.0,
            patience=patience,
            verbose=False,
            mode=mode,
            check_finite=True,
            stopping_threshold=None,
            divergence_threshold=None,
            check_on_train_epoch_end=None
        )
    )
    
    return callbacks

def logger(output_dir: str) -> LightningLoggerBase:
    """returns a logger

    Args:
        output_dir (str): output dir
        name (str): name
        version (str): versione

    Returns:
        pytorch_lightning.loggers.LightningLoggerBase: logger
    """
    return TensorBoardLogger(
        save_dir=os.path.join(output_dir, "tensorboard")
    )