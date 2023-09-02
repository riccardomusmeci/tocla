from typing import Dict, List, Optional, Union
from pathlib import Path

import os

from ..pl import ClassificationDataModule, ClassificationModelModule
from ..io import ToclaConfiguration
from ..transform import Transform
from ..model import create_model
from ..loss import create_criterion
from ..optimizer import create_optimizer
from ..lr_scheduler import create_lr_scheduler
from ..pl import create_callbacks
from ..utils import now

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


def train(
    config_path: Union[str, Path],
    train_data_dir: Union[str, Path],
    val_data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    resume_from: Optional[Union[str, Path]] = None,
    seed: int = 42,
) -> None:
    """Train classification model.

    Args:
        config_path (Union[str, Path]): path to configuration file
        train_data_dir (Union[str, Path]): path to data train directory
        val_data_dir (Union[str, Path]): path to data validation directory
        output_dir (Union[str, Path]): path to output directory
        resume_from (Optional[Union[str, Path]], optional): path to checkpoint to resume train from. Defaults to None.
        seed (int, optional): random seed (for reproducibility). Defaults to 42.
    """

    if resume_from is not None:
        output_dir = "/".join(resume_from.split("/")[:-2])  # type: ignore
    else:
        output_dir = os.path.join(output_dir, now())
        os.makedirs(output_dir, exist_ok=True)
    print("\n" + "*" * 40)
    print(f"Output directory: {output_dir}")
    print("\n" + "*" * 40)

    # Loading and saving configuration
    config = ToclaConfiguration.load(config_path=config_path)
    ToclaConfiguration.save(config=config, output_path=os.path.join(output_dir, "config.yaml"))

    # Setting up datamodule, model, callbacks, logger, and trainer
    datamodule = ClassificationDataModule(
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        train_transform=Transform(train=True, **config["transform"]),
        val_transform=Transform(train=False, **config["transform"]),
        **config["datamodule"],
    )

    num_classes = len(config["datamodule"]["class_map"])

    clf_model = create_model(
        model_name=config["model"]["model_name"],
        pretrained=config["model"]["pretrained"],
        num_classes=num_classes,
    )

    if config["loss"]["weight"] == "auto":
        config["loss"]["weight"] = datamodule.get_class_weights()
    criterion = create_criterion(**config["loss"])
    optimizer = create_optimizer(params=clf_model.parameters(), **config["optimizer"])
    lr_scheduler = create_lr_scheduler(optimizer=optimizer, **config["lr_scheduler"])

    model = ClassificationModelModule(
        model=clf_model,
        num_classes=num_classes,
        loss=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        bn_to_zero_on_sam=config["model"]["bn_to_zero_on_sam"],
        **config["metrics"],
    )

    callbacks = create_callbacks(output_dir=output_dir, **config["callbacks"])
    logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, config["logger"]["save_dir"]),
        name=config["logger"]["name"],
    )

    trainer = Trainer(logger=logger, callbacks=callbacks, **config["trainer"])

    # Training
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_from)  # type: ignore
