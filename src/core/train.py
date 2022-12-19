import os
from shutil import copy
from src.utils import now
from src.model import Model
from src.loss import Criterion
import pytorch_lightning as pl
from src.io import load_config
from src.model import create_model
from src.optimizer import Optimizer
from src.transform import Transform
from src.datamodule import DataModule
from src.lr_scheduler import LRScheduler
from src.trainer import Callbacks, Logger

def train(args):
    
    pl.seed_everything(args.seed)
    config = load_config(args.config)
    
    # creating output dir and copying config
    output_dir = os.path.join(args.output_dir, now())
    os.makedirs(output_dir)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    copy(args.config, output_dir)
    
    # Datamodule
    datamodule = DataModule(
        root_dir=args.data_dir,
        train_transform=Transform(train=True, **config["transform"]),
        val_transform=Transform(train=False, **config["transform"]),
        **config["datamodule"],
    )
    
    # Creating model, criterion, optimizer, and lr_scheduler
    model = create_model(
        **config["model"],
        num_classes=len(datamodule.class_map)
    )
    
    criterion = Criterion(**config["loss"])
    optimizer = Optimizer(params=model.parameters(), **config["optimizer"])
    lr_scheduler = LRScheduler(optimizer=optimizer, **config["lr_scheduler"])
    
    # pl.Module to easily train the classifier
    model = Model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler
    )
    
    logger = Logger(output_dir=output_dir)
    callbacks = Callbacks(output_dir=checkpoint_dir, **config["callbacks"])
    
    if args.resume_from:
        print(f"Resuming training from {args.resume_from}")
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        default_root_dir=checkpoint_dir,
        resume_from_checkpoint=args.resume_from,
        **config["trainer"]
    )
    
    trainer.fit(model=model, datamodule=datamodule)
    

    
    