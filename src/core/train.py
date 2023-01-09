import os
from shutil import copy
from src.loss import Criterion
from src.io import load_config
from src.trainer import Trainer
from src.model import create_model
from src.optimizer import Optimizer
from src.transform import Transform
from src.data import create_dataloader
from src.lr_scheduler import LRScheduler
from src.utils import now, seed_everything
from src.utils.model_checkpoint import ModelCheckpoint

def train(args):
    
    seed_everything(args.seed)
    config = load_config(args.config)
    
    # creating output dir and copying config
    output_dir = os.path.join(args.output_dir, now())
    os.makedirs(output_dir)
    copy(args.config, output_dir)
    
    # setup classes for Trainer
    train_dataloader = create_dataloader(
        root_dir=args.data_dir,
        train=True,
        transform=Transform(train=True, **config["transform"]),
        **config["datamodule"]
    )
    val_dataloader = create_dataloader(
        root_dir=args.data_dir,
        train=False,
        transform=Transform(train=False, **config["transform"]),
        **config["datamodule"]
    )
    
    # setup model, criterion, optimizer and lr_scheduler
    model = create_model(
        **config["model"],
        num_classes=len(config['datamodule']['class_map'])
    )
    
    criterion = Criterion(**config["loss"])
    optimizer = Optimizer(params=model.parameters(), **config["optimizer"])
    lr_scheduler = LRScheduler(optimizer=optimizer, **config["lr_scheduler"])
    
    model_checkpoint = ModelCheckpoint(
        output_dir=output_dir,
        **config["checkpoint"]
    )
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        model_checkpoint=model_checkpoint,
        **config["trainer"]
    )
    
    trainer.fit()