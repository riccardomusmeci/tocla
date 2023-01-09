from typing import Iterable
from torch.optim import Optimizer
from src.optimizer.sam import SAM
from torch.optim import SGD, Adam, AdamW

FACTORY = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
    "sam": SAM
}

def optimizer(
    name: str,
    params: Iterable,
    **kwargs
) -> Optimizer:
    name = name.lower()
    assert name in FACTORY.keys(), f"Only {list(FACTORY.keys())} optimizers are supported. Change {name} to one of them."
    
    if name == "sam":
        print(f"> Training with SAM optimizer, it'll take twice the training time.")
        kwargs["base_optimizer"] = FACTORY[kwargs["base_optimizer"]]
    
    return FACTORY[name](params, **kwargs)
    