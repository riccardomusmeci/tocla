from typing import Iterable
from torch.optim import Optimizer
from torch.optim import SGD, Adam, AdamW

FACTORY = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW
}

def optimizer(
    name: str,
    params: Iterable,
    **kwargs
) -> Optimizer:
    name = name.lower()
    assert name in FACTORY.keys(), f"Only {list(FACTORY.keys())} optimizers are supported. Change {name} to one of them."
    return FACTORY[name](params, **kwargs)
    