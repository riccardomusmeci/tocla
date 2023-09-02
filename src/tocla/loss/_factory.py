from typing import List

import numpy as np
import torch
from torch.nn.modules.loss import _Loss

from .utils import find_class_weights

FACTORY = {
    "xent": torch.nn.CrossEntropyLoss,
}

__all__ = ["list_criteria", "create_criterion"]


def list_criteria() -> List[str]:
    """List available criteria.

    Returns:
        list: list of available criteria
    """
    return list(FACTORY.keys())


def create_criterion(name: str = "xent", **kwargs) -> _Loss:  # type: ignore
    """Create a loss criterion.

    Args:
        name (str, optional): loss criterion name. Defaults to "xent".

    Returns:
        nn.Module: loss criterion
    """
    name = name.lower()
    assert (
        name in FACTORY.keys()
    ), f"Only {list(FACTORY.keys())} criterions are supported. Change {name} to one of them."

    if "label_smoothing" in kwargs and name == "xent":
        if kwargs["label_smoothing"] > 0:
            kwargs["label_smoothing"] = np.float32(kwargs["label_smoothing"])

    return FACTORY[name](**kwargs)
