from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


# SAM - disable and enable BN running stats
def disable_running_stats(model: nn.Module) -> None:
    def _disable(module):  # type: ignore
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model: nn.Module) -> None:
    def _enable(module):  # type: ignore
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def get_layer(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Get layer from model by name.

    Args:
        model (nn.Module): model
        name (str): layer name

    Returns:
        Optional[nn.Module]: layer
    """
    for layer_name, layer in model.named_modules():
        if layer_name == name:
            print(f"Found layer {layer_name}")
            return layer
    print(f"No layer found with name {name}")
    return None
