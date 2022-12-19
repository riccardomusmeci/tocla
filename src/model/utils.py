import os
import timm
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

def create_model(
    model_name: str,
    pretrained: bool = True,
    num_classes: int = 0,
    checkpoint_path: str = None,
    to_replace: str = "model."
) -> nn.Module:
    """create model utils function

    Args:
        model_name (str): model name from timm library
        pretrained (bool, optional): whether to load pretrained weights. Defaults to True.
        num_classes (int, optional): num output classes. Defaults to 0.
        checkpoint_path (str, optional): path to local checkpoint to load. Defaults to None.
        to_replace (str, optional): initial string in state_dict to replace with "" to match keys. Defaults to "model.".
        
    Returns:
        nn.Module: model
    """
    
    model: nn.Module = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    print(f"> Created {model_name} with {'no ' if not pretrained else ''}pretrained weights. Num classes set to {num_classes}.")
    
    if checkpoint_path is not None:
        assert os.path.exists(checkpoint_path), f"{checkpoint_path} does not exist"
        try:
            state_dict = torch.load(checkpoint_path)
        except RuntimeError:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            
        if 'state_dict' in state_dict:
            state_dict = state_dict["state_dict"]
        
        state_dict = {
            k.replace(to_replace, ""): w for k, w in state_dict.items() if "criterion" not in k
        }
        model.load_state_dict(state_dict=state_dict)
        print(f"> Loaded state dict from {checkpoint_path}")
    
    return model

# SAM - disable and enable BN
def disable_running_stats(model: nn.Module):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model: nn.Module):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)