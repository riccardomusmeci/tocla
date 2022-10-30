import os
import timm
import torch
import torch.nn as nn

def create_model(
    model_name: str,
    pretrained: bool = True,
    num_classes: int = 0,
    checkpoint_path: str = None
) -> nn.Module:
    """create model utils function

    Args:
        model_name (str): model name from timm library
        pretrained (bool, optional): whether to load pretrained weights. Defaults to True.
        num_classes (int, optional): num output classes. Defaults to 0.
        checkpoint_path (str, optional): path to local checkpoint to load. Defaults to None.

    Returns:
        nn.Module: model
    """
    
    model: nn.Module = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    print(f"> Created {model_name} with {'no ' if not pretrained else ''}pretrained weights. Num classes set to {num_classes}.")
    
    if checkpoint_path is not None:
        assert os.path.exists(checkpoint_path), f"{checkpoint_path} does not exist"
        state_dict = torch.load(checkpoint_path)
        if 'state_dict' in state_dict:
            state_dict = state_dict["state_dict"]
        
        state_dict = {
            k.replace("model.", ""): w for k, w in state_dict.items() if "criterion" not in k
        }
        model.load_state_dict(state_dict=state_dict)
        print(f"> Loaded state dict from {checkpoint_path}")
    
    return model