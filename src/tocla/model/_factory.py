import os
from pathlib import Path
from typing import Optional, Union

import timm
import torch
import torch.nn as nn


def create_model(
    model_name: str,
    pretrained: bool = True,
    num_classes: int = 0,
    ckpt_path: Optional[Union[str, Path]] = None,
    to_replace: str = "model.",
) -> nn.Module:
    """Create model utils function.

    Args:
        model_name (str): model name from timm library
        pretrained (bool, optional): whether to load pretrained weights. Defaults to True.
        num_classes (int, optional): num output classes. Defaults to 0.
        ckpt_path (str, Optional[Union[str, Path]]): path to local checkpoint to load. Defaults to None.
        to_replace (str, optional): initial string in state_dict to replace with "" to match keys. Defaults to "model.".

    Returns:
        nn.Module: model
    """

    model = timm.create_model(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    print(
        f"> Created {model_name} with {'no ' if not pretrained else ''}pretrained weights. Num classes set to {num_classes}."
    )

    if ckpt_path is not None:
        assert os.path.exists(ckpt_path), f"{ckpt_path} does not exist"
        try:
            state_dict = torch.load(ckpt_path)
        except RuntimeError:
            state_dict = torch.load(ckpt_path, map_location="cpu")

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        state_dict = {k.replace(to_replace, ""): w for k, w in state_dict.items() if "loss" not in k}

        model.load_state_dict(state_dict=state_dict)
        print(f"> Loaded state dict from {ckpt_path}")

    return model
