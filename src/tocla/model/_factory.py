import os
from pathlib import Path
from typing import Dict, Optional, Union

import timm
import torch
import torch.nn as nn


def create_model(
    model_name: str,
    pretrained: bool = True,
    num_classes: int = 0,
    ckpt_path: Optional[Union[str, Path]] = None,
    to_replace: Optional[str] = "model.",
    prefix_key: Optional[str] = None,
) -> nn.Module:
    """Create model utils function.

    Args:
        model_name (str): model name from timm library
        pretrained (bool, optional): whether to load pretrained weights. Defaults to True.
        num_classes (int, optional): num output classes. Defaults to 0.
        ckpt_path (str, Optional[Union[str, Path]]): path to local checkpoint to load. Defaults to None.
        to_replace (str, Optional[str]): initial string in state_dict to replace with "" to match keys. Defaults to None.
        prefix_key (str, Optional[str]): prefix key to add to state_dict keys. Defaults to None.

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
        try:
            state_dict = load_ckpt(ckpt_path=ckpt_path, prefix_key=prefix_key, to_replace=to_replace)
            model.load_state_dict(state_dict=state_dict)
            print(f"> Loaded state dict from {ckpt_path}")
        except Exception as e:
            print(f"> Found error while loading state dict from {ckpt_path}: {e}")
            print("> Trying with adapting state dict keys...")
            state_dict = load_and_adapt_ckpt(
                model_state_dict=model.state_dict(),
                ckpt_path=ckpt_path,
                prefix_key=prefix_key,
                to_replace=to_replace,
            )
            model.load_state_dict(state_dict=state_dict)
            print(f"> Loaded state dict from {ckpt_path}")

    return model


def load_ckpt(ckpt_path: Union[Path, str], prefix_key: Optional[str] = None, to_replace: Optional[str] = None) -> Dict:
    """Load checkpoint.

    Args:
        ckpt_path (Union[Path, str]): path to checkpoint.
        prefix_key (Optional[str], optional): prefix key for loading state dict. Defaults to None.
        to_replace (Optional[str], optional): string to replace in state dict keys. Defaults to None.

    Returns:
        Dict: state dict
    """

    assert os.path.exists(ckpt_path), f"{ckpt_path} does not exist"
    try:
        state_dict = torch.load(ckpt_path)
    except RuntimeError:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if prefix_key is not None:
        state_dict = {prefix_key + k: w for k, w in state_dict.items()}

    if to_replace is not None:
        state_dict = {k.replace(to_replace, ""): w for k, w in state_dict.items()}

    return state_dict


def load_and_adapt_ckpt(
    model_state_dict: Dict,
    ckpt_path: Union[Path, str],
    prefix_key: Optional[str] = None,
    to_replace: Optional[str] = None,
) -> Dict:
    """Load checkpoint and adapt model state dict to match keys.

    Args:
        model_state_dict (Dict): model state dict
        ckpt_path (Union[Path, str]): path to checkpoint.
        prefix_key (Optional[str], optional): prefix key for loading state dict. Defaults to None.
        to_replace (Optional[str], optional): string to replace in state dict keys. Defaults to None.

    Returns:
        Dict: state dict
    """
    print("> Running load and adapt checkpoint...")

    state_dict = load_ckpt(ckpt_path=ckpt_path, prefix_key=prefix_key, to_replace=to_replace)
    for k, w in model_state_dict.items():
        if k in state_dict:
            model_state_dict[k] = state_dict[k]
        else:
            print(f"\t- Layer {k} with shape {w.shape} not in checkpoint")
    return model_state_dict
