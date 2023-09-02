import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..io import read_rgb, resize_rgb, save_image
from ..model import get_layer
from ..utils import get_device


def run_gradcam(
    model: nn.Module,
    data_loader: DataLoader,
    output_dir: Union[str, Path],
    layer: Optional[str] = None,
    preds_df: Optional[pd.DataFrame] = None,
) -> None:
    """Run GradCAM on model and save to output_dir.

    Args:
        model (nn.Module): model
        data_loader (DataLoader): data loader
        output_dir (Union[str, Path]): output directory
        layer (Optional[str], optional): layer to use for gradcam. Defaults to None.
        preds_df (Optional[pd.DataFrame], optional): predictions dataframe. Defaults to None.
    """
    target_layer = get_layer(model=model, name=layer)  # type: ignore
    if target_layer is None:
        print(f"> Found no layer with name {layer}. GradCAM will not be applied.")
        return

    print(f"> Saving gradcam to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    device = get_device()
    use_cuda = False if "cuda" not in device else True
    model.to("cpu" if not use_cuda else "cuda")

    gradcam = GradCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda)

    print("> Running GradCAM. It may take a while.")

    # force data loader to have batch size 1
    if preds_df is not None:
        assert (
            "label" in preds_df.columns and "filepath" in preds_df.columns
        ), "preds_df must have columns 'label' and 'filepath'"
        print("> Saving GradCAM images split in folders by class")
        for class_id in set(preds_df["label"].unique()):
            os.makedirs(os.path.join(output_dir, str(class_id)), exist_ok=True)

    for batch in tqdm(data_loader, total=len(data_loader)):
        x, filepaths = batch
        grayscale_cams = gradcam(input_tensor=x, targets=None)

        for filepath, grayscale_cam in zip(filepaths, grayscale_cams):
            # loading image and retreiving w, h
            img = read_rgb(filepath)
            grayscale_cam = resize_rgb(grayscale_cam, h=img.shape[0], w=img.shape[1])  # type: ignore
            output = show_cam_on_image(img=img / 255, mask=grayscale_cam, use_rgb=True)

            if preds_df is not None:
                pred = preds_df[preds_df["filepath"] == filepath]["label"].values[0]
                output_path = os.path.join(output_dir, str(pred), os.path.basename(filepath))
            else:
                output_path = os.path.join(output_dir, os.path.basename(filepath))

            save_image(image=output, output_path=output_path)

    print("GradCAM completed.")
