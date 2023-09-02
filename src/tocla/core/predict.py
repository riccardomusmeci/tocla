import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset import InferenceDataset
from ..gradcam import run_gradcam
from ..io import ToclaConfiguration
from ..model import create_model, get_layer
from ..transform import Transform
from ..utils import get_device


def predict(
    ckpt_path: Union[str, Path],
    config_path: Union[str, Path],
    images_dir: Union[str, Path],
    output_dir: Union[str, Path],
    apply_gradcam: bool,
    layer: str,
    gradcam_with_preds: bool = True,
    save_images: bool = True,
) -> None:
    """Predict from classification model.

    Args:
        ckpt_path (Union[str, Path]): path to checkpoint
        config_path (Union[str, Path]): path to configuration file
        images_dir (Union[str, Path]): path to images directory
        output_dir (Union[str, Path]): path to output directory
        gradcam (bool): whether to save gradcam
        layer (str): layer to use for gradcam
        gradcam_with_preds (bool): whether to save gradcam images split in folders by prediction class. Defaults to True.
        save_images (bool, optional): whether to save images. Defaults to True.
    """

    csv_path = os.path.join(output_dir, "predictions.csv")

    config = ToclaConfiguration.load(config_path=config_path)
    class_map = config["datamodule"]["class_map"]

    if save_images:
        print(f"> Saving images to {output_dir}/images split in folders by class ({list(class_map.keys())})")
        for class_id in class_map.keys():
            os.makedirs(os.path.join(output_dir, "images", str(class_id)), exist_ok=True)

    if apply_gradcam:
        print(f"> Saving gradcam to {output_dir}/gradcam")
        os.makedirs(os.path.join(output_dir, "gradcam"), exist_ok=True)

    dataset = InferenceDataset(
        data_dir=images_dir,
        transform=Transform(train=False, **config["transform"]),
        engine=config["datamodule"]["engine"],
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config["datamodule"]["batch_size"],
        num_workers=config["datamodule"]["num_workers"],
        shuffle=False,
        drop_last=False,
    )

    device = get_device()
    model = create_model(
        model_name=config["model"]["model_name"],
        num_classes=len(class_map),
        ckpt_path=ckpt_path,
    )
    model.to(device)
    model.eval()

    images, predictions = [], []

    print("> Running predict..")
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            x, file_paths = batch
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            # save images and predictions
            images.extend(file_paths)
            predictions.extend(list(preds.cpu().numpy()))

    # save predictions in csv
    print(f"> Saving predictions to {csv_path}")
    preds_df = pd.DataFrame({"filepath": images, "label": predictions})
    preds_df.to_csv(csv_path, index=False)

    # save images
    if save_images:
        print(f"> Saving images to {output_dir} split in folders by class ({class_map.keys()})")
        for image, label in tqdm(zip(images, predictions), total=len(images)):
            image_name = os.path.basename(image)
            output_path = os.path.join(output_dir, "images", str(label), image_name)
            shutil.copy(image, output_path)

    # apply gradcam
    if apply_gradcam:
        run_gradcam(
            model=model,
            data_loader=data_loader,
            output_dir=os.path.join(output_dir, "gradcam"),
            layer=layer,
            preds_df=preds_df if gradcam_with_preds else None,
        )
