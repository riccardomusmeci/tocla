import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset import ImageFolderDataset
from ..io import ToclaConfiguration
from ..model import create_model
from ..transform import Transform
from ..utils import get_device


def test(
    ckpt_path: Union[str, Path],
    config_path: Union[str, Path],
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    to_replace: str = "model.",
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

    os.path.join(output_dir, "predictions.csv")

    config = ToclaConfiguration.load(config_path=config_path)
    class_map = config["datamodule"]["class_map"]
    num_classes = len(class_map)

    dataset = ImageFolderDataset(
        root_dir=data_dir,
        transform=Transform(train=False, **config["transform"]),
        class_map=class_map,
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
        num_classes=num_classes,
        ckpt_path=ckpt_path,
    )

    model.to(device)
    model.eval()

    y_pred = []
    y_true = []
    print("> Running test..")
    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader)):
            x, target = (el.to(device) for el in batch)
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            y_pred.extend(list(preds.cpu().numpy()))
            y_true.extend(list(target.cpu().numpy()))

    print("\n> Metrics report:")
    print(f"\t- accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\t- macro metrics:")
    print(f"\t\t- precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"\t\t- recall: {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"\t\t- f1: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("\t- micro metrics:")
    print(f"\t\t- precision: {precision_score(y_true, y_pred, average='micro'):.4f}")
    print(f"\t\t- recall: {recall_score(y_true, y_pred, average='micro'):.4f}")
    print(f"\t\t- f1: {f1_score(y_true, y_pred, average='micro'):.4f}")
    print("\t- weighted metrics:")
    print(f"\t\t- precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"\t\t- recall: {recall_score(y_true, y_pred, average='weighted'):.4f}")
    print(f"\t\t- f1: {f1_score(y_true, y_pred, average='weighted'):.4f}")

    print("\n> Classification Report:")
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=[str(i) for i in range(num_classes)]))

    # # save predictions in csv
    report = {"image_path": dataset.images, "y_pred": y_pred, "y_true": y_true}
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "test_report.csv")
    pd.DataFrame(report).to_csv(report_path, index=False)

    with open(os.path.join(output_dir, "test_metrics.txt"), "w") as f:
        f.write("Metrics report:\n")
        f.write(f"\t- accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write("\t- macro metrics:\n")
        f.write(f"\t\t- precision: {precision_score(y_true, y_pred, average='macro'):.4f}\n")
        f.write(f"\t\t- recall: {recall_score(y_true, y_pred, average='macro'):.4f}\n")
        f.write(f"\t\t- f1: {f1_score(y_true, y_pred, average='macro'):.4f}\n")
        f.write("\t- micro metrics:\n")
        f.write(f"\t\t- precision: {precision_score(y_true, y_pred, average='micro'):.4f}\n")
        f.write(f"\t\t- recall: {recall_score(y_true, y_pred, average='micro'):.4f}\n")
        f.write(f"\t\t- f1: {f1_score(y_true, y_pred, average='micro'):.4f}\n")
        f.write("\t- weighted metrics:\n")
        f.write(f"\t\t- precision: {precision_score(y_true, y_pred, average='weighted'):.4f}\n")
        f.write(f"\t\t- recall: {recall_score(y_true, y_pred, average='weighted'):.4f}\n")
        f.write(f"\t\t- f1: {f1_score(y_true, y_pred, average='weighted'):.4f}\n")

        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true=y_true, y_pred=y_pred, target_names=[str(i) for i in range(num_classes)]))
