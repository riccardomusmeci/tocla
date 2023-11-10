import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset


def find_class_weights(dataset: Dataset, class_weight: str = "balanced") -> np.array:  # type: ignore
    """Find class weights based on dataset samples.

    Args:
        dataset (Dataset): dataset
        class_weight (str, optional): class weight strategy. Defaults to "balanced".

    Returns:
        np.array: class weights
    """
    y = dataset.targets  # type: ignore
    return compute_class_weight(class_weight=class_weight, classes=np.unique(y), y=y)
