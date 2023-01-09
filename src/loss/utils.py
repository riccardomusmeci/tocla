import numpy as np
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight


def find_class_weights(
    dataset: Dataset,    
) -> np.array:
    """finds class weights based on dataset samples

    Args:
        dataset (Dataset): dataset

    Returns:
        np.array: class weights
    """
    y = dataset.targets
    return compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y), 
        y=y
    )
    