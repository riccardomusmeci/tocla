from typing import Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Sampler

from ..dataset import ImageFolderDataset


class ImbalancedSampler(Sampler):
    """Imbalanced Dataset sampler init.

    Args:
        dataset (ImageFolderDataset): dataset to sample
        indices (List[int], optional): indices to sample. Defaults to None.
        num_samples (int, optional): number of samples. Defaults to None.
    """

    def __init__(
        self, dataset: ImageFolderDataset, indices: Optional[List[int]] = None, num_samples: Optional[int] = None
    ) -> None:
        super().__init__(dataset)

        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        # distribution of classes in the dataset
        label_count = {}  # type: ignore
        for idx in self.indices:
            label = dataset.targets[idx]
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_count[dataset.targets[idx]] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self) -> Iterator:
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self) -> int:
        return self.num_samples
