import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

from ..io import read_rgb


class InferenceDataset(Dataset):
    """Inference Dataset.

    Args:
        data_dir (str): data directory with images
        transform (Optional[Callable], optional): augmentation function. Defaults to None.
        engine (str, optional): image loading engine (pil/cv2). Defaults to "pil".
        verbose (bool, optional): verbose mode. Defaults to True.
    """

    EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".gif")

    def __init__(
        self,
        data_dir: Union[Path, str],
        transform: Optional[Callable] = None,
        engine: str = "pil",
        verbose: bool = True,
    ) -> None:
        assert os.path.exists(data_dir), f"Dataset directory {data_dir} does not exist."

        self.data_dir = data_dir
        self.verbose = verbose
        self.engine = engine

        # loading images and masks
        self.images = [
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if os.path.splitext(f)[-1] in self.EXTENSIONS
        ]
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """Return an image ready for inference.

        Args:
            index (int): index of element

        Returns:
            Tuple[torch.Tensor, str]: image, image_name
        """
        image_path = self.images[index]

        image = read_rgb(file_path=image_path, engine=self.engine)
        w, h = image.shape[:2]  # type: ignore
        os.path.splitext(os.path.basename(image_path))[0]

        if self.transform:
            try:
                image = self.transform(image=image)
            except Exception as e:
                print(f"Transform on image {image_path} not working. Reason: {e}")
                quit()

        image = torch.from_numpy(image.transpose(2, 0, 1))  # type: ignore

        return image, image_path

    def __len__(self) -> int:
        """Return number of images in the dataset.

        Returns:
            int: number of images in the dataset
        """
        return len(self.images)
