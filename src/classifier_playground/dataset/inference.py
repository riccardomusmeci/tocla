import os
import random
import numpy as np
from PIL import Image
from ..io import read_rgb
from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

# TODO: implement nested folders images loading
class InferenceDataset(Dataset):
    """Inference Dataset (a folder with images)

    Args:
        data_dir (str): root data dir (must be with train and val folders)
        transform (Callable, optional): set of data transformations. Defaults to None.

    Raises:
        FileNotFoundError: if something is found erroneous in the dataset
    """
    
    EXTENSIONS = (
        "jpg",
        "jpeg",
        "png",
        "ppm",
        "bmp",
        "pgm",
        "tif",
        "tiff",
        "webp",
    )
    
    def __init__(
        self,
        data_dir: Union[Path, str],
        transform: Callable = None,
        verbose: bool = True
    ) -> None:
        
        super().__init__()
        self.verbose = verbose
        # checking structure
        try:
            self._sanity_check(data_dir)
        except Exception as e:
            raise e
        
        self.data_dir = data_dir
        self.images = self._load_samples()
        self.transform = transform
    
    def _sanity_check(self, data_dir: str) -> None:
        """Check dataset structure

        Args:
            data_dir (str): data directory
            class_map (Dict[int, Union[str, List[str]]]): class map {e.g. {0: 'class_a', 1: ['class_b', 'class_c']}}
            
        Raises:
            FileNotFoundError: if the data folder is not right based on the structure in class_map
            FileExistsError: if some label does not have images in its folder
        """
        
        if not (os.path.exists(data_dir)):
                raise FileNotFoundError(f"Folder {data_dir} does not exist") 
        images = [f for f in os.listdir(data_dir) if f.split(".")[-1].lower() in self.EXTENSIONS]
        if len(images) == 0:
            raise FileExistsError(f"Folder {data_dir} does not have images.")
        
        if self.verbose:
            print(f"> [INFO] InferenceDataset sanity check OK")
    
    def _load_samples(self) -> List[str]:
        """Load samples from data directory

        Returns:
           List[str]: sample paths
        """

        paths = [
            os.path.join(self.data_dir, f) 
            for f in os.listdir(self.data_dir) 
            if f.split(".")[-1].lower() in self.EXTENSIONS
        ]
        return paths
        
    def __getitem__(self, index: int) -> Tuple[Image.Image, str]:
        """Return a tuple (image, path) given an index

        Args:
            index (int): sample index

        Returns:
            Tuple[Image.Image, str]: image, path
        """
        
        img_path = self.images[index]
        img = read_rgb(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_path
            
    def __len__(self) -> int:
        """Return number of samples in the dataset

        Returns:
            int: number of samples in the dataset
        """
        return len(self.images)
    