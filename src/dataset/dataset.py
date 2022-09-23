import os
import numpy as np
from PIL import Image
from src.io import read_rgb
from torch.utils.data import Dataset
from typing import Callable, Dict, List, Tuple, Union

class Dataset(Dataset):
    
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
        root_dir: str,
        train: bool,
        class_map: Dict[int, Union[str, List[str]]],
        max_samples_per_class: int = None,
        transform: Callable = None,
    ) -> None:
        """Image Classification Dataset init (image folder dataset)

        Args:
            root_dir (str): root data dir
            train (bool): train mode
            class_map (Dict[int, Union[str, List[str]]], optional): class map {e.g. {0: 'class_a', 1: ['class_b', 'class_c']}}
            max_samples_per_class (int, optional): max number of samples for each class in the dataset. Defaults to None.
            transform (Callable, optional): set of data transformations. Defaults to None.

        Raises:
            e: if something is found erroneous in the dataset
        """
        
        super().__init__()
        
        assert isinstance(class_map, dict), "class_map must be a Python dict"
        
        data_dir = os.path.join(root_dir, "train" if train else "val")
        # checking structure
        try:
            self._sanity_check(
                data_dir=data_dir, 
                class_map=class_map
            )
        except Exception as e:
            raise e
        self.data_dir = data_dir
        self.class_map = class_map
        self.images, self.targets = self._load_samples(max_samples_per_class=max_samples_per_class)
        self.transform = transform
        
    def _sanity_check(
        self,
        data_dir: str,
        class_map: Dict[int, Union[str, List[str]]]
    ):
        """Checks dataset structure

        Args:
            data_dir (str): data directory
            class_map (Dict[int, Union[str, List[str]]]): class map {e.g. {0: 'class_a', 1: ['class_b', 'class_c']}}
            
        Raises:
            FileNotFoundError: if the data folder is not right based on the structure in class_map
            FileExistsError: if some label does not have images in its folder

        """
        for k, labels in class_map.items():
            if not isinstance(labels, list):
                labels = [labels]
            for l in labels:
                label_dir =  os.path.join(data_dir, l)
                if not (os.path.exists(label_dir)):
                    raise FileNotFoundError(f"Folder {label_dir} does not exist") 
                if len(os.listdir(label_dir))==0:
                    raise FileExistsError(f"Folder {label_dir} is empty.")
                
        print(f"> Dataset sanity check OK")
    
    def _load_samples(
        self, 
        max_samples_per_class: int
    ) -> Tuple[List[str], List[int]]:
        """loads image paths + targets for the dataset

        Returns:
            Tuple[List[str], List[int]]: image paths, targets
        """
        paths = []
        targets = []
        for c, labels in self.class_map.items():
            if isinstance(labels, str):
                labels = [labels]
            c_images, c_targets = [], []
            for label in labels:
                label_dir = os.path.join(self.data_dir, label)
                c_images += [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.split(".")[-1].lower() in self.EXTENSIONS]
            if max_samples_per_class is not None:
                if len(c_images) > max_samples_per_class:
                    print(f"> Images will be limited from {len(c_images)} to {max_samples_per_class} for label {c} ({self.class_map[c]})")
                    c_images = c_images[:max_samples_per_class]
            c_targets += [c] * len(c_images)
            
            paths += c_images
            targets += c_targets
        
        return paths, targets
    
    def stats(self):
        """prints stats of the dataset
        """
        unique, counts = np.unique(self.targets, return_counts=True)
        num_samples = len(self.targets)
        print(f" ----------- Dataset Stats -----------")
        for k in range(len(unique)):
            classes = self.class_map[k]
            if isinstance(classes, str):
                classes = [classes]
            print(f"> {classes} : {counts[k]}/{num_samples} -> {100 * counts[k] / num_samples:.3f}%")
        print(f" -------------------------------------")
    
    def __getitem__(self, index) -> Tuple:
        
        img_path = self.images[index]
        target = self.targets[index]
        
        img = read_rgb(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
            
    def __len__(self):
        return len(self.images)
    
