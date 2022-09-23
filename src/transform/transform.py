import PIL
import numpy as np
from PIL import Image
import albumentations as A
from src.utils import to_tensor
from typing import Union, Tuple

class Transform:
    
    def __init__(
        self,
        train: bool,
        img_size: Union[int, list, tuple], 
        mean: list = [0.485, 0.456, 0.406], 
        std: list = [0.229, 0.224, 0.225],
        crop_resize_p: float = 0.5,
        brightness: float = 0.4, 
        contrast: float = 0.4, 
        saturation: float = 0.2, 
        hue: float = 0.1,
        color_jitter_p: float = .5,
        grayscale_p: float = 0.2,
        h_flip_p: float = .5,
        kernel: tuple = (5, 5),
        sigma: tuple = (.1, 2),
        solarization_p: float = 0.2,
        solarize_t: int = 170,
    ):
        """DINO Transform

        Args:
            train (bool): train/val transformation
            img_size (Union[int, list, tuple]): image size. 
            mean (list, optional): normalization mean. Defaults to [0.485, 0.456, 0.406].
            std (list, optional): normalization std. Defaults to [0.229, 0.224, 0.225].
            crop_resize_p (float, optional): crop and resize prob. Defaults to 0.5.
            brightness (float, optional): color jitter brightness val. Defaults to 0.4.
            contrast (float, optional): color jitter contrast val. Defaults to 0.4.
            saturation (float, optional): color jitter saturation val. Defaults to 0.2.
            hue (float, optional): color jitter hue val. Defaults to 0.1.
            color_jitter_p (float, optional): color jitter prob. Defaults to 0.8.
            grayscale_p (float, optional): grayscale prob. Defaults to 0.1.
            h_flip_p (float, optional): horizontal flip prob. Defaults to 0.5.
            kernel (tuple, optional): gaussian blur kernel. Defaults to (3, 3).
            sigma (tupla, optional): gaussian blur std. Defaults to (.1, 2).
            gaussian_blur_p (float, optional): gaussian blur prob. Defaults to 0.1.
            solarization_p (float, optional): solarization prob. Defaults to 0.2.
            solarize_t (int, optional): solarization threshold. Defaults to 170.
        """
        
        if isinstance(img_size, tuple) or isinstance(img_size, list):
            height, width = img_size[0], img_size[1]
        else:
            height, width = img_size, img_size
            
        if train:
            self.transform = A.Compose([
                A.RandomResizedCrop(
                    height=height, 
                    width=width, 
                    p=crop_resize_p, 
                    interpolation=Image.BICUBIC
                ),
                # Flip and ColorJitter
                A.HorizontalFlip(p=h_flip_p),
                A.ColorJitter(
                    brightness=brightness, 
                    contrast=contrast, 
                    saturation=saturation, 
                    hue=hue, 
                    p=color_jitter_p
                ),
                A.ToGray(p=grayscale_p),
                # Gaussion Blur + Solarization
                A.GaussianBlur(
                    blur_limit=kernel, 
                    sigma_limit=sigma, 
                    p=0.1
                ), 
                A.Solarize(
                    threshold=solarize_t, 
                    p=solarization_p
                ),
                A.Resize(
                    height=height, 
                    width=width, 
                    interpolation=Image.BICUBIC
                ),
                # Normalization
                A.Normalize(mean=mean, std=std)            
            ])
        else:
            self.transform = A.Compose([
                A.Resize(
                    height=height, 
                    width=width, 
                    interpolation=Image.BICUBIC
                ),
                A.Normalize(mean=mean, std=std),
            ])
            
    def __call__(
        self, 
        img: Union[np.array, PIL.Image.Image]
    ) -> Tuple[np.array, np.array, np.array]:
        """Apply augmentations

        Args:
            img (Union[np.array, PIL.Image.Image]): input image

        Returns:
            Tuple[np.array, np.array, np.array]: vanilla img (resize + normalize), view 1, view 2
        """
        
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        
        img = self.transform(image=img)['image']
        img = to_tensor(img)
        
        return img