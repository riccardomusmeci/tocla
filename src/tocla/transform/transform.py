from typing import List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import PIL
from PIL import Image


class Transform:
    """Data Augmentation for Image Classification.

    Args:
        train (bool): train mode
        input_size (Union[int, list, tuple]): input size
        interpolation (int, optional): interpolation. Defaults to 3.
        mean (Tuple[float, float, float], optional): normalization mean. Defaults to (0.485, 0.456, 0.406).
        std (Tuple[float, float, float], optional): normalizaton std. Defaults to (0.229, 0.224, 0.225).
        horizontal_flip_prob (float, optional): horizontal flip probability. Defaults to 0.1.
        vertical_flip_prob (float, optional): vertical flip probability. Defaults to 0.1.
        border_mode (int, optional): border mode. Defaults to 0.
        ssr_prob (float, optional): random shift, scale and rotate probability (0 to disable). Defaults to 0.1.
        ssr_rotate_limit (float, optional): maximum rotation angle. Defaults to 20.
        ssr_scale_limit (float, optional): maximum scale factor. Defaults to 0.5.
        ssr_shift_limit (float, optional): maximum shift factor. Defaults to 0.1.
        ssr_value (float, optional): padding value. Defaults to 0.
        gauss_prob (float, optional): gaussian noise probability. Defaults to 0.1.
        gauss_var_limit (Tuple[float, float], optional): gaussian noise variance limit. Defaults to (10.0, 50.0).
        gauss_mean (float, optional): gaussian noise mean. Defaults to 0.0.
        random_crop_prob (float, optional): random crop prob. Defaults to 0.1.
        first_oneof_prob (float, optional): first oneof prob. Defaults to 0.66.
        sharpen_prob (float, optional): sharpen prob. Defaults to 0.5.
        sharpen_alpha (Tuple[float, float], optional): sharpen alpha. Defaults to (0.1, 0.5).
        sharpen_lightness (Tuple[float, float], optional): sharpen lightness. Defaults to (0.5, 1.0).
        blur_prob (float, optional): blur prob. Defaults to 0.25.
        blur_limit (float, optional): blur limit. Defaults to 3.
        motion_blur_prob (float, optional): motion blur prob. Defaults to 0.25.
        motion_blur_limit (float, optional): motion blur limit. Defaults to 3.
        second_oneof_prob (float, optional): second oneof prob. Defaults to .5.
        random_brightness_contrast_prob (float, optional): random brightness and contrast prob. Defaults to 0.5.
        brightness_limit (float, optional): brightness limit. Defaults to 0.2.
        contrast_limit (float, optional): contrast limit. Defaults to 0.2.
        hsv_prob (float, optional): hue saturation value prob. Defaults to 0.5.
        hue_shift_limit (float, optional): hue shift limit . Defaults to 20.
        sat_shift_limit (float, optional): saturation shift limit. Defaults to 30.
        val_shift_limit (float, optional): value shift limit. Defaults to 20.
    """

    def __init__(
        self,
        train: bool,
        input_size: Union[int, list, tuple],
        interpolation: int = 3,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        horizontal_flip_prob: float = 0.1,
        vertical_flip_prob: float = 0.1,
        border_mode: int = 0,
        ssr_prob: float = 0.1,
        ssr_rotate_limit: float = 20,
        ssr_scale_limit: float = 0.5,
        ssr_shift_limit: float = 0.1,
        ssr_value: float = 0,
        gauss_prob: float = 0.1,
        gauss_var_limit: Tuple[float, float] = (10.0, 50.0),
        gauss_mean: float = 0.0,
        random_crop_prob: float = 0.1,
        first_oneof_prob: float = 0.66,
        sharpen_prob: float = 0.5,
        sharpen_alpha: Tuple[float, float] = (0.1, 0.5),
        sharpen_lightness: Tuple[float, float] = (0.5, 1.0),
        blur_prob: float = 0.25,
        blur_limit: float = 3,
        motion_blur_prob: float = 0.25,
        motion_blur_limit: float = 3,
        second_oneof_prob: float = 0.5,
        random_brightness_contrast_prob: float = 0.5,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        hsv_prob: float = 0.5,
        hue_shift_limit: float = 20,
        sat_shift_limit: float = 30,
        val_shift_limit: float = 20,
    ) -> None:
        if isinstance(input_size, tuple) or isinstance(input_size, list):
            height = input_size[0]
            width = input_size[1]
        else:
            height = input_size
            width = input_size

        if train:
            self.transform = A.Compose(
                [
                    A.Resize(height=height, width=width, interpolation=interpolation, always_apply=True),
                    A.VerticalFlip(p=vertical_flip_prob),
                    A.HorizontalFlip(p=horizontal_flip_prob),
                    A.ShiftScaleRotate(
                        border_mode=border_mode,
                        p=ssr_prob,
                        rotate_limit=ssr_rotate_limit,
                        scale_limit=ssr_scale_limit,
                        shift_limit=ssr_shift_limit,
                        value=ssr_value,
                    ),
                    A.GaussNoise(p=gauss_prob, var_limit=gauss_var_limit, mean=gauss_mean),
                    A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=border_mode),
                    A.RandomCrop(height=height, width=width, p=random_crop_prob),
                    A.OneOf(
                        [
                            A.Sharpen(alpha=sharpen_alpha, lightness=sharpen_lightness, p=sharpen_prob),
                            A.Blur(blur_limit=blur_limit, p=blur_prob),
                            A.MotionBlur(blur_limit=motion_blur_limit, p=motion_blur_prob),
                        ],
                        p=first_oneof_prob,
                    ),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=brightness_limit,
                                contrast_limit=contrast_limit,
                                p=random_brightness_contrast_prob,
                            ),
                            A.HueSaturationValue(
                                hue_shift_limit=hue_shift_limit,
                                sat_shift_limit=sat_shift_limit,
                                val_shift_limit=val_shift_limit,
                                p=hsv_prob,
                            ),
                        ],
                        p=second_oneof_prob,
                    ),
                    A.Resize(height=height, width=width, interpolation=interpolation, always_apply=True),
                    A.Normalize(mean=mean, std=std, always_apply=True),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=height, width=width),
                    A.Normalize(mean=mean, std=std, always_apply=True),
                ]
            )

    def __call__(self, image: Union[np.array, PIL.Image.Image]) -> np.array:  # type: ignore
        """Apply augmentations.

        Args:
            img (Union[np.array, PIL.Image.Image]): input image

        Returns:
            np.array: transformed image
        """

        if isinstance(image, Image.Image):
            image = np.array(image)
        image = self.transform(image=image)["image"]
        return image
