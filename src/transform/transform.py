import PIL
import numpy as np
from PIL import Image
import albumentations as A
from src.utils import to_tensor
from typing import Union, Tuple, List

#TODO: add augmix
class Transform:
    
    def __init__(
        self,
        train: bool,
        img_size: Union[int, list, tuple], 
        mean: list = [0.485, 0.456, 0.406], 
        std: list = [0.229, 0.224, 0.225],
        h_flip_p: float = .5,
        v_flip_p: float = .5,
        ssr_p: float = 0.3,
        ssr_border_mode: int = 0,
        ssr_rotate_limit: int = 20,
        ssr_scale_limit: float = 0.5,
        ssr_shift_limit: float = 0.1,
        ssr_value: float = 0,
        gn_var_limit: List[float] = [10, 50],
        gn_mean: float = 0,
        gn_p: float = 0.2,
        crop_p: float = 0.5,
        one_of_p: float = .66,
        sharpen_alpha: List[float] = [.2, .5],
        sharpen_lightness: List[float] = [.5, 1],
        sharpen_p: float = .5,
        blur_limit: List[int] = [3, 3],
        blur_p: float = 0.25,
        mblur_limit: List[int] = [3, 3],
        mblur_p: List[int] = 0.25,
        rbc_brightness_limit: float =0.1,
        rbc_contrast_limit: float = 0.1,
        rbc_p: float = .5,
        hsv_sat_shift_limit: int = 10,
        hsv_hue_shift_limit: int = 10,
        hsv_val_shift_limit: int = 10,
        hsv_p: float = .5,  
    ):
        """Classifer Transform

        Args:
            train (bool): train mode
            img_size (Union[int, list, tuple]): image input size
            mean (list, optional): normalization mean. Defaults to [0.485, 0.456, 0.406].
            std (list, optional): normalization std. Defaults to [0.229, 0.224, 0.225].
            h_flip_p (float, optional): HorizontalFlip transformation probabilty. Defaults to .5.
            v_flip_p (float, optional): VerticalFlip transformation probability. Defaults to .5.
            ssr_p (float, optional): ShiftScaleRotate probability. Defaults to 0.3.
            ssr_border_mode (int, optional): ShiftScaleRotate border mode. Defaults to 0.
            ssr_rotate_limit (int, optional): ShiftScaleRotate rotate limit. Defaults to 20.
            ssr_scale_limit (float, optional): ShiftScaleRotate scale limit. Defaults to 0.5.
            ssr_shift_limit (float, optional): ShiftScaleRotate shift limit. Defaults to 0.1.
            ssr_value (float, optional): ShiftScaleRotate value. Defaults to 0.
            gn_var_limit (List[float], optional): GaussianNoise variance limit. Defaults to [10, 50].
            gn_mean (float, optional): GaussianNoise mean. Defaults to 0.
            gn_p (float, optional): GaussianNoise probability. Defaults to 0.2.
            crop_p (float, optional): Crop probabilty  . Defaults to 0.5.
            one_of_p (float, optional): OneOf transformation probability (first set: Sharpen + Blur + MotionBlur, second set: RandomBrightnessContrast + HueSaturationValue). Defaults to .66.
            sharpen_alpha (List[float], optional): Sharpen alpha. Defaults to [.2, .5].
            sharpen_lightness (List[float], optional): Sharpen lightness. Defaults to [.5, 1].
            sharpen_p (float, optional): Sharpen transformation probability. Defaults to .5.
            blur_limit (List[int], optional): Blur limit. Defaults to [3, 3].
            blur_p (float, optional): Blur transformation probability. Defaults to 0.25.
            mblur_limit (List[int], optional): MotionBlur limit. Defaults to [3, 3].
            mblur_p (List[int], optional): MotionBlur transformation probability. Defaults to 0.25.
            rbc_brightness_limit (float, optional): RandomBrightnessContrast brightness limit. Defaults to 0.1.
            rbc_contrast_limit (float, optional): RandomBrightnessContrast contrast limit. Defaults to 0.1.
            rbc_p (float, optional): RandomBrightnessContrast transformation probability. Defaults to .5.
            hsv_sat_shift_limit (int, optional): HueSaturationValue saturation limit. Defaults to 10.
            hsv_hue_shift_limit (int, optional): HueSaturationValue hue limit. Defaults to 10.
            hsv_val_shift_limit (int, optional): HueSaturationValue value limit. Defaults to 10.
            hsv_p (float, optional): HueSaturationValue transformation probability. Defaults to .5.
        """
        
        if isinstance(img_size, tuple) or isinstance(img_size, list):
            height, width = img_size[0], img_size[1]
        else:
            height, width = img_size, img_size
            
        if train:
            self.transform = A.Compose([
                A.Resize(
                    height=height, 
                    width=width, 
                    always_apply=True,
                    interpolation=Image.BICUBIC
                ),
                # Flip and ColorJitter
                A.HorizontalFlip(p=h_flip_p),
                A.VerticalFlip(p=v_flip_p),
                A.ShiftScaleRotate(
                    p=ssr_p,
                    border_mode=ssr_border_mode,
                    rotate_limit=ssr_rotate_limit,
                    scale_limit=ssr_scale_limit,
                    shift_limit=ssr_shift_limit,
                    value=ssr_value,
                ),
                A.GaussNoise(
                    var_limit=gn_var_limit,
                    mean=gn_mean,
                    p=gn_p
                ),
                A.RandomCrop(
                    height=height, 
                    width=width, 
                    p=crop_p, 
                ),
                A.PadIfNeeded(
                    min_height=height,
                    min_width=width,
                ),
                A.OneOf(
                    transforms=[
                        A.Sharpen(
                            alpha=sharpen_alpha,
                            lightness=sharpen_lightness,
                            p=sharpen_p
                        ),
                        A.Blur(
                            blur_limit=blur_limit,
                            p=blur_p
                        ),
                        A.MotionBlur(
                            blur_limit=mblur_limit,
                            p=mblur_p
                        )
                    ],
                    p=one_of_p
                ),
                A.OneOf(
                    transforms=[
                        A.RandomBrightnessContrast(
                            brightness_limit=rbc_brightness_limit,
                            contrast_limit=rbc_contrast_limit,
                            p=rbc_p
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=hsv_hue_shift_limit,
                            sat_shift_limit=hsv_sat_shift_limit,
                            val_shift_limit=hsv_val_shift_limit,
                            p=hsv_p
                        )
                    ],
                    p=one_of_p
                ),
                A.Resize(
                    height=height, 
                    width=width, 
                    interpolation=Image.BICUBIC,
                    always_apply=True
                ),
                # Normalization
                A.Normalize(
                    mean=mean, 
                    std=std
                )            
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