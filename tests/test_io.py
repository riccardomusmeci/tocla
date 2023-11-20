import os

import numpy as np
from path import CONFIG_PATH, IMAGES_DIR

from tocla.io import ToclaConfiguration, read_rgb, resize_rgb


def test_load_config() -> None:
    config = ToclaConfiguration.load(config_path=CONFIG_PATH)
    assert config["model"]["model_name"] == "efficientnet_b0"


def test_read_rgb_pil() -> None:
    """Test read_rgb."""
    image = read_rgb(os.path.join(IMAGES_DIR, "dog", "dog1.png"), engine="pil")
    assert image.shape == (292, 398, 3)
    assert image.dtype == np.uint8


def test_read_rgb_cv2() -> None:
    """Test read_rgb."""
    image = read_rgb(os.path.join(IMAGES_DIR, "dog", "dog1.png"), engine="cv2")
    assert image.shape == (292, 398, 3)
    assert image.dtype == np.uint8


def test_resize_rgb() -> None:
    """Test resize_rgb."""
    image = read_rgb(os.path.join(IMAGES_DIR, "dog", "dog1.png"), engine="cv2")
    image = resize_rgb(image, w=224, h=224)
    assert image.shape == (224, 224, 3)
    assert image.dtype == np.uint8
