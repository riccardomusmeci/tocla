import os
from PIL import Image

def read_rgb(file_path: str) -> Image:
    """Load an image from a path with Pillow

    Args:
        file_path (str): path to image file 

    Raises:
        FileNotFoundError: if file was not found.

    Returns:
        Image: image
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The path {file_path} does not exist")
    image = Image.open(file_path).convert("RGB")
    return image
    