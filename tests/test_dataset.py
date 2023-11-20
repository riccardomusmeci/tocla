import os

from path import IMAGES_DIR

from tocla.dataset import ImageFolderDataset, InferenceDataset
from tocla.transform import Transform


def test_train_dataset() -> None:
    dataset = ImageFolderDataset(
        root_dir=IMAGES_DIR, transform=Transform(train=True, input_size=224), class_map={0: "cat", 1: "dog"}
    )

    img, target = dataset[0]
    assert img.shape == (3, 224, 224)
    assert len(dataset) == 6


def test_inference_dataset() -> None:
    dataset = InferenceDataset(
        data_dir=os.path.join(IMAGES_DIR, "dog"),
        transform=Transform(train=False, input_size=224),
    )

    img, _ = dataset[0]
    assert img.shape == (3, 224, 224)
    assert len(dataset) == 3
