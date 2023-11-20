import os

import torch
from path import IMAGES_DIR
from torch.nn.functional import sigmoid

from tocla.dataset import InferenceDataset
from tocla.loss import create_criterion
from tocla.model import create_model
from tocla.transform import Transform

torch.manual_seed(42)


def test_model() -> None:
    model = create_model(model_name="resnet18", num_classes=1, pretrained=False)
    inference_dataset = InferenceDataset(
        data_dir=os.path.join(IMAGES_DIR, "dog"), transform=Transform(train=False, input_size=224)
    )
    x, _ = inference_dataset[0]
    x = x.unsqueeze(dim=0)
    out = sigmoid(model(x).squeeze(dim=0))
    assert out.shape == torch.Size([1])
    assert out.max() <= 1
    assert out.min() >= 0


def test_loss() -> None:
    loss = create_criterion("xent", label_smoothing=0.1)
    model = create_model(model_name="resnet18", num_classes=1, pretrained=False)
    x = torch.rand((1, 3, 224, 224))
    logits = model(x)
    target = torch.randint(1, (1,), dtype=torch.int64)
    loss_value = loss(logits, target)
    assert loss_value >= 0
