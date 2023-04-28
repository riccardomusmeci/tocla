import torch
import torch.nn as nn

_EMBEDDINGS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    "dino_vits16": 384,
    "dino_vits8": 384,
    "dino_vitb16": 768,
    "dino_vitb8": 768
}

class ViTDINO(nn.Module):
    
    def __init__(self, model_name: str, num_classes: int = 0, **kwargs) -> None:
        super().__init__()
        
        assert model_name in _EMBEDDINGS.keys(), f"DINOv2 versions available: {_EMBEDDINGS.keys()}"
        repo = 'facebookresearch/dinov2' if "dinov2" in model_name else 'facebookresearch/dino:main'
        self.backbone = torch.hub.load(repo, model_name)
        self.head = nn.Linear(
            in_features=_EMBEDDINGS[model_name],
            out_features=num_classes
        )
        
    def forward(self, x: torch.Tensor):
        return self.head(self.backbone(x))