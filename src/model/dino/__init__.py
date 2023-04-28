from .vit_dino import ViTDINO
from .vit_dino import _EMBEDDINGS

__all__ = ["ViTDINO", "vit_dinov2_models"]

vit_dinov2_models: list = list(_EMBEDDINGS.keys())
