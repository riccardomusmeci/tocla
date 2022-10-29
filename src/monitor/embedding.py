import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple
from src.utils import Device
import torch.nn.functional as F
from src.monitor import ModelMonitor
from torch.utils.data import DataLoader
from src.dataset import InferenceDataset

class EmbeddingExtractor:
    
    def __init__(
        self,
        model: nn.Module,
        embed_layer: str,
        dataset: InferenceDataset,
        batch_size: int = 32,
        verbose: bool = True,
        is_vit: bool = False
    ) -> None:
        """Embedding extractor class

        Args:
            model (nn.Module): model 
            embed_layer (str): which embed layer of the model to consider
            dataset (InferenceDataset): dataset to extract embeddings from
            batch_size (int, optional): batch size. Defaults to 32.
            verbose (bool, optional): verbose mode. Defaults to True.
            is_vit (bool, optional): if model is ViT base. Defaults to False.
        """
        
        self.model = model.to(Device())
        self.model_monitor = ModelMonitor(model, verbose)
        if verbose:
            print(f"Adding {embed_layer} to monitor with hooks.")
        self.model_monitor.add_layer(layer_name=embed_layer)
        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size
        )
        self.embed_layer = embed_layer
        self.is_vit = is_vit
        
    @torch.no_grad()
    def extract(self) -> Tuple[np.array, np.array]:
        """computes embedding the embed layer

        Returns:
            Tuple[np.array, np.array]: tuple with embeddings and predictions from model
        """
        embeds, preds = [], []
        for batch in tqdm(self.data_loader, total=len(self.data_loader)):
            x, _ = batch
            x = x.to(Device())
            logits = self.model(x)
            _, pred = torch.max(logits, dim=1)
            layer_out = self.model_monitor.get_data(layer_name=self.embed_layer)
            
            # avg pool on feature maps (h, w)
            if self.is_vit:
                if len(layer_out.shape)==2:
                    embed = layer_out
                if len(layer_out.shape)==3:
                    embed = layer_out[:, 0, :]
            else:
                embed = F.avg_pool2d(
                    layer_out, (layer_out.size(-2), layer_out.size(-1))
                )
            embeds.append(embed.cpu().numpy().squeeze())
            preds.append(pred.cpu().numpy().squeeze())
        # stacking different batches
        try:
            embeds = np.vstack(embeds)
        except Exception as e:
            print(f"Unable to stack embed, returning list of embeds. Error {e}.")
        
        preds = np.concatenate(preds)
            
        return embeds, preds
        
        
        
        
    
        
    