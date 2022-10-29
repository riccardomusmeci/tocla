import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from src.utils import Device
import torch.nn.functional as F
from src.monitor import ModelMonitor
from torch.utils.data import DataLoader
from src.dataset import InferenceDataset
from typing import List, Tuple, Union, Dict
from scipy.stats import wasserstein_distance

class DomainShiftMonitor:
        
    def __init__(
        self,
        model: nn.Module,
        datasets: Tuple[InferenceDataset, InferenceDataset],
        target_layers: Union[List[str], str], # for ModelMonitor
        batch_size: int = 32,
        verbose: bool = True
    ) -> None:
        
        
        assert len(datasets)==2, f"You can analyze two datasets at a time, not {len(datasets)}."
        
        if not isinstance(target_layers, list):
            target_layers = [target_layers]
        
        self.model = model
        self.model.to(Device())
        self.datasets = datasets
        self.target_layers = target_layers
        self.batch_size = batch_size
        
        print(f"> Creating model monitor to get embeddings at specific layers.")
        self.model_monitor = ModelMonitor(
            model=model,
            verbose=verbose
        )
                
        for target_layer in self.target_layers:
            print(f"Adding {target_layer} to monitor with hooks.")
            self.model_monitor.add_layer(layer_name=target_layer)
    
    @torch.no_grad()
    def get_embedding(
        self,
        data_loader: DataLoader
    ) -> Dict[str, np.array]:
        """computes embedding for each target layer

        Args:
            data_loader (DataLoader): data loader

        Returns:
            Dict[str, np.array]: dict with target layer as key and dataset embeddings
        """
        embeddings = { layer_name: [] for layer_name in self.target_layers }
        for layer_name in self.target_layers:
            for batch in tqdm(data_loader, total=len(data_loader)):
                x, target = batch
                x = x.to(Device())
                logits = self.model(x)
                layer_out = self.model_monitor.get_data(layer_name=layer_name)
                # avg pool on feature maps (h, w)
                avg_pooled_embed = F.avg_pool2d(
                    layer_out, (layer_out.size(-2), layer_out.size(-1))
                )
                embeddings[layer_name].append(
                    avg_pooled_embed.cpu().numpy().squeeze()
                )
            # stacking different batches
            embeddings[layer_name] = np.vstack(embeddings[layer_name])
        
        return embeddings
                
    def estimate(
        self,
    ) -> Dict[str, np.array]:
        """estimates the domain shift with wasserstein distance for each target layer

        Returns:
            Dict[str, np.array]: for each target layer X, y distribution (y represents two datasets).
        """
        embeddings = []
        for i, dataset in enumerate(self.datasets):
            print(f"> Extracting embedding for the {'first' if i==0 else 'second'} dataset.")
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size
            )
            embeddings.append(self.get_embedding(data_loader=data_loader))
        
        print(" ================================= ")
        
        out_dist = { layer_name: {"X": [], "y": []} for layer_name in self.target_layers }
        for layer_name in self.target_layers:
            print(f"> Estimating domain shift for layer: {layer_name}")
            print(f"> Dataset 1 embedding shape: {embeddings[0][layer_name].shape}")
            print(f"> Dataset 2 embedding shape: {embeddings[1][layer_name].shape}")
            
            # aggregated emebddings between the two datasets
            X = np.vstack([
                embeddings[0][layer_name],
                embeddings[1][layer_name],
            ])
            
            # assigning 0 to first dataset and 1 to second dataset
            y = np.concatenate([
                    np.zeros(embeddings[0][layer_name].shape[0]),
                    np.ones(embeddings[1][layer_name].shape[0])
            ])
            out_dist[layer_name]["X"] = X
            out_dist[layer_name]["y"] = y
            
            R_mean = 0
            for k in range(X.shape[1]):
                R_mean += wasserstein_distance(
                    embeddings[0][layer_name][:, k],
                    embeddings[1][layer_name][:, k],
                )
            R_mean /= X.shape[1]
            print(f"Representation shift using wasserstein distance: ", R_mean)
            
        return out_dist
            
            
        
    
    
        
        
        
    
    