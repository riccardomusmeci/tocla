import torch

def device():    
    if torch.has_cuda: return "cuda"
    if torch.has_mps: return "mps"
    return "cpu"