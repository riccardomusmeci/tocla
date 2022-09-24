import torch
import numpy as np
import torch.nn as nn

FACTORY = {
    "xent": nn.CrossEntropyLoss
}

def criterion(
    name: str = "xent",
    **kwargs
) -> nn.Module:
    
    name = name.lower()
    assert name in FACTORY.keys(), f"Only {list(FACTORY.keys())} criterions are supported. Change {name} to one of them."
    
    if 'weight' in kwargs and name=="xent":
        if isinstance(kwargs['weight'], list):
            kwargs['weight'] = torch.Tensor(kwargs['weight'])
        else:
            kwargs['weight'] = None
            
    if 'label_smoothing' in kwargs and name=="xent":
        if kwargs['label_smoothing'] > 0:
           kwargs['label_smoothing'] = np.float32(kwargs['label_smoothing'])
    
    return FACTORY[name](**kwargs)