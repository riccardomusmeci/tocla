import os
import torch 
import random
import numpy as np

def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
 
def device() -> str:
    if torch.cuda.is_available(): 
        return "cuda"
    if torch.has_mps: 
        return "mps"
    return "cpu"

def num_cpus(mode: str = "max") -> int:
    """Return number of cpus in the local machine.

    Args:
        mode (str, optional): if mode is safe, return num_cpus/2; else maximum number of cpus. Defaults to max.

    Returns:
        int: number of cpus
    """
    if mode not in ["max", "safe"]:
        print(f"> [WARNING] param mode with value {mode} is not valid. Setting to 'safe'")
        mode = "safe"
    if mode == "max":
        return os.cpu_count()
    else:
        return int(os.cpu_count()/2) 

NUM_CPUS = lambda mode: num_cpus(mode)
DEVICE = device()