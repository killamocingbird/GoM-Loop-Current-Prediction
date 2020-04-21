import numpy as np
import time
import torch

# Make runs deterministic for tracability
def set_seed(seed, device=None):  
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Buffered print to not break TQDM printing
def bprint(x, delay=0.5):
    time.sleep(delay/2)
    print(x)
    time.sleep(delay/2)

# Count parameters in a model   
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())