import torch
from torch.nn import Module
from torch.optim import Adam, Optimizer

LAYERS = 4
HIDDEN = 1024

BATCH = 64
TOLERANCE = 5

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_optimiser(model: Module) -> Optimizer:
    return Adam(model.parameters())
