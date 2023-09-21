import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from copy import deepcopy

# Data distribution is 30% N(0.5, 0.0001), 30% N(0, 0.0009), 40% N(-0.4, 0.0001)
data  = torch.cat([
    torch.randn(3000, 1) * 0.01 + 0.5,
    torch.randn(3000, 1) * 0.03 + 0.1,
    torch.randn(4000, 1) * 0.01 - 0.4,
])
plt.hist(data.numpy(), bins=100)
plt.title("Ground Truth Data distribution")
plt.show()

# Create dataset and dataloader
dataset = TensorDataset(data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# print(dataloader)

# Define global constants
T = 5            # The maximum time, which is equal to the maximum noise standard deviation

EPSILON = 0.002  # The minimum time

class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        "x: shape (batch, 1) t: shape (batch,); return shape (batch, 1)"
        inputs = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        outputs = self.mlp(inputs)
        return outputs

# The score function is derived from the denoiser
def score_function(denoiser, x, t):
    return (denoiser(x, t) - x) / torch.square(t).unsqueeze(-1)

class ConsistencyFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        "x: shape (batch, 1) t: shape (batch,); return shape (batch, 1)"
        inputs = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        outputs = self.mlp(inputs)
        return ((T - t) / (T - EPSILON)).unsqueeze(-1) * x + ((t - EPSILON) / (T - EPSILON)).unsqueeze(-1) * outputs