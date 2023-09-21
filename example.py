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
# plt.show()
plt.savefig("./assert/Ground_Truth_Distribution.png")

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
    

# Score function (Denoiser actually...) training
denoiser = Denoiser()
denoiser_optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)
denoiser.train()

# Discretize time steps. From EDM Karras et al. It is dense in the beginning and sparse in the end
n_steps_straining = 1000
t_steps = [(EPSILON ** (1 / 7) + (i / (n_steps_straining - 1)) * (T ** (1 / 7) - EPSILON ** (1 / 7))) ** 7 for i in range(0, n_steps_straining)]

N_EPOCHS = 100

with trange(N_EPOCHS) as pbar:
    for epoch in range(N_EPOCHS):
        tot_loss = 0.0
        for x, in dataloader:
            batch_size = x.shape[0]
            
            # Sample time
            t = torch.tensor(random.choices(t_steps, k=batch_size))
            
            # Sample random Gaussian noise
            z = torch.randn(batch_size, 1)

            # Compute loss and update score function xt|x ~ N(xt; x, t^2 I)
            loss = F.mse_loss(denoiser(x + t.unsqueeze(-1) * z, t), x)
            
            loss.backward()
            denoiser_optimizer.step()
            denoiser_optimizer.zero_grad()

            tot_loss += loss.item()
        pbar.set_postfix(loss=tot_loss / len(dataloader))
        pbar.update()


@torch.no_grad()
def deterministic_sampling(xT: torch.Tensor, n_steps: int):
    "Deterministic sampling using Heun's 2nd order method"
    t_steps = [(EPSILON ** (1 / 7) + (j / (n_steps - 1)) * (T ** (1 / 7) - EPSILON ** (1 / 7))) ** 7 for j in range(n_steps - 1, 0, -1)]
    batch_size = xT.shape[0]
    xi = xT
    trajectory = [xT]
    for i in range(0, len(t_steps) - 1):
        d = -t_steps[i] * score_function(denoiser, xi, torch.ones(batch_size) * t_steps[i])
        xi_1 = xi + (t_steps[i + 1] - t_steps[i]) * d
        d_ = -t_steps[i + 1] * score_function(denoiser, xi_1, torch.ones(batch_size) * t_steps[i + 1])
        xi_1 = xi + (t_steps[i + 1] - t_steps[i]) / 2 * (d + d_)

        xi = xi_1
        trajectory.append(xi)
        
    return xi, (t_steps, trajectory)