"""Check that PyTorch can train on the Apple GPU (MPS backend).

Run: .venv/bin/python 00-setup/mps_check.py
Exits with code 1 if MPS is not available.
"""

import platform
import sys

import torch
from torch import nn

# 1. Versions and MPS status.
print(f"python  {platform.python_version()}")
print(f"torch   {torch.__version__}")
print(f"mps built:     {torch.backends.mps.is_built()}")
print(f"mps available: {torch.backends.mps.is_available()}")

if not torch.backends.mps.is_available():
    print("MPS is not available. Stop here.", file=sys.stderr)
    sys.exit(1)

device = torch.device("mps")
torch.manual_seed(0)

# 2. Fake data: y = X @ w_true + b_true + small noise.
#    1024 rows, 8 features. The model must recover w_true and b_true.
X = torch.randn(1024, 8, device=device)
w_true = torch.randn(8, 1, device=device)
b_true = 0.5
y = X @ w_true + b_true + 0.01 * torch.randn(1024, 1, device=device)

# 3. Model: one linear layer (8 inputs -> 1 output), moved to the GPU.
model = nn.Linear(8, 1).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

# 4. Training loop: forward, loss, backward, update.
for step in range(1, 201):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"step {step:3d}  loss {loss.item():.6f}")

# 5. Where the parameters live, and how close they are to the truth.
print(f"weight device: {model.weight.device}")
print(f"bias device:   {model.bias.device}")
w_err = (model.weight.detach().T - w_true).abs().max().item()
print(f"max |w - w_true|: {w_err:.4f}   bias: {model.bias.item():.4f} (true {b_true})")
