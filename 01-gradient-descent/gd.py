"""Fit a line y = w*x + b to three points by gradient descent.

Same setup as the course lab "Three points, one line".
Run: .venv/bin/python 01-gradient-descent/gd.py
Fill in the two TODOs first. The script writes loss.png next to this file.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to a file, no window
import matplotlib.pyplot as plt
import numpy as np

# Data: the three points from the lab.
x = np.array([0.0, 1.0, 2.5])
y = np.array([0.4, 1.7, 2.1])

# Start point and learning rate (eta) from the lab.
W_START, B_START = -1.5, 2.5
ETA = 0.01
N_STEPS = 300


def sse(w, b):
    """Loss = sum of squared errors, the same loss as the lab.

    (Divide by len(x) to get the mean squared error. The lab does not,
    so keep the sum here to make ETA behave as it does in the lab.)
    """
    residual = w * x + b - y
    return np.sum(residual**2)


def gradient(w, b):
    """Return (dL/dw, dL/db) at the point (w, b)."""
    # TODO 1: compute the two slopes of sse with respect to w and b.
    d0 = d1 = 0.0
    for xi, yi in zip(x, y):
        err = w * xi + b - yi   # f[x_i, φ] − y_i
        d0 += 2 * err * xi
        d1 += 2 * err
    return d0, d1


def step(w, b, dw, db, eta):
    """Return the new (w, b) after one gradient descent update."""
    # TODO 2: move (w, b) one step of size eta.
    return w - eta * dw, b - eta * db 


# Run gradient descent and record the loss after each step.
w, b = W_START, B_START
losses = [sse(w, b)]
for i in range(N_STEPS):
    dw, db = gradient(w, b)
    w, b = step(w, b, dw, db, ETA)
    losses.append(sse(w, b))

print(f"start: w={W_START:.4f}  b={B_START:.4f}  loss={losses[0]:.6f}")
print(f"end:   w={w:.4f}  b={b:.4f}  loss={losses[-1]:.6f}  ({N_STEPS} steps)")

# Plot loss per step. Log scale on y shows the slow tail.
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(losses)
ax.set_yscale("log")
ax.set_xlabel("step")
ax.set_ylabel("loss (sum of squared errors)")
ax.set_title(f"Gradient descent, eta = {ETA}")
fig.tight_layout()
out = Path(__file__).with_name("loss.png")
fig.savefig(out, dpi=120)
print(f"saved {out}")
