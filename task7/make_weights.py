"""Rebuild the tested task-7 candidate from participant starter weights.

Place this script in the root of the extracted participant package, next to
examples/ and torch_lander/, then run: python make_weights.py
"""
import torch
from torch_lander import load, export

model = load("examples/weights.pt")
with torch.no_grad():
    model[0].weight[0].mul_(1.6)
    model[0].weight[1].mul_(1.3)
export(model, "weights.pt")
print("Saved weights.pt")
