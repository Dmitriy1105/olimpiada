"""CELL 11A: experimental adaptive channel in the toy lander network."""
import copy
import torch

@torch.no_grad()
def adaptive_candidate(base_model, alpha, sharpness=60.0):
    model = copy.deepcopy(base_model).cpu().eval()
    w2 = model[2].weight
    w3 = model[4].weight
    # Select a weakly connected auxiliary neuron, not either main channel.
    influence = (w3.abs() @ w2.abs()).sum(dim=0)
    j = int(influence[2:].argmin()) + 2
    model[0].weight[j].zero_()
    model[0].bias[j].zero_()
    model[0].weight[j, 8] = sharpness * 1.675
    model[0].weight[j, 9] = -2.0 * sharpness
    w2[:, j].zero_()
    w2[0, j] = alpha / 2
    model[2].bias[0].add_(alpha / 2)
    return model, j
