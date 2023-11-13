import torch.nn as nn


def scale_grad(x, weight):
    return weight * x + ((1.0 - weight) * x).detach()


class GradScaler(nn.Module):
    def __init__(self, weight):
        super(GradScaler, self).__init__()
        self.weight = (weight,)  # wrap the weight as a "pointer" to not add it as parameter

    def forward(self, x):
        return scale_grad(x, self.weight[0])
