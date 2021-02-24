import torch
import torch.nn as nn
from .defense_module import DefenseModule


class LossCombine(DefenseModule):
    def __init__(self, apply_fit=False, apply_predict=True):
        super(LossCombine, self).__init__(apply_fit, apply_predict)

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "Loss Combine cannot be list"
        if x.dim() == 2:
            return x
        assert(x.dim() == 3)
        return torch.mean(x, dim=1)