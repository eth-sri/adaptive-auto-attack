# MIT License
#
# Copyright (C) IBM Corporation 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the Reverse Sigmoid perturbation for the classifier output.

| Paper link: https://arxiv.org/abs/1806.00054
"""

import torch
import torch.nn as nn
from .defense_module import DefenseModule


class ReverseSigmoid(DefenseModule):
    """
    Implementation of a postprocessor based on adding the Reverse Sigmoid perturbation to classifier output.
    """

    params = ["beta", "gamma"]

    def __init__(self, beta=1.0, gamma=0.1, apply_fit=False, apply_predict=True):
        """
        Create a ReverseSigmoid postprocessor.
        :param beta: A positive magnitude parameter.
        :type beta: `float`
        :param gamma: A positive dataset and model specific convergence parameter.
        :type gamma: `float`
        :param apply_fit: True if applied during fitting/training.
        :type apply_fit: `bool`
        :param apply_predict: True if applied during predicting.
        :type apply_predict: `bool`
        """
        super(ReverseSigmoid, self).__init__(apply_fit=False, apply_predict=True)
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        kwargs = {"beta": beta, "gamma": gamma}
        self.set_params(**kwargs)

    def forward(self, preds):
        clip_min = 1e-5
        clip_max = 1.0 - clip_min
        def sigmoid(var_z):
            return 1.0 / (1.0 + torch.exp(-var_z))

        # preds = torch.tensor(preds)
        if preds.dim() == 1:
            preds = preds.unsqueeze(0)
        predfcn = nn.Softmax(dim=1)
        prob = predfcn(preds)
        prob_clipped = torch.clamp(prob, clip_min, clip_max)

        if preds.shape[1] > 1:
            perturbation_r = self.beta * (sigmoid(-self.gamma * torch.log((1.0 - prob_clipped) / prob_clipped)) - 0.5)
            prob_clipped = prob - perturbation_r
            prob_clipped = torch.clamp(prob_clipped, 0.0, 1.0)
            coeff = 1.0 / torch.sum(prob_clipped, axis=-1, keepdims=True)
            reverse_sigmoid = coeff * prob_clipped
        else:
            preds_1 = prob
            preds_2 = 1.0 - prob

            preds_clipped_1 = prob_clipped
            preds_clipped_2 = 1.0 - prob_clipped

            perturbation_r_1 = self.beta * (
                sigmoid(-self.gamma * torch.log((1.0 - preds_clipped_1) / preds_clipped_1)) - 0.5
            )
            perturbation_r_2 = self.beta * (
                sigmoid(-self.gamma * torch.log((1.0 - preds_clipped_2) / preds_clipped_2)) - 0.5
            )

            preds_perturbed_1 = preds_1 - perturbation_r_1
            preds_perturbed_2 = preds_2 - perturbation_r_2

            preds_perturbed_1 = torch.clip(preds_perturbed_1, 0.0, 1.0)
            preds_perturbed_2 = torch.clip(preds_perturbed_2, 0.0, 1.0)

            alpha = 1.0 / (preds_perturbed_1 + preds_perturbed_2)
            reverse_sigmoid = alpha * preds_perturbed_1

        return reverse_sigmoid

