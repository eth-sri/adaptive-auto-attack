# MIT License
#
# Copyright (C) IBM Corporation 2018
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
This module implements the feature squeezing defence in `FeatureSqueezing`.

| Paper link: https://arxiv.org/abs/1704.01155

| Please keep in mind the limitations of defences. For more information on the limitations of this defence, see
    https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import torch
from .defense_module import DefenseModule


class FeatureSqueezing(DefenseModule):
    """
    Reduces the sensibility of the features of a sample.

    | Paper link: https://arxiv.org/abs/1704.01155

    | Please keep in mind the limitations of defences. For more information on the limitations of this defence,
        see https://arxiv.org/abs/1803.09868 . For details on how to evaluate classifier security in general,
        see https://arxiv.org/abs/1902.06705
    """

    params = ["clip_values", "bit_depth"]

    def __init__(self, clip_values, bit_depth=8, apply_fit=False, apply_predict=True):
        """
        Create an instance of feature squeezing.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param bit_depth: The number of bits per channel for encoding the data.
        :type bit_depth: `int`
        :param apply_fit: True if applied during fitting/training.
        :type apply_fit: `bool`
        :param apply_predict: True if applied during predicting.
        :type apply_predict: `bool`
        """
        super(FeatureSqueezing, self).__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.set_params(clip_values=clip_values, bit_depth=bit_depth)

    def forward(self, x):
        """
        Apply feature squeezing to sample `x`.

        :param x: Sample to squeeze. `x` values are expected to be in the data range provided by `clip_values`.
        :type x: `np.ndarrray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :return: Squeezed sample.
        :rtype: `np.ndarray`
        """
        device = None
        if isinstance(x, torch.Tensor):
            device = x.device
            x = x.clone().cpu().detach().numpy()
        else:
            assert 0

        x_normalized = x - self.clip_values[0]
        x_normalized = x_normalized / (self.clip_values[1] - self.clip_values[0])

        max_value = np.rint(2 ** self.bit_depth - 1)
        res = np.rint(x_normalized * max_value) / max_value

        res = res * (self.clip_values[1] - self.clip_values[0])
        res = res + self.clip_values[0]

        return torch.as_tensor(res).to(device)

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param bit_depth: The number of bits per channel for encoding the data.
        :type bit_depth: `int`
        """
        # Save defence-specific parameters
        super(FeatureSqueezing, self).set_params(**kwargs)

        if not isinstance(self.bit_depth, (int, np.int)) or self.bit_depth <= 0 or self.bit_depth > 64:
            raise ValueError("The bit depth must be between 1 and 64.")

        if len(self.clip_values) != 2:
            raise ValueError("`clip_values` should be a tuple of 2 floats containing the allowed data range.")

        if np.array(self.clip_values[0] >= self.clip_values[1]).any():
            raise ValueError("Invalid `clip_values`: min >= max.")

        return True
