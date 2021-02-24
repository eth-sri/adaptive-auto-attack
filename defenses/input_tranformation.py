#
# transform = nn.Sequential(
#     kornia.color.AdjustBrightness(0.5),
#     kornia.color.AdjustGamma(gamma=2.),
#     kornia.color.AdjustContrast(0.7),
# )

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

from __future__ import absolute_import, division, print_function, unicode_literals

from io import BytesIO
import logging

import torch
import torch.nn as nn
import torchvision
import numpy as np
import kornia

from art.config import ART_NUMPY_DTYPE
from .defense_module import DefenseModule

logger = logging.getLogger(__name__)


class InputTransformation(DefenseModule):

    params = ["degrees", "translate", "scales"]

    def __init__(self, degrees=(-40, 40), apply_fit=True, translate=None, scales=None, apply_predict=True):
        super(InputTransformation, self).__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.set_params(degrees=degrees, translate=translate, scales=scales)

    def forward(self, x):
        my_fcn = kornia.augmentation.RandomAffine(self.degrees, self.translate, self.scales, return_transform=False)
        return my_fcn(x)

