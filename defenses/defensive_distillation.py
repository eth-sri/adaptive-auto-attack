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
This module implements the transforming defence mechanism of defensive distillation.

| Paper link: https://arxiv.org/abs/1511.04508
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F
import numpy as np

from art.defences.transformer.transformer import Transformer
from art.utils import is_probability


class DefensiveDistillation:

    params = ["batch_size", "nb_epochs"]

    def __init__(self, classifier):

        self.target_classifier = classifier
        self._is_fitted = True

    def fit(self, classifier, loader, optimizer, nb_epochs, temp=1):
        """
        IMPORTANT : The two models have to output logits
        """
        # Check if the trained classifier produces probability outputs
        from tqdm import tqdm
        # if isinstance(loss_fcn, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss, torch.nn.MultiMarginLoss)):

        # Using KL divergence as the loss
        # loss_fcn = torch.nn.KLDivLoss(reduction='batchmean')
        loss_fcn = torch.nn.KLDivLoss(reduction='batchmean')
        # Start training
        optimizer.zero_grad()
        for i in range(nb_epochs):
            pbar = tqdm(loader)
            for i_batch, _ in pbar:
                i_batch = i_batch.to('cuda')
                target_batch = F.softmax(self.target_classifier(i_batch)/temp, dim=-1)
                model_outputs = F.log_softmax(classifier(i_batch), dim=-1)  # pytorch KLdiv requires log softmax for the first argument
                loss = -torch.mean(model_outputs * target_batch) * target_batch.shape[1]  # batch mean of x'*logx, x is the
                loss.backward()
                optimizer.step()
                pbar.set_description("epoch {:d}: loss {:.3f}".format(i + 1, loss))
                optimizer.zero_grad()

        return classifier
