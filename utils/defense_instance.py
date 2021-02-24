from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn as nn
from utils.utils import batch_forward


class DetectorWrapper(nn.Module):
    def __init__(self, model):
        super(DetectorWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


class DefenseInstance:
    """
    DefenseInstance is the wrapper for a model and detector
    The main method is predict, which returns model prediction and detector prediction
    """
    def __init__(
        self,
        model,
        detector=None,
        device="cuda",
    ):
        self.device = device
        self.model = model
        self.detector = detector

    def predict(self, x):
        if self.detector is not None:
            det = batch_forward(self.detector, x)
            if det.dim() > 1:
                det = torch.argmax(det, dim=1).to(torch.bool)
            else:
                det = det.to(torch.bool)
        else:
            # return all True in the absence of a detector
            det = torch.ones(x.shape[0]).to(torch.bool)
        return [self.forward(x), det.to(self.device)]

    def forward(self, x):
        return batch_forward(self.model, x)

    def __call__(self, x):
        return batch_forward(self.model, x)

    def detector_fit(self, loader, loss_fcn, optimizer, nb_epochs=10, **kwargs):
        self.detector.fit(loader=loader, loss_fcn=loss_fcn, optimizer=optimizer, nb_epochs=nb_epochs, **kwargs)

    def fit(self, loader, loss_fcn, optimizer, nb_epochs=10, **kwargs):
        self.model.fit(loader=loader, loss_fcn=loss_fcn, optimizer=optimizer, nb_epochs=nb_epochs, **kwargs)

    def advfit(self, loader, loss_fcn, optimizer, epsilon, attack, nb_epochs=10, ratio=0.5, **kwargs):
        return self.model.advfit(loader=loader, loss_fcn=loss_fcn, optimizer=optimizer, epsilon=epsilon,
                                 attack=attack, nb_epochs=nb_epochs, ratio=ratio, **kwargs)

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(mode=False)

    def get_attack_model(self, bounds):
        # model = self.model
        # return fb.PyTorchModel(model.eval(), bounds=bounds)
        return self.model

