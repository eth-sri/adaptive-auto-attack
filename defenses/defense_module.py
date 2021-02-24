import torch
import torch.nn as nn

class DefenseModule(nn.Module):

    def __init__(self, apply_fit=False, apply_predict=True):
        super(DefenseModule, self).__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict

        # kwargs = {"beta": beta, "gamma": gamma}
        # self.set_params(**kwargs)

    @property
    def apply_fit(self):
        return self._apply_fit

    @property
    def apply_predict(self):
        return self._apply_predict

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and apply checks before saving them as attributes.

        :return: `True` when parsing was successful.
        """
        for key, value in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        return True
