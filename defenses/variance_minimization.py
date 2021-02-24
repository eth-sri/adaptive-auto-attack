"""
This module implements the total variance minimization defence `TotalVarMin`.

| Paper link: https://openreview.net/forum?id=SyJ7ClWCb

| Please keep in mind the limitations of defences. For more information on the limitations of this defence,
    see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general, see
    https://arxiv.org/abs/1902.06705
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
from scipy.optimize import minimize

from art.config import ART_NUMPY_DTYPE
import torch
import torch.nn as nn
from .defense_module import DefenseModule

logger = logging.getLogger(__name__)


class VarianceMinimization(DefenseModule):
    """
    Implement the total variance minimization defence approach.

    | Paper link: https://openreview.net/forum?id=SyJ7ClWCb

    | Please keep in mind the limitations of defences. For more information on the limitations of this
        defence, see https://arxiv.org/abs/1802.00420 . For details on how to evaluate classifier security in general,
        see https://arxiv.org/abs/1902.06705
    """

    params = ["prob", "norm", "lamb", "solver", "max_iter", "clip_values"]

    def __init__(
        self,
        prob=0.3,
        norm=2,
        lamb=0.5,
        solver="L-BFGS-B",
        max_iter=10,
        clip_values=None,
        apply_fit=False,
        apply_predict=True,
    ):
        """
        Create an instance of total variance minimization.

        :param prob: Probability of the Bernoulli distribution.
        :type prob: `float`
        :param norm: The norm (positive integer).
        :type norm: `int`
        :param lamb: The lambda parameter in the objective function.
        :type lamb: `float`
        :param solver: Current support: `L-BFGS-B`, `CG`, `Newton-CG`.
        :type solver: `str`
        :param max_iter: Maximum number of iterations when performing optimization.
        :type max_iter: `int`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        :param apply_fit: True if applied during fitting/training.
        :type apply_fit: `bool`
        :param apply_predict: True if applied during predicting.
        :type apply_predict: `bool`
        """
        super(VarianceMinimization, self).__init__()
        self._is_fitted = True
        self._apply_fit = apply_fit
        self._apply_predict = apply_predict
        self.set_params(prob=prob, norm=norm, lamb=lamb, solver=solver, max_iter=max_iter, clip_values=clip_values)

    def forward(self, x):
        """
        Apply total variance minimization to sample `x`.

        :param x: Sample to compress with shape `(batch_size, width, height, depth)`.
        :type x: `np.ndarray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :return: Similar samples.
        :rtype: `np.ndarray`
        """
        if len(x.shape) == 2:
            raise ValueError(
                "Feature vectors detected. Variance minimization can only be applied to data with spatial" "dimensions."
            )

        device = x.device
        x_preproc = x.detach().cpu().clone().numpy()

        # Minimize one input at a time
        for i, x_i in enumerate(x_preproc):
            mask = (np.random.rand(*x_i.shape) < self.prob).astype("int")
            x_preproc[i] = self._minimize(x_i, mask)

        if self.clip_values is not None:
            np.clip(x_preproc, self.clip_values[0], self.clip_values[1], out=x_preproc)

        return torch.as_tensor(x_preproc).to(device)

    def _minimize(self, x, mask):
        """
        Minimize the total variance objective function.

        :param x: Original image.
        :type x: `np.ndarray`
        :param mask: A matrix that decides which points are kept.
        :type mask: `np.ndarray`
        :return: A new image.
        :rtype: `np.ndarray`
        """
        z_min = x.copy()

        for i in range(x.shape[2]):
            res = minimize(
                self._loss_func,
                z_min[:, :, i].flatten(),
                (x[:, :, i], mask[:, :, i], self.norm, self.lamb),
                method=self.solver,
                jac=self._deri_loss_func,
                options={"maxiter": self.max_iter},
            )
            z_min[:, :, i] = np.reshape(res.x, z_min[:, :, i].shape)

        return z_min

    @staticmethod
    def _loss_func(z_init, x, mask, norm, lamb):
        """
        Loss function to be minimized.

        :param z_init: Initial guess.
        :type z_init: `np.ndarray`
        :param x: Original image.
        :type x: `np.ndarray`
        :param mask: A matrix that decides which points are kept.
        :type mask: `np.ndarray`
        :param norm: The norm (positive integer).
        :type norm: `int`
        :param lamb: The lambda parameter in the objective function.
        :type lamb: `float`
        :return: Loss value.
        :rtype: `float`
        """
        res = np.sqrt(np.power(z_init - x.flatten(), 2).dot(mask.flatten()))
        z_init = np.reshape(z_init, x.shape)
        res += lamb * np.linalg.norm(z_init[1:, :] - z_init[:-1, :], norm, axis=1).sum()
        res += lamb * np.linalg.norm(z_init[:, 1:] - z_init[:, :-1], norm, axis=0).sum()

        return res

    @staticmethod
    def _deri_loss_func(z_init, x, mask, norm, lamb):
        """
        Derivative of loss function to be minimized.

        :param z_init: Initial guess.
        :type z_init: `np.ndarray`
        :param x: Original image.
        :type x: `np.ndarray`
        :param mask: A matrix that decides which points are kept.
        :type mask: `np.ndarray`
        :param norm: The norm (positive integer).
        :type norm: `int`
        :param lamb: The lambda parameter in the objective function.
        :type lamb: `float`
        :return: Derivative value.
        :rtype: `float`
        """
        # First compute the derivative of the first component of the loss function
        nor1 = np.sqrt(np.power(z_init - x.flatten(), 2).dot(mask.flatten()))
        if nor1 < 1e-6:
            nor1 = 1e-6
        der1 = ((z_init - x.flatten()) * mask.flatten()) / (nor1 * 1.0)

        # Then compute the derivative of the second component of the loss function
        z_init = np.reshape(z_init, x.shape)

        if norm == 1:
            z_d1 = np.sign(z_init[1:, :] - z_init[:-1, :])
            z_d2 = np.sign(z_init[:, 1:] - z_init[:, :-1])
        else:
            z_d1_norm = np.power(np.linalg.norm(z_init[1:, :] - z_init[:-1, :], norm, axis=1), norm - 1)
            z_d2_norm = np.power(np.linalg.norm(z_init[:, 1:] - z_init[:, :-1], norm, axis=0), norm - 1)
            z_d1_norm[z_d1_norm < 1e-6] = 1e-6
            z_d2_norm[z_d2_norm < 1e-6] = 1e-6
            z_d1_norm = np.repeat(z_d1_norm[:, np.newaxis], z_init.shape[1], axis=1)
            z_d2_norm = np.repeat(z_d2_norm[np.newaxis, :], z_init.shape[0], axis=0)
            z_d1 = norm * np.power(z_init[1:, :] - z_init[:-1, :], norm - 1) / z_d1_norm
            z_d2 = norm * np.power(z_init[:, 1:] - z_init[:, :-1], norm - 1) / z_d2_norm

        der2 = np.zeros(z_init.shape)
        der2[:-1, :] -= z_d1
        der2[1:, :] += z_d1
        der2[:, :-1] -= z_d2
        der2[:, 1:] += z_d2
        der2 = lamb * der2.flatten()

        # Total derivative
        return der1 + der2

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    def set_params(self, **kwargs):
        """
        Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.

        :param prob: Probability of the Bernoulli distribution.
        :type prob: `float`
        :param norm: The norm (positive integer).
        :type norm: `int`
        :param lamb: The lambda parameter in the objective function.
        :type lamb: `float`
        :param solver: Current support: `L-BFGS-B`, `CG`, `Newton-CG`.
        :type solver: `str`
        :param max_iter: Maximum number of iterations when performing optimization.
        :type max_iter: `int`
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :type clip_values: `tuple`
        """
        # Save defence-specific parameters
        super(VarianceMinimization, self).set_params(**kwargs)

        if not isinstance(self.prob, (float, int)) or self.prob < 0.0 or self.prob > 1.0:
            logger.error("Probability must be between 0 and 1.")
            raise ValueError("Probability must be between 0 and 1.")

        if not isinstance(self.norm, (int, np.int)) or self.norm <= 0:
            logger.error("Norm must be a positive integer.")
            raise ValueError("Norm must be a positive integer.")

        if not (self.solver == "L-BFGS-B" or self.solver == "CG" or self.solver == "Newton-CG"):
            logger.error("Current support only L-BFGS-B, CG, Newton-CG.")
            raise ValueError("Current support only L-BFGS-B, CG, Newton-CG.")

        if not isinstance(self.max_iter, (int, np.int)) or self.max_iter <= 0:
            logger.error("Number of iterations must be a positive integer.")
            raise ValueError("Number of iterations must be a positive integer.")

        if self.clip_values is not None:

            if len(self.clip_values) != 2:
                raise ValueError("`clip_values` should be a tuple of 2 floats containing the allowed data range.")

            if np.array(self.clip_values[0] >= self.clip_values[1]).any():
                raise ValueError("Invalid `clip_values`: min >= max.")

        return True
