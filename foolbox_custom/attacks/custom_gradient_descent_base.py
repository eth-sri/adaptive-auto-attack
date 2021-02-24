from typing import Union, Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
import eagerpy as ep

from ..devutils import flatten
from ..devutils import atleast_kd

from ..types import Bounds

from ..models.base import Model

from attack.criteria import Misclassification, TargetedMisclassification, Criterion

from ..distances import l1, l2, linf

from .base import FixedEpsilonAttack
from .base import T
from .base import get_criterion
from .base import raise_if_kwargs


def get_is_adversarial(
    criterion: Criterion, model: Model
) -> Callable[[ep.Tensor], ep.Tensor]:
    def is_adversarial(perturbed: ep.Tensor) -> ep.Tensor:
        outputs = model(perturbed)
        return criterion(perturbed, outputs)
    return is_adversarial


class CustomBaseGradientDescent(FixedEpsilonAttack, ABC):
    def __init__(
        self,
        *,
        rel_stepsize: float,
        param_dict: dict = None,
        abs_stepsize: Optional[float] = None,
        steps: int,
        random_start: bool,
        EOT: int = 1,
        nes: bool = False,
        parallel: bool = True,  # NES whether to parallelize over batch (False) or samples (True)
        loss: str = 'ce',
    ):
        self.rel_stepsize = rel_stepsize
        self.param_dict = param_dict or {}
        self.abs_stepsize = abs_stepsize
        self.stepsize = None
        self.steps = steps
        self.random_start = random_start
        self.EOT = EOT
        assert(EOT > 0), "EOT value has to be at least 1"
        self.nes = nes
        self.parallel = parallel
        self.loss = loss

    def get_loss_fn(
        self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        # can be overridden by users
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            logits = model(inputs)
            return ep.crossentropy(logits, labels).sum()

        return loss_fn

    def get_dlr_loss_fn(
        self, model: Model, labels: ep.Tensor
    ) -> Callable[[ep.Tensor], ep.Tensor]:
        def loss_fn(inputs: ep.Tensor) -> ep.Tensor:
            import numpy as np
            x = model(inputs.raw)
            x_sorted, ind_sorted = x.sort(dim=1)
            ind = (ind_sorted[:, -1] == labels.raw).float()
            result = -(x[np.arange(x.shape[0]), labels.raw] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (
                        x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
            return ep.astensor(result.sum())
        return loss_fn

    def get_stepsize_fn(self):
        end = self.rel_stepsize * self.param_dict["rel_slope"]
        freq = self.param_dict["freq"]
        interval = (self.rel_stepsize - end) * freq / (self.steps)

        def get_stepsize(i):
            return self.rel_stepsize - (i // freq) * interval
        return get_stepsize

    def value_and_grad(
        # can be overridden by users
        self,
        loss_fn: Callable[[ep.Tensor], ep.Tensor],
        x: ep.Tensor,
    ) -> Tuple[ep.Tensor, ep.Tensor]:
        return ep.value_and_grad(loss_fn, x)

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        if isinstance(criterion_, Misclassification):
            labels = criterion_.labels
        elif isinstance(criterion_, TargetedMisclassification):
            labels = criterion_.target_classes
        else:
            raise ValueError("unsupported criterion")

        if self.loss == 'ce':
            loss_fn = self.get_loss_fn(model, labels)
        elif self.loss == 'dlr':
            loss_fn = self.get_dlr_loss_fn(model, labels)
        else:
            assert False, "Unrecognized loss function"

        if self.abs_stepsize is None:
            self.stepsize = self.rel_stepsize * epsilon
        else:
            self.stepsize = self.abs_stepsize

        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model.bounds)
        else:
            x = x0

        for i in range(self.steps):
            _, mean_gradients = self.value_and_grad(loss_fn, x)
            # loss_val = loss_fn(x)
            for n in range(2, self.EOT + 1):
                """
                Computes numerically stable mean:
                \mu_n = (1 / n) * sum_{x=1}^n (x_i)
                      = (1 / n) * (x_n + sum_{x=1}^{n-1} (x_i))
                      = (1 / n) * (x_n + (n - 1) \mu_{n-1})
                      = \mu_{n-1} + (1 / n) * (x_n - \mu_{n-1})
                """
                _, gradients = self.value_and_grad(loss_fn, x)
                mean_gradients = mean_gradients + (gradients - mean_gradients) / n
            mean_gradients = self.normalize(mean_gradients, x=x, bounds=model.bounds)

            get_stepsize = self.get_stepsize_fn()
            stepsize = get_stepsize(i)
            if isinstance(criterion_, Misclassification):
                # step away from the original label
                x = x + stepsize * mean_gradients
            else:
                # step towards the target label
                x = x - stepsize * mean_gradients
            x = self.project(x, x0, epsilon)
            x = ep.clip(x, *model.bounds)

        return restore_type(x)

    @abstractmethod
    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        ...

    @abstractmethod
    def normalize(
        self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        ...

    @abstractmethod
    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        ...


def clip_lp_norms(x: ep.Tensor, *, norm: float, p: float) -> ep.Tensor:
    assert 0 < p < ep.inf
    norms = flatten(x).norms.lp(p=p, axis=-1)
    norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
    factor = ep.minimum(1, norm / norms)  # clipping -> decreasing but not increasing
    factor = atleast_kd(factor, x.ndim)
    return x * factor


def normalize_lp_norms(x: ep.Tensor, *, p: float) -> ep.Tensor:
    assert 0 < p < ep.inf
    norms = flatten(x).norms.lp(p=p, axis=-1)
    norms = ep.maximum(norms, 1e-12)  # avoid divsion by zero
    factor = 1 / norms
    factor = atleast_kd(factor, x.ndim)
    return x * factor


def uniform_l1_n_balls(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    # https://mathoverflow.net/a/9188
    u = ep.uniform(dummy, (batch_size, n))
    v = u.sort(axis=-1)
    vp = ep.concatenate([ep.zeros(v, (batch_size, 1)), v[:, : n - 1]], axis=-1)
    assert v.shape == vp.shape
    x = v - vp
    sign = ep.uniform(dummy, (batch_size, n), low=-1.0, high=1.0).sign()
    return sign * x


def uniform_l2_n_spheres(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    x = ep.normal(dummy, (batch_size, n + 1))
    r = x.norms.l2(axis=-1, keepdims=True)
    s = x / r
    return s


def uniform_l2_n_balls(dummy: ep.Tensor, batch_size: int, n: int) -> ep.Tensor:
    """Sampling from the n-ball

    Implementation of the algorithm proposed by Voelker et al. [#Voel17]_

    References:
        .. [#Voel17] Voelker et al., 2017, Efficiently sampling vectors and coordinates
            from the n-sphere and n-ball
            http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """
    s = uniform_l2_n_spheres(dummy, batch_size, n + 1)
    b = s[:, :n]
    return b


class CustomL1BaseGradientDescent(CustomBaseGradientDescent):
    distance = l1

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        batch_size, n = flatten(x0).shape
        r = uniform_l1_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def normalize(
        self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        return normalize_lp_norms(gradients, p=1)

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + clip_lp_norms(x - x0, norm=epsilon, p=1)


class CustomL2BaseGradientDescent(CustomBaseGradientDescent):
    distance = l2

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        batch_size, n = flatten(x0).shape
        r = uniform_l2_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + epsilon * r

    def normalize(
        self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        return normalize_lp_norms(gradients, p=2)

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + clip_lp_norms(x - x0, norm=epsilon, p=2)


class CustomLinfBaseGradientDescent(CustomBaseGradientDescent):
    distance = linf

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + ep.uniform(x0, x0.shape, -epsilon, epsilon)

    def normalize(
        self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        return gradients.sign()

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        return x0 + ep.clip(x - x0, -epsilon, epsilon)
