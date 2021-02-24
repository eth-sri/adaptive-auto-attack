from typing import Optional

from .custom_gradient_descent_base import CustomL1BaseGradientDescent
from .custom_gradient_descent_base import CustomL2BaseGradientDescent
from .custom_gradient_descent_base import CustomLinfBaseGradientDescent


class CustomL1ProjectedGradientDescentAttack(CustomL1BaseGradientDescent):
    """L1 Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        param_dict: dict = None,
        steps: int = 50,
        random_start: bool = True,
        EOT: int = 1,
        loss: str = 'ce',
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            param_dict=param_dict,
            steps=steps,
            random_start=random_start,
            EOT=EOT,
            loss=loss
        )


class CustomL2ProjectedGradientDescentAttack(CustomL2BaseGradientDescent):
    """L2 Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon (defaults to 0.01 / 0.3).
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        param_dict: dict = None,
        steps: int = 50,
        random_start: bool = True,
        EOT: int = 1,
        loss: str = 'ce',
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            param_dict=param_dict,
            steps=steps,
            random_start=random_start,
            EOT=EOT,
            loss=loss,
        )


class CustomLinfProjectedGradientDescentAttack(CustomLinfBaseGradientDescent):
    """Linf Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon (defaults to 0.01 / 0.3).
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps : Number of update steps to perform.
        random_start : Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.01 / 0.3,
        abs_stepsize: Optional[float] = None,
        param_dict: dict = None,
        steps: int = 40,
        random_start: bool = True,
        EOT: int = 1,
        loss: str = 'ce',
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            param_dict=param_dict,
            steps=steps,
            random_start=random_start,
            EOT=EOT,
            loss=loss,
        )
