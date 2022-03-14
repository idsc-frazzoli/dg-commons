from dataclasses import dataclass
from typing import Optional, Callable
from dg_commons_dev.controllers.controller_types import SteeringController
from dg_commons_dev.utils import BaseParams


class SCIdentity(SteeringController):
    REF_PARAMS: Callable = BaseParams

    def __init__(self, params: Optional[BaseParams] = None):
        self.params = BaseParams() if params is None else params
        super().__init__()

    def _get_steering_vel(self, current_steering: float) -> float:
        """
        Returns delta ref itself.
        @param current_steering: Current steering angle
        @return: Desired steering velocity
        """
        return self.delta_ref


@dataclass
class SCPParam(BaseParams):
    """
    Parameters of P - steering controller
    """

    ddelta_kp: float = 10
    """ Kappa p - parameter """

    def __post_init__(self):
        assert 0 <= self.ddelta_kp


class SCP(SteeringController):
    REF_PARAMS: Callable = SCPParam

    def __init__(self, params: Optional[SCPParam] = None):
        self.params = SCPParam() if params is None else params
        super().__init__()

    def _get_steering_vel(self, current_steering: float) -> float:
        """
        Computes steering velocity with a p-controller
        @param current_steering: Current steering angle
        @return: Desired steering velocity
        """
        return self.params.ddelta_kp * (self.delta_ref - current_steering)
