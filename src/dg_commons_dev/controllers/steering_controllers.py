from dataclasses import dataclass
from typing import Optional, Union, List
from dg_commons_dev.controllers.controller_types import SteeringController
from dg_commons_dev.utils import BaseParams


class SCIdentity(SteeringController):
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


class SCP(SteeringController):
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
