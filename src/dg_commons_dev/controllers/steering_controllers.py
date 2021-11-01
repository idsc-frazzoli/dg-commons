from dataclasses import dataclass
from typing import Optional, Union, List
from dg_commons_dev.controllers.controller_types import SteeringController
from dg_commons_dev.utils import BaseParams


@dataclass
class SCIdentityParam(BaseParams):
    pass


class SCIdentity(SteeringController):
    def __init__(self, params: Optional[SCIdentityParam] = None):
        self.params = SCIdentityParam() if params is None else params
        super().__init__()

    def _get_steering_vel(self, current_steering: float) -> float:
        return self.delta_ref


@dataclass
class SCPParam(BaseParams):
    ddelta_kp: Union[List[float], float] = 10


class SCP(SteeringController):
    def __init__(self, params: Optional[SCPParam] = None):
        self.params = SCPParam() if params is None else params
        super().__init__()

    def _get_steering_vel(self, current_steering: float) -> float:
        return self.params.ddelta_kp * (self.delta_ref - current_steering)
