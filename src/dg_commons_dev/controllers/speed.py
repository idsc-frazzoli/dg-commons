from dataclasses import dataclass
from typing import Tuple, Optional, Callable
from dg_commons.controllers.pid import PIDParam, PID
from dg_commons.sim.models import kmh2ms
from dg_commons_dev.controllers.controller_types import LongitudinalController
from dg_commons import X
from dg_commons_dev.utils import BaseParams

__all__ = ["SpeedControllerParam", "SpeedController"]


@dataclass
class SpeedControllerParam(BaseParams, PIDParam):
    """Default values are tuned roughly for a default car model"""
    kP: float = 4
    kI: float = 0.01
    kD: float = 0.1
    antiwindup: Tuple[float, float] = (-2, 2)
    setpoint_minmax: Tuple[float, float] = (-kmh2ms(10), kmh2ms(150))
    output_minmax: Tuple[float, float] = (-8, 5)  # acc minmax

    def __post_init__(self):
        assert self.antiwindup[0] < self.antiwindup[1]
        assert self.setpoint_minmax[0] < self.setpoint_minmax[1]
        assert self.output_minmax[0] < self.output_minmax[1]
        assert 0 <= self.kP
        assert 0 <= self.kI
        assert 0 <= self.kD
        super().__post_init__()


class SpeedController(PID, LongitudinalController):
    """ Low-level controller for reference tracking of speed """
    REF_PARAMS: Callable = SpeedControllerParam

    def __init__(self, params: Optional[PIDParam] = None):
        params = SpeedControllerParam() if params is None else params
        super(SpeedController, self).__init__(params)

    def _update_obs(self, new_obs: X):
        """
        A new observation is processed and an input for the system formulated
        @param new_obs: New Observation
        """
        self.update_measurement(new_obs.vx)

    def _get_acceleration(self, at: float) -> float:
        """
        @param at: current time instant
        @return: desired acceleration
        """
        self.update_reference(self.speed_ref)
        return self.get_control(at)
