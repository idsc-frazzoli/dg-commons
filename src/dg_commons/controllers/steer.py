import math
from dataclasses import dataclass
from typing import Tuple, Optional

from dg_commons.controllers.pid import PIDParam, PID
from dg_commons.sim.models.vehicle_utils import VehicleParameters


@dataclass
class SteerControllerParam(PIDParam):
    """Default values are tuned roughly for a default car model"""

    kP: float = 4
    kI: float = 0.1
    kD: float = 0.2
    antiwindup: Tuple[float, float] = (-0.5, 0.5)
    setpoint_minmax: Tuple[float, float] = (-math.pi / 6, math.pi / 6)
    output_minmax: Tuple[float, float] = (-1, 1)  # minmax steer derivative [rad/s]

    @classmethod
    def from_vehicle_params(cls, vehicle_param: VehicleParameters) -> "SteerControllerParam":
        return SteerControllerParam(
            setpoint_minmax=(-vehicle_param.delta_max, vehicle_param.delta_max),
            output_minmax=(-vehicle_param.ddelta_max, vehicle_param.ddelta_max),
        )


class SteerController(PID):
    """Low-level controller for reference tracking of steering angle"""

    def __init__(self, params: Optional[PIDParam] = None):
        params = SteerControllerParam() if params is None else params
        super(SteerController, self).__init__(params)

    @classmethod
    def from_vehicle_params(cls, vehicle_param: VehicleParameters) -> "SteerController":
        params = SteerControllerParam(
            setpoint_minmax=(-vehicle_param.delta_max, vehicle_param.delta_max),
            output_minmax=(-vehicle_param.ddelta_max, vehicle_param.ddelta_max),
        )
        return SteerController(params)
