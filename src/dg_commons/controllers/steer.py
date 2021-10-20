import math
from dataclasses import dataclass
from typing import Tuple, Optional

from dg_commons.controllers.pid import PIDParam, PID


@dataclass
class SteerControllerParam(PIDParam):
    """Default values are tuned roughly for a default car model"""

    kP: float = 4
    kI: float = 0.1
    kD: float = 0.2
    antiwindup: Tuple[float, float] = (-0.5, 0.5)
    setpoint_minmax: Tuple[float, float] = (-math.pi / 6, math.pi / 6)
    output_minmax: Tuple[float, float] = (-1, 1)  # minmax steer derivative [rad/s]


class SteerController(PID):
    """Low-level controller for reference tracking of steering angle"""

    def __init__(self, params: Optional[PIDParam] = None):
        params = SteerControllerParam() if params is None else params
        super(SteerController, self).__init__(params)
