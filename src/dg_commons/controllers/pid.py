from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from dg_commons import logger


@dataclass
class PIDParam:
    kP: float
    kI: float
    kD: float
    antiwindup: Tuple[float, float] = (-1, 1)
    setpoint_minmax: Tuple[float, float] = (-1, 1)
    output_minmax: Tuple[float, float] = (-1, 1)

    def __post_init__(self):
        assert self.antiwindup[0] < self.antiwindup[1]
        assert self.setpoint_minmax[0] < self.setpoint_minmax[1]
        assert self.output_minmax[0] < self.output_minmax[1]


class PID:
    """PID controller for reference tracking"""

    def __init__(self, params: PIDParam):
        self.params = params
        self.measurement: float = 0
        self.reference: float = 0
        self.last_request_at: Optional[float] = None
        self.last_integral_error: float = 0
        self.last_proportional_error: float = 0

    def update_measurement(self, measurement: float):
        self.measurement = measurement

    def update_reference(self, reference: float):
        if not self.params.setpoint_minmax[0] <= reference <= self.params.setpoint_minmax[1]:
            logger.warn(f"PID-controller: Desired ref {reference} out of range {self.params.setpoint_minmax}.")
        self.reference = np.clip(reference, self.params.setpoint_minmax[0], self.params.setpoint_minmax[1])

    def get_control(self, at: float) -> float:
        """A simple digital PID"""
        dt = 0 if self.last_request_at is None else at - self.last_request_at
        self.last_request_at = at
        p_error = self.reference - self.measurement
        self.last_integral_error += self.params.kI * p_error * dt
        self.last_integral_error = np.clip(
            self.last_integral_error, self.params.antiwindup[0], self.params.antiwindup[1]
        )
        # d part
        d_error = (p_error - self.last_proportional_error) / dt if not dt == 0 else 0
        self.last_proportional_error = p_error

        output = self.params.kP * p_error + self.last_integral_error + self.params.kD * d_error
        return float(np.clip(output, self.params.output_minmax[0], self.params.output_minmax[1]))
