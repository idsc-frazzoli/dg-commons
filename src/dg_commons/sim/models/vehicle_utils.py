import math
from dataclasses import dataclass

from dg_commons.sim import logger
from dg_commons.sim.models.model_structures import ModelParameters
from dg_commons.sim.models.utils import kmh2ms


@dataclass(frozen=True, unsafe_hash=True)
class VehicleParameters(ModelParameters):
    delta_max: float
    """ Maximum steering angle [rad] """
    ddelta_max: float
    """ Minimum and Maximum steering rate [rad/s] """

    @classmethod
    def default_car(cls) -> "VehicleParameters":
        # data from https://copradar.com/chapts/references/acceleration.html
        return VehicleParameters(
            vx_limits=(kmh2ms(-10), kmh2ms(130)), acc_limits=(-8, 5), delta_max=math.pi / 6, ddelta_max=1
        )

    @classmethod
    def default_truck(cls) -> "VehicleParameters":
        return VehicleParameters(
            vx_limits=(kmh2ms(-10), kmh2ms(90)), acc_limits=(-6, 3.5), delta_max=math.pi / 4, ddelta_max=1
        )

    @classmethod
    def default_bicycle(cls) -> "VehicleParameters":
        return VehicleParameters(
            vx_limits=(kmh2ms(-1), kmh2ms(50)), acc_limits=(-4, 3), delta_max=math.pi / 6, ddelta_max=1
        )

    def __post_init__(self):
        super(VehicleParameters, self).__post_init__()
        assert self.delta_max > 0
        assert self.ddelta_max > 0


def steering_constraint(steering_angle: float, steering_velocity: float, vp: VehicleParameters):
    """Enforces steering limits"""
    if (steering_angle <= -vp.delta_max and steering_velocity < 0) or (
        steering_angle >= vp.delta_max and steering_velocity > 0
    ):
        steering_velocity = 0
        logger.warn(
            f"Reached max steering boundaries:\n angle:{steering_angle:.2f}\tlimits:[{-vp.delta_max:.2f},{vp.delta_max:.2f}]"
        )
    elif steering_velocity < -vp.ddelta_max:
        logger.warn(
            f"Commanded steering rate out of limits, clipping value: {steering_velocity:.2f}<{-vp.ddelta_max:.2f}"
        )
        steering_velocity = -vp.ddelta_max
    elif steering_velocity > vp.ddelta_max:
        logger.warn(
            f"Commanded steering rate out of limits, clipping value: {steering_velocity:.2f}>{vp.ddelta_max:.2f}"
        )
        steering_velocity = vp.ddelta_max
    return steering_velocity
