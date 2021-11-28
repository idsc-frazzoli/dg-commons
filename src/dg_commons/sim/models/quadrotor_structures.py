import math
from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

from shapely.geometry import Polygon

from dg_commons import Color
from dg_commons.sim.models import kmh2ms
from dg_commons.sim.models.model_structures import (
    ModelGeometry,
    ModelParameters,
    QUADROTOR,
    ModelType,
)

__all__ = ["QuadGeometry", "QuadParameters"]


@dataclass(frozen=True, unsafe_hash=True)
class QuadGeometry(ModelGeometry):
    """Geometry parameters of the vehicle (and colour)"""

    w_half: float
    """ Half width of the drone - distance from CoG to end of rotor [m] """
    c_drag: float
    """ Drag coefficient """
    a_drag: float
    """ Section Area interested by drag """
    h_cog: float = 0.7
    """ Height of the CoG [m] """
    model_type: ModelType = QUADROTOR

    # todo fix default rotational inertia
    @classmethod
    def default(
        cls,
        color: Color = "royalblue",
        m=10.0,
        Iz=100,
        w_half=0.9,
    ) -> "QuadGeometry":
        return QuadGeometry(
            m=m,
            Iz=Iz,
            w_half=w_half,
            c_drag=0.3756,
            a_drag=2,
            e=0.0,
            color=color,
        )

    @cached_property
    def width(self):
        return self.w_half * 2

    @cached_property
    def outline(self) -> Tuple[Tuple[float, float], ...]:
        """Outline of the vehicle intended as the whole car body."""
        return (
            (self.width, 0),
            (0, self.width),
            (-self.width, 0),
            (0, -self.width),
            (self.width, 0),
        )

    @cached_property
    def outline_as_polygon(self) -> Polygon:
        return Polygon(self.outline)


@dataclass(frozen=True, unsafe_hash=True)
class QuadParameters(ModelParameters):
    dpsi_max: float
    """ Maximum yaw rate [rad/s] """
    ddpsi_max: float
    """ Maximum derivative of yaw rate [rad/s^2] """

    @classmethod
    def default(cls) -> "QuadParameters":
        return QuadParameters(
            vx_limits=(kmh2ms(-20), kmh2ms(20)), acc_limits=(-5, 5), dpsi_max=2 * math.pi, ddpsi_max=1
        )

    def __post_init__(self):
        super(QuadParameters, self).__post_init__()
        assert self.dpsi_max > 0
        assert self.ddpsi_max > 0
