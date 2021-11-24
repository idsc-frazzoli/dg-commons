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
    c_rr_f: float
    """ Rolling Resistance coefficient front """
    c_rr_r: float
    """ Rolling Resistance coefficient rear """
    h_cog: float = 0.7
    """ Height of the CoG [m] """
    model_type: ModelType = QUADROTOR

    # todo fix default rotational inertia
    @classmethod
    def default(
        cls,
        color: Color = "royalblue",
        m=1500.0,
        Iz=1300,
        w_half=0.9,
        lf=1.7,
        lr=1.7,
    ) -> "QuadGeometry":
        return QuadGeometry(
            m=m,
            Iz=Iz,
            w_half=w_half,
            c_drag=0.3756,
            a_drag=2,
            e=0.0,
            c_rr_f=0.003,
            c_rr_r=0.003,
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
    delta_max: float
    """ Maximum steering angle [rad] """
    ddelta_max: float
    """ Minimum and Maximum steering rate [rad/s] """

    @classmethod
    def default_car(cls) -> "QuadParameters":
        # data from https://copradar.com/chapts/references/acceleration.html
        return QuadParameters(
            vx_limits=(kmh2ms(-10), kmh2ms(130)), acc_limits=(-8, 5), delta_max=math.pi / 6, ddelta_max=1
        )

    @classmethod
    def default(cls) -> "QuadParameters":
        return QuadParameters(
            vx_limits=(kmh2ms(-10), kmh2ms(90)), acc_limits=(-6, 3.5), delta_max=math.pi / 4, ddelta_max=1
        )

    def __post_init__(self):
        super(QuadParameters, self).__post_init__()
        assert self.delta_max > 0
        assert self.ddelta_max > 0
