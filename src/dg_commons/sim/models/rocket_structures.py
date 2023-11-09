import math
from dataclasses import dataclass
from functools import cached_property

from geometry import SE2value, SE2_from_xytheta
from shapely import affinity
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from dg_commons import Color, transform_xy
from dg_commons.sim.models import kmh2ms
from dg_commons.sim.models.model_structures import (
    ModelGeometry,
    ModelParameters,
    SPACECRAFT,
    ModelType,
)

__all__ = ["RocketGeometry", "RocketParameters"]


# todo to review all


@dataclass(frozen=True, unsafe_hash=True)
class RocketGeometry(ModelGeometry):
    """Geometry parameters of the rocket (and colour)"""

    w_half: float
    """ Half width of the rocket - half width of the rocket [m] """
    lf: float
    """ Front length of rocket - dist from CoG to front axle [m] """
    lr: float
    """ Rear length of rocket - dist from CoG to back axle [m] """
    model_type: ModelType = SPACECRAFT

    def __post_init__(self):
        assert self.lr > self.lf

    @classmethod
    def default(
        cls,
        color: Color = "royalblue",
        m=50,  # MASS TO BE INTENDED AS MASS OF THE ROCKET WITHOUT FUEL
        Iz=100,
        w_half=1,
        lf=2,
        lr=3,
    ) -> "RocketGeometry":
        return RocketGeometry(
            m=m,
            Iz=Iz,
            w_half=w_half,
            lf=lf,
            lr=lr,
            e=0.7,
            color=color,
        )

    @cached_property
    def width(self):
        return self.w_half * 2

    @cached_property
    def outline(self) -> tuple[tuple[float, float], ...]:
        """
        Outline of the rocket. The outline is made by the union of an ellipse and a rectangle.
        The cog is at the end of the rectangle (circle center).
        """
        circle = Point(0, 0).buffer(self.w_half)
        ellipse = affinity.scale(circle, self.lf / self.w_half, 1)
        rect = Polygon(
            [
                (-self.lr, self.w_half),
                (-self.lr, -self.w_half),
                (1e-3, -self.w_half),
                (1e-3, self.w_half),
                (-self.lr, self.w_half),
            ]
        )
        rocket_poly = unary_union([ellipse, rect])
        return tuple(rocket_poly.exterior.coords)

    @cached_property
    def outline_as_polygon(self) -> Polygon:
        return Polygon(self.outline)

    @property
    def n_thrusters(self) -> int:
        return 2

    @property
    def thruster_shape(self):
        w_half, l_half = 0.15, 0.5
        return w_half, l_half

    @property
    def thruster_outline(self) -> tuple[tuple[float, float], ...]:
        w_half, l_half = self.thruster_shape
        return (l_half, -w_half), (-l_half, -w_half), (-l_half, w_half), (l_half, w_half), (l_half, -w_half)

    @property
    def thrusters_position(self) -> list[SE2value]:
        positions = [SE2_from_xytheta((-self.lr, -self.w_half, 0)), SE2_from_xytheta((-self.lr, self.w_half, 0))]
        return positions

    @cached_property
    def thrusters_outline_in_body_frame(self) -> list[tuple[tuple[float, float], ...]]:
        thrusters_outline = [transform_xy(q, self.thruster_outline) for q in self.thrusters_position]
        return thrusters_outline


@dataclass(frozen=True, unsafe_hash=True)
class RocketParameters(ModelParameters):
    dpsi_limits: tuple[float, float]
    """ Maximum yaw rate [rad/s] """

    @classmethod
    def default(cls) -> "RocketParameters":
        return RocketParameters(
            vx_limits=(kmh2ms(-50), kmh2ms(50)), acc_limits=(-10, 10), dpsi_limits=(-2 * math.pi, 2 * math.pi)
        )

    def __post_init__(self):
        super(RocketParameters, self).__post_init__()
        assert self.dpsi_limits[0] < self.dpsi_limits[1]
