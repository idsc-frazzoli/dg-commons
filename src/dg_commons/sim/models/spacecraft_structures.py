import math
from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

from dg_commons import Color
from dg_commons.sim.models import kmh2ms
from dg_commons.sim.models.model_structures import (
    ModelGeometry,
    ModelParameters,
    SPACECRAFT,
    ModelType,
)

__all__ = ["SpacecraftGeometry", "SpacecraftParameters"]


@dataclass(frozen=True, unsafe_hash=True)
class SpacecraftGeometry(ModelGeometry):
    """Geometry parameters of the vehicle (and colour)"""

    w_half: float
    """ Half width of the drone - distance from CoG to end of rotor [m] """
    lf: float
    """ Front length of vehicle - dist from CoG to front axle [m] """
    lr: float
    """ Rear length of vehicle - dist from CoG to back axle [m] """
    model_type: ModelType = SPACECRAFT

    # todo fix default rotational inertia
    @classmethod
    def default(
        cls,
        color: Color = "royalblue",
        m=50,
        Iz=100,
        w_half=2,
        lf=3,
        lr=2,
    ) -> "SpacecraftGeometry":
        return SpacecraftGeometry(
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
    def outline(self) -> Tuple[Tuple[float, float], ...]:
        """Outline of the vehicle intended as the whole car body."""
        cog2cup = self.lf / 4
        circle = Point(cog2cup, 0).buffer(self.w_half)
        ellipse = affinity.scale(circle, 1, 1.5)
        rect = Polygon(
            [(-self.lr, self.w_half), (-self.lr, -self.w_half), (cog2cup, -self.w_half), (cog2cup, self.w_half)]
        )
        spacecraft_poly = unary_union([ellipse, rect])
        return tuple(spacecraft_poly.exterior.coords)

    @cached_property
    def outline_as_polygon(self) -> Polygon:
        return Polygon(self.outline)

    @cached_property
    def trusther_shape(self):
        halfwidth, radius = 0.1, 0.3  # size of the wheels
        return halfwidth, radius

    @cached_property
    def wheel_outline(self):
        halfwidth, radius = self.trusther_shape
        return np.array(
            [
                [radius, -radius, -radius, radius, radius],
                [-halfwidth, -halfwidth, halfwidth, halfwidth, -halfwidth],
                [1, 1, 1, 1, 1],
            ]
        )

    @cached_property
    def wheels_position(self) -> np.ndarray:
        positions = np.array([[self.lf, -self.lr], [0, 0], [1, 1]])
        return positions


@dataclass(frozen=True, unsafe_hash=True)
class SpacecraftParameters(ModelParameters):
    dpsi_max: float
    """ Maximum yaw rate [rad/s] """

    @classmethod
    def default(cls) -> "SpacecraftParameters":
        return SpacecraftParameters(vx_limits=(kmh2ms(-50), kmh2ms(50)), acc_limits=(-10, 10), dpsi_max=2 * math.pi)

    def __post_init__(self):
        super(SpacecraftParameters, self).__post_init__()
        assert self.dpsi_max > 0
