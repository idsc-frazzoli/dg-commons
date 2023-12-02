from dataclasses import dataclass
from functools import cached_property
from typing import Self

import numpy as np
from shapely import Point
from shapely.geometry import Polygon

from dg_commons import Color
from dg_commons.sim import kmh2ms
from dg_commons.sim.models.model_structures import (
    ModelGeometry,
    ModelType,
    DIFF_DRIVE,
    ModelParameters,
)

__all__ = ["DiffDriveGeometry", "DiffDriveParameters"]


@dataclass(frozen=True, unsafe_hash=True)
class DiffDriveGeometry(ModelGeometry):
    """Geometry parameters of the differential drive robot (and colour)"""

    vehicle_type: ModelType
    """Type of the vehicle"""
    wheelbase: float
    """ Wheel axis [m] """
    wheelradius: float
    """ Wheel radius [m] """
    radius: float
    """ Radius determining the occupancy of the robot (Assume the diff drive has circular shape) [m] """

    @classmethod
    def default(
        cls,
        color: Color = "royalblue",
        m: float = 3,
        Iz: float = 5,
        wheelbase: float = 1,
        wheelradius: float = 0.1,
        radius: float = 0.6,
    ) -> Self:
        return DiffDriveGeometry(
            vehicle_type=DIFF_DRIVE,
            m=m,
            Iz=Iz,
            wheelbase=wheelbase,
            wheelradius=wheelradius,
            radius=radius,
            e=0.5,
            color=color,
        )

    @property
    def width(self) -> float:
        return self.radius * 2

    @property
    def length(self) -> float:
        return self.width

    @cached_property
    def outline(self) -> tuple[tuple[float, float], ...]:
        """Outline of the vehicle intended as the whole car body."""
        poly = self.outline_as_polygon
        return poly.exterior.coords.xy

    @cached_property
    def outline_as_polygon(self) -> Polygon:
        return Point(0, 0).buffer(self.radius)

    @cached_property
    def outline_as_polygon_wkt(self) -> str:
        return self.outline_as_polygon.wkt

    @property
    def wheel_shape(self) -> tuple[float, float]:
        # size of the wheels
        halfwidth, radius = self.wheelradius / 3, self.wheelradius
        return halfwidth, radius

    @cached_property
    def wheel_outline(self):
        halfwidth, radius = self.wheel_shape
        # fixme uniform points handling to native list of tuples
        return np.array(
            [
                [radius, -radius, -radius, radius, radius],
                [-halfwidth, -halfwidth, halfwidth, halfwidth, -halfwidth],
                [1, 1, 1, 1, 1],
            ]
        )

    @cached_property
    def wheels_position(self) -> np.ndarray:
        positions = np.array([[-self.wheelbase / 2, self.wheelbase / 2], [0, 0], [1, 1]])
        return positions

    @cached_property
    def n_wheels(self) -> int:
        return self.wheels_position.shape[1]


@dataclass(frozen=True, unsafe_hash=True)
class DiffDriveParameters(ModelParameters):
    omega_limits: tuple[float, float]
    """ min/max acceleration """

    @classmethod
    def default(cls) -> Self:
        return cls(
            vx_limits=(kmh2ms(-10), kmh2ms(10)),
            acc_limits=(-5, 5),
            omega_limits=(-5, 5),
        )

    def __post_init__(self):
        super().__post_init__()
        assert self.omega_limits[0] < self.omega_limits[1]
