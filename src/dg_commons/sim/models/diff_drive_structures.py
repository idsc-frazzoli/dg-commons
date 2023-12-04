from dataclasses import dataclass
from functools import cached_property

import numpy as np
from geometry import SE2_from_xytheta
from shapely import Point
from shapely.geometry import Polygon

from dg_commons import Color
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
        wheelradius: float = 0.2,
        radius: float = 0.6,
    ) -> "DiffDriveGeometry":
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
        return tuple(zip(poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]))

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
        half_width, r = self.wheel_shape
        # fixme uniform points handling to native list of tuples
        return np.array(
            [
                [r, -r, -r, r, r],
                [-half_width, -half_width, half_width, half_width, -half_width],
                [1, 1, 1, 1, 1],
            ]
        )

    @cached_property
    def wheels_position(self) -> np.ndarray:
        positions = np.array([[0, 0], [-self.wheelbase / 2, self.wheelbase / 2], [1, 1]])
        return positions

    @cached_property
    def n_wheels(self) -> int:
        return self.wheels_position.shape[1]

    @cached_property
    def wheels_outlines(self) -> list[np.ndarray]:
        """Returns a list of wheel outlines in the model frame"""
        wheels_position = self.wheels_position
        assert self.n_wheels == 2
        transformed_wheels_outlines = []
        for i in range(self.n_wheels):
            transform = SE2_from_xytheta((wheels_position[0, i], wheels_position[1, i], 0))
            transformed_wheels_outlines.append(transform @ self.wheel_outline)
        return transformed_wheels_outlines


@dataclass(frozen=True, unsafe_hash=True)
class DiffDriveParameters(ModelParameters):
    omega_limits: tuple[float, float]
    """ min/max rotational velocity of wheels [rad/s] """

    @classmethod
    def default(
        cls,
        omega_limits: tuple[float, float] = (-5, 5),
    ) -> "DiffDriveParameters":
        """vx, and acc are irrelevant for the diff drive model"""

        return cls(omega_limits=omega_limits, vx_limits=(0, 1), acc_limits=(0, 1))

    def __post_init__(self):
        super().__post_init__()
        assert self.omega_limits[0] < self.omega_limits[1]
