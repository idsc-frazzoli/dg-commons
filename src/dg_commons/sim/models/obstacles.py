from dataclasses import dataclass
from math import inf
from typing import Sequence, Tuple

from commonroad.scenario.obstacle import ObstacleType
from shapely.geometry.base import BaseGeometry

from dg_commons import Color
from dg_commons.sim.models import ModelGeometry, ModelParameters


@dataclass(frozen=True, unsafe_hash=True)
class ObstacleGeometry(ModelGeometry):
    m: float
    """ Mass [kg] """
    Iz: float
    """ Moment of inertia (used only in the dynamic model) """
    e: float
    """ Restitution coefficient (used only in collisions energy transfer).
    Ratio of the differences in vehicle speeds before and after the collision -> 0 < e < 1 """
    color: Color = "green"
    """ Color must be able to be parsed by matplotlib"""

    @property
    def outline(self) -> Sequence[Tuple[float, float]]:
        raise NotImplementedError("Outline is not implement for ObstacleGeometry")

    @classmethod
    def default_static(
        cls, m: float = inf, Iz: float = inf, e: float = 0.1, color: Color = "darkorchid"
    ) -> "ObstacleGeometry":
        return ObstacleGeometry(m, Iz, e, color)


@dataclass
class StaticObstacle:
    shape: BaseGeometry
    """ Shapely geometry """
    geometry: ObstacleGeometry = ObstacleGeometry.default_static()
    """ Geometry of the obstacle """
    obstacle_type: ObstacleType = ObstacleType.UNKNOWN


@dataclass(frozen=True, unsafe_hash=True)
class DynObstacleParameters(ModelParameters):
    dpsi_limits: Tuple[float, float] = (-2, 2)
    """ Limits of the rotational speed [rad/s] """
    ddpsi_limits: Tuple[float, float] = (-1, 1)
    """ Limits of the rotational acceleration [rad/s^2] """

    def __post_init__(self):
        super(DynObstacleParameters, self).__post_init__()
        assert self.dpsi_limits[0] < self.dpsi_limits[1]
        assert self.ddpsi_limits[0] < self.ddpsi_limits[1]
