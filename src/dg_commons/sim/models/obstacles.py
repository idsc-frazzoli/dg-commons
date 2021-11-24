from dataclasses import dataclass
from math import inf
from typing import Sequence, Tuple

from shapely.geometry.base import BaseGeometry

from dg_commons import Color
from dg_commons.sim.models import ModelGeometry


@dataclass(frozen=True, unsafe_hash=True)
class ObstacleGeometry(ModelGeometry):
    m: float = inf
    """ Vehicle Mass [kg] """
    Iz: float = inf
    """ Moment of inertia (used only in the dynamic model) """
    e: float = 0.1
    """ Restitution coefficient (used only in collisions energy transfer).
    Ratio of the differences in vehicle speeds before and after the collision -> 0 < e < 1"""
    color: Color = "darkorchid"
    """ Color must be able to be parsed by matplotlib"""

    @property
    def outline(self) -> Sequence[Tuple[float, float]]:
        raise NotImplementedError("Outline is not implement for ObstacleGeometry")


@dataclass
class StaticObstacle:
    shape: BaseGeometry
    """ Shapely geometry """
    geometry: ObstacleGeometry = ObstacleGeometry()
    """ Geometry of the obstacle """
