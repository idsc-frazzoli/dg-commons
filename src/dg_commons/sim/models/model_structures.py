from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import inf
from typing import Sequence, NewType, TypeVar

from dg_commons import Color

__all__ = [
    "ModelType",
    "CAR",
    "MOTORCYCLE",
    "BICYCLE",
    "PEDESTRIAN",
    "TRUCK",
    "SPACECRAFT",
    "DYNAMIC_OBSTACLE",
    "ModelGeometry",
    "ModelParameters",
    "TModelGeometry",
    "TModelParameters",
    "TwoWheelsTypes",
    "FourWheelsTypes",
]

ModelType = NewType("ModelType", str)
CAR = ModelType("car")
TRUCK = ModelType("truck")
MOTORCYCLE = ModelType("motorcycle")
BICYCLE = ModelType("bicycle")
PEDESTRIAN = ModelType("pedestrian")
SPACECRAFT = ModelType("spacecraft")
ROCKET = ModelType("rocket")
DYNAMIC_OBSTACLE = ModelType("dynamic_obstacle")
TwoWheelsTypes = frozenset({BICYCLE, MOTORCYCLE})
FourWheelsTypes = frozenset({CAR, TRUCK})


@dataclass(frozen=True, unsafe_hash=True)
class ModelGeometry(ABC):
    m: float
    """ Vehicle Mass [kg] """
    Iz: float
    """ Moment of inertia (used only in the dynamic model) """
    e: float
    """ Restitution coefficient (used only in collisions energy transfer).
    Ratio of the differences in vehicle speeds before and after the collision -> 0 < e < 1"""
    color: Color
    """ Color must be able to be parsed by matplotlib"""

    @property
    @abstractmethod
    def outline(self) -> Sequence[tuple[float, float]]:
        pass


# from 3.11 can switch to Self
TModelGeometry = TypeVar("TModelGeometry", bound=ModelGeometry)


@dataclass(frozen=True, unsafe_hash=True)
class StaticModelGeometry(ModelGeometry):
    def outline(self) -> Sequence[tuple[float, float]]:
        raise NotImplementedError("Outline method for static model geometry is not implemented")

    @staticmethod
    def default() -> "StaticModelGeometry":
        return StaticModelGeometry(m=inf, Iz=inf, e=0.4, color="black")


@dataclass(frozen=True, unsafe_hash=True)
class ModelParameters:
    vx_limits: tuple[float, float]
    """ Minimum and Maximum velocities [m/s] """
    acc_limits: tuple[float, float]
    """ Minimum and Maximum acceleration [m/s^2] """

    def __post_init__(self):
        assert self.vx_limits[0] < self.vx_limits[1]
        assert self.acc_limits[0] < self.acc_limits[1]


# from 3.11 can switch to Self
TModelParameters = TypeVar("TModelParameters", bound=ModelParameters)
