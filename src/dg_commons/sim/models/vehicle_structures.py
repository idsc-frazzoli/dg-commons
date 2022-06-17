from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, List, Mapping

import numpy as np
from geometry import SE2_from_xytheta
from shapely.geometry import Polygon
from zuper_commons.types import ZValueError

from dg_commons import Color
from dg_commons.sim.models.model_structures import (
    ModelGeometry,
    ModelType,
    CAR,
    BICYCLE,
    MOTORCYCLE,
    TRUCK,
    FourWheelsTypes,
)

__all__ = ["VehicleGeometry"]


@dataclass(frozen=True, unsafe_hash=True)
class VehicleGeometry(ModelGeometry):
    """Geometry parameters of the vehicle (and colour)"""

    vehicle_type: ModelType
    """Type of the vehicle"""
    w_half: float
    """ Half width of vehicle (between center of wheels) [m] """
    lf: float
    """ Front length of wheelbase - dist from CoG to front axle [m] """
    lr: float
    """ Rear length of wheelbase - dist from CoG to back axle [m] """
    c_drag: float
    """ Drag coefficient """
    a_drag: float
    """ Section Area interested by drag """
    c_rr_f: float = 0.003
    """ Rolling Resistance coefficient front """
    c_rr_r: float = 0.003
    """ Rolling Resistance coefficient rear """
    h_cog: float = 0.7
    """ Height of the CoG [m] """

    # todo fix default rotational inertia
    @classmethod
    def default_car(
        cls,
        color: Color = "royalblue",
        m=1300.0,
        Iz=1600,
        w_half=0.8,
        lf=1.2,
        lr=1.2,
    ) -> "VehicleGeometry":
        """Numbers roughly gotten from:
        https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/blob/master/vehicleModels_commonRoad.pdf"""
        return VehicleGeometry(
            vehicle_type=CAR,
            m=m,
            Iz=Iz,
            w_half=w_half,
            lf=lf,
            lr=lr,
            c_drag=0.3756,
            a_drag=2,
            e=0.5,
            color=color,
        )

    @classmethod
    def default_bicycle(
        cls,
        color: Color = "saddlebrown",
        m=85.0,
        Iz=90,
        w_half=0.25,
        lf=1.0,
        lr=1.0,
    ) -> "VehicleGeometry":
        return VehicleGeometry(
            vehicle_type=BICYCLE,
            m=m,
            Iz=Iz,
            w_half=w_half,
            lf=lf,
            lr=lr,
            c_drag=0.01,
            a_drag=0.2,
            e=0.35,
            color=color,
        )

    @classmethod
    def default_truck(cls, color: Color = "darkgreen", m=8000.0, Iz=6300, w_half=1.2, lf=4, lr=4) -> "VehicleGeometry":
        return VehicleGeometry(
            vehicle_type=TRUCK,
            m=m,
            Iz=Iz,
            w_half=w_half,
            lf=lf,
            lr=lr,
            c_drag=0.3756,
            a_drag=4,
            e=0.5,
            color=color,
        )

    @cached_property
    def width(self) -> float:
        return self.w_half * 2

    @cached_property
    def length(self) -> float:
        return self.wheelbase + sum(self.bumpers_length)

    @cached_property
    def wheelbase(self) -> float:
        """Length between the two wheel axles, it does not consider bumpers etc..."""
        return self.lf + self.lr

    @cached_property
    def outline(self) -> Tuple[Tuple[float, float], ...]:
        """Outline of the vehicle intended as the whole car body."""
        tyre_halfw, _ = self.wheel_shape
        frontbumper, backbumper = self.bumpers_length
        return (
            (-self.lr - backbumper, -self.w_half - tyre_halfw),
            (-self.lr - backbumper, +self.w_half + tyre_halfw),
            (+self.lf + frontbumper, +self.w_half + tyre_halfw),
            (+self.lf + frontbumper, -self.w_half - tyre_halfw),
            (-self.lr - backbumper, -self.w_half - tyre_halfw),
        )

    @cached_property
    def bumpers_length(self) -> Tuple[float, float]:
        """Returns size of bumpers from wheels' axle to border
        @:return: (front,back)"""
        tyre_halfw, radius = self.wheel_shape
        if self.vehicle_type in FourWheelsTypes:
            frontbumper = self.lf / 2
        else:  # self.vehicle_type == MOTORCYCLE or self.vehicle_type == BICYCLE
            frontbumper = radius
        return frontbumper, radius * 2

    @cached_property
    def outline_as_polygon(self) -> Polygon:
        return Polygon(self.outline)

    @cached_property
    def outline_as_polygon_wkt(self) -> str:
        return self.outline_as_polygon.wkt

    @cached_property
    def wheel_shape(self):
        if self.vehicle_type == CAR:
            halfwidth, radius = 0.1, 0.3  # size of the wheels
        elif self.vehicle_type == TRUCK:
            halfwidth, radius = 0.2, 0.4  # size of the wheels
        elif self.vehicle_type == MOTORCYCLE or self.vehicle_type == BICYCLE:
            halfwidth, radius = 0.05, 0.3  # size of the wheels
        else:
            raise ZValueError("Unrecognised vehicle type", vehicle_type=self.vehicle_type)
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
        if self.vehicle_type in FourWheelsTypes:
            # return 4 wheels position (always the first half are the front ones)
            positions = np.array(
                [
                    [self.lf, self.lf, -self.lr, -self.lr],
                    [self.w_half, -self.w_half, self.w_half, -self.w_half],
                    [1, 1, 1, 1],
                ]
            )

        else:  # self.vehicle_type == MOTORCYCLE or self.vehicle_type == BICYCLE
            positions = np.array([[self.lf, -self.lr], [0, 0], [1, 1]])
        return positions

    @cached_property
    def lights_position(self) -> Mapping[str, Tuple[float, float]]:
        halfwidth, _ = self.wheel_shape
        frontbumper, backbumper = self.bumpers_length
        return {
            "back_left": (-self.lr - backbumper, +self.w_half - halfwidth),
            "back_right": (-self.lr - backbumper, -self.w_half + halfwidth),
            "front_left": (self.lf + frontbumper, +self.w_half - halfwidth),
            "front_right": (self.lf + frontbumper, -self.w_half + halfwidth),
        }

    @cached_property
    def n_wheels(self) -> int:
        return self.wheels_position.shape[1]

    def get_rotated_wheels_outlines(self, delta: float) -> List[np.ndarray]:
        """
        :param delta: Steering angle of front wheels
        :return:
        """
        wheels_position = self.wheels_position
        assert self.n_wheels in (2, 4), self.n_wheels
        transformed_wheels_outlines = []
        for i in range(self.n_wheels):
            # the first half of the wheels are the ones that get rotated
            if i < self.n_wheels / 2:
                transform = SE2_from_xytheta((wheels_position[0, i], wheels_position[1, i], delta))
            else:
                transform = SE2_from_xytheta((wheels_position[0, i], wheels_position[1, i], 0))
            transformed_wheels_outlines.append(transform @ self.wheel_outline)
        return transformed_wheels_outlines
