import math
import numpy as np
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
    SPACESHIP,
    ModelType,
)

__all__ = ["SpaceshipGeometry", "SpaceshipParameters"]


@dataclass(frozen=True, unsafe_hash=True)
class SpaceshipGeometry(ModelGeometry):
    """Geometry parameters of the rocket (and colour)"""

    w_half: float
    """ Half width of the rocket - half width of the rocket [m] """
    l_c: float
    """ Length of nose cone [m] """
    l_f: float
    """ Front length of rocket - dist from CoG to nose cone [m] """
    l_r: float
    """ Rear length of rocket - dist from back thruster to CoG [m] """
    l_t_half: float
    """ Half Length of the thruster [m] """
    w_t_half: float
    """ Half Width of the thruster [m] """
    F_max: float
    """ Maximum thrust for plotting[N] """
    model_type: ModelType = SPACESHIP

    @classmethod
    def default(
        cls,
        color: Color = "royalblue",
        m=2.0,  # MASS TO BE INTENDED AS MASS OF THE ROCKET WITHOUT FUEL
        Iz=1e-00,
        w_half=0.4,
        l_c=0.8,
        l_f=0.5,
        l_r=1,
        l_t_half=0.3,
        w_t_half=0.05,
        F_max=3.0,
    ) -> "SpaceshipGeometry":
        return SpaceshipGeometry(
            m=m,
            Iz=Iz,
            w_half=w_half,
            l_c=l_c,
            l_f=l_f,
            l_r=l_r,
            l_t_half=l_t_half,
            w_t_half=w_t_half,
            F_max=F_max,
            e=0.7,
            color=color,
        )

    @cached_property
    def width(self):
        return self.w_half * 2

    @cached_property
    def l(self):
        """Total length of rocket - dist from nosecone tip to the back [m]"""
        return self.l_f + self.l_r + self.l_c

    @cached_property
    def outline(self) -> tuple[tuple[float, float], ...]:
        """
        Outline of the rocket. The outline is made up of a rectangle and a triangle.
        The cog is at the end of the rectangle (circle center).
        """
        body = Polygon(
            [
                (-self.l_r, self.w_half),
                (self.l_f, self.w_half),
                (self.l_f, -self.w_half),
                (-self.l_r, -self.w_half),
                (-self.l_r, self.w_half),
            ]
        )
        header = Polygon(
            [
                (self.l_f, self.w_half),
                (self.l_f + self.l_c, 0),
                (self.l_f, -self.w_half),
                (self.l_f, self.w_half),
            ]
        )
        rocket_poly = unary_union([body, header])
        return tuple(rocket_poly.exterior.coords)

    @cached_property
    def outline_as_polygon(self) -> Polygon:
        return Polygon(self.outline)

    @property
    def n_thrusters(self) -> int:
        return 1

    @property
    def thruster_outline(self) -> tuple[tuple[float, float], ...]:
        w_half, l_half = self.w_t_half, self.l_t_half
        thruster = Polygon(
            [
                (0, -w_half),
                (0, w_half),
                (l_half, w_half),
                (l_half, -w_half),
                (0, -w_half),
            ]
        )

        return tuple(thruster.exterior.coords)

    def thrusters_position(self, phi: float) -> list[SE2value]:
        """Takes phi orientation (yaw angle) of the spaceship"""
        positions = [
            SE2_from_xytheta((-self.l_r, 0, phi + math.pi)),
        ]
        return positions

    def thrusters_outline_in_body_frame(self, phi: float) -> list[tuple[tuple[float, float], ...]]:
        """Takes phi angle of nozzle w.r.t. body frame"""
        thrusters_outline = [transform_xy(q, self.thruster_outline) for q in self.thrusters_position(phi)]
        return thrusters_outline

    def flame_outline(self, F: float) -> tuple[tuple[float, float], ...]:
        w_half, l_half = self.w_t_half, self.l_t_half

        l_flame = F / self.F_max / 2
        flame = Polygon(
            [
                (l_half, -w_half),
                (l_flame + l_half, 0),
                (l_half, w_half),
                (l_half, -w_half),
            ]
        )
        return tuple(flame.exterior.coords)

    def flame_position(self, phi: float) -> list[SE2value]:
        # positions = [SE2_from_xytheta((-self.l_m, self.w_half, phi)), SE2_from_xytheta((-self.l_m, -self.w_half, -phi))]
        positions = [
            SE2_from_xytheta((-self.l_r, 0, phi + math.pi)),
        ]
        return positions

    def flames_outline_in_body_frame(
        self, phi: float, command: [float, float]
    ) -> list[tuple[tuple[float, float], ...]]:
        """Takes phi angle of nozzle w.r.t. body frame"""
        flame_pos = self.flame_position(phi)
        flame_outline = [self.flame_outline(command)]
        flame_outline = [transform_xy(q, flame_outline[i]) for i, q in enumerate(flame_pos)]
        return flame_outline


@dataclass(frozen=True, unsafe_hash=True)
class SpaceshipParameters(ModelParameters):
    m_v: float
    """ Mass of the Spaceship [kg] """
    C_T: float
    """ Thrust coefficient [1/(I_sp) I_sp: specific impulse] [N] """
    thrust_limits: tuple[float, float]
    """ Maximum thrust [N] """
    delta_limits: tuple[float, float]
    """ Maximum nozzle angle [rad] """
    ddelta_limits: tuple[float, float]
    """ Maximum nozzle angular velocity [rad/s] """

    @classmethod
    def default(
        cls,
        m_v=2.0,
        C_T=0.01,
        vx_limits=(kmh2ms(-10), kmh2ms(10)),
        acc_limits=(-1.0, 1.0),
        thrust_limits=(0.0, 2.0),
        delta_limits=(-np.deg2rad(60), np.deg2rad(60)),
        ddelta_limits=(-np.deg2rad(45), np.deg2rad(45)),
    ) -> "SpaceshipParameters":
        return SpaceshipParameters(
            m_v=m_v,
            C_T=C_T,
            vx_limits=vx_limits,
            acc_limits=acc_limits,
            thrust_limits=thrust_limits,
            delta_limits=delta_limits,
            ddelta_limits=ddelta_limits,
        )

    def __post_init__(self):
        super().__post_init__()
        assert self.ddelta_limits[0] < self.ddelta_limits[1]
        assert self.delta_limits[0] < self.delta_limits[1]
        assert self.thrust_limits[0] < self.thrust_limits[1]
