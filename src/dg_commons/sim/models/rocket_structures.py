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
    ROCKET,
    ModelType,
)

__all__ = ["RocketGeometry", "RocketParameters"]


# todo to review all


@dataclass(frozen=True, unsafe_hash=True)
class RocketGeometry(ModelGeometry):
    """Geometry parameters of the rocket (and colour)"""

    w_half: float
    """ Half width of the rocket - half width of the rocket [m] """
    l_c: float
    """ Length of nose cone [m] """
    l_f: float
    """ Front length of rocket - dist from CoG to the nose [m] """
    l_m: float
    """ Middle length of rocket - dist from CoG to thruster location [m] """
    l_r: float
    """ Rear length of rocket - dist from thruster location to back [m] """
    l: float
    """ Total length of rocket - dist from nosecone tip to the back [m] """
    r: float
    """ Safety radius of the rocket - radius of the rocket [m] """
    l_t_half: float
    """ Half Length of the thruster [m] """
    w_t_half: float
    """ Half Width of the thruster [m] """
    F_max: float
    """ Maximum thrust for plotting[N] """

    model_type: ModelType = ROCKET

    @classmethod
    def default(
        cls,
        color: Color = "royalblue",
        m=2.0,  # MASS TO BE INTENDED AS MASS OF THE ROCKET WITHOUT FUEL
        Iz=1e-00,
        w_half=0.2,
        l_c=0.4,
        l_f=0.2,
        l_m=0.3,
        l_r=0.3,
        l=1.2,
        l_t_half=0.1,
        w_t_half=0.05,
        F_max=2.0,
    ) -> "RocketGeometry":
        return RocketGeometry(
            m=m,
            Iz=Iz,
            w_half=w_half,
            l_c=l_c,
            l_f=l_f,
            l_m=l_m,
            l_r=l_r,
            l=l_r + l_m + l_f + l_c,
            r=max(l_m + l_r, l_f + l_c) * 1.2,
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
    def outline(self) -> tuple[tuple[float, float], ...]:
        """
        Outline of the rocket. The outline is made up of a rectangle and a triangle.
        The cog is at the end of the rectangle (circle center).
        """
        body = Polygon(
            [
                (-self.l_r - self.l_m, self.w_half),
                (self.l_f, self.w_half),
                (self.l_f, -self.w_half),
                (-self.l_r - self.l_m, -self.w_half),
                (-self.l_r - self.l_m, self.w_half),
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
        return 2

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
        positions = [SE2_from_xytheta((-self.l_m, self.w_half, phi)), SE2_from_xytheta((-self.l_m, -self.w_half, -phi))]
        return positions

    def thrusters_outline_in_body_frame(self, phi: float) -> list[tuple[tuple[float, float], ...]]:
        """Takes phi angle of nozzle w.r.t. body frame"""
        thrusters_outline = [transform_xy(q, self.thruster_outline) for q in self.thrusters_position(phi + math.pi / 2)]
        return thrusters_outline

    def flame_outline(self, F: float) -> tuple[tuple[float, float], ...]:
        w_half, l_half = self.w_t_half, self.l_t_half

        l_flame = F / self.F_max * l_half
        l_flame = max(l_flame, 0)
        l_flame = min(l_flame, 2 * l_half)
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
        positions = [SE2_from_xytheta((-self.l_m, self.w_half, phi)), SE2_from_xytheta((-self.l_m, -self.w_half, -phi))]
        return positions

    def flames_outline_in_body_frame(
        self, phi: float, command: [float, float]
    ) -> list[tuple[tuple[float, float], ...]]:
        """Takes phi angle of nozzle w.r.t. body frame"""
        flame_pos = self.flame_position(phi + math.pi / 2)
        flame_outline = [self.flame_outline(command[0]), self.flame_outline(command[1])]
        flame_outline = [transform_xy(q, flame_outline[i]) for i, q in enumerate(flame_pos)]
        return flame_outline


@dataclass(frozen=True, unsafe_hash=True)
class RocketParameters(ModelParameters):
    m_v: float
    """ Mass of the vehicle [kg] """
    C_T: float
    """ Thrust coefficient [1/(I_sp) I_sp: specific impulse] [N] """
    F_limits: tuple[float, float]
    """ Maximum thrust [N] """
    phi_limits: tuple[float, float]
    """ Maximum nozzle angle [rad] """
    dphi_limits: tuple[float, float]
    """ Maximum nozzle angular velocity [rad/s] """

    @classmethod
    def default(
        cls,
        m_v=2.0,
        C_T=0.01,
        vx_limits=(kmh2ms(-7.2), kmh2ms(7.2)),
        acc_limits=(-1.0, 1.0),
        F_limits=(0.0, 2.0),
        phi_limits=(-np.deg2rad(60), np.deg2rad(60)),
        dphi_limits=(-np.deg2rad(20), np.deg2rad(20)),
    ) -> "RocketParameters":
        return RocketParameters(
            m_v=m_v,
            C_T=C_T,
            vx_limits=vx_limits,
            acc_limits=acc_limits,
            F_limits=F_limits,
            phi_limits=phi_limits,
            dphi_limits=dphi_limits,
        )

    def __post_init__(self):
        super(RocketParameters, self).__post_init__()
        assert self.dphi_limits[0] < self.dphi_limits[1]
        assert self.phi_limits[0] < self.phi_limits[1]
        assert self.F_limits[0] < self.F_limits[1]
