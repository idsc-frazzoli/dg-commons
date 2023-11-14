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
    """ half width of the rocket - half width of the rocket [m] """
    l_f: float
    """ Front length of rocket - dist from CoG to front axle [m] """
    l_m: float
    """ Middle length of rocket - dist from CoG to thruster location [m] """
    l_r: float
    """ Rear length of rocket - dist from thruster location to rear axle [m] """
    l: float
    """ Total length of rocket - dist from front axle to rear axle [m] """
    r: float
    """ Safety radius of the rocket - radius of the rocket [m] """
    l_t_half: float
    """ Half Length of the thruster [m] """
    w_t_half: float
    """ Half Width of the thruster [m] """

    model_type: ModelType = ROCKET

    @classmethod
    def default(
        cls,
        color: Color = "royalblue",
        m=2.0,  # MASS TO BE INTENDED AS MASS OF THE ROCKET WITHOUT FUEL
        Iz=1e-00,
        w_half=0.1,
        l_f=0.3,
        l_m=0.15,
        l_r=0.15,
        l=0.6,
        l_t_half=0.1,
        w_t_half=0.05,
    ) -> "RocketGeometry":
        return RocketGeometry(
            m=m,
            Iz=Iz,
            w_half=w_half,
            l_f=l_f,
            l_m=l_m,
            l_r=l_r,
            l=l_r + l_m + l_f,
            r=max(l-l_f, l_f)*1.25,
            l_t_half=l_t_half,
            w_t_half=w_t_half,
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
                (-self.l_f, self.w_half, 1),
                (self.l_m+self.l_r, self.w_half, 1),
                (self.l_m+self.l_r, -self.w_half, 1),
                (-self.l_f, -self.w_half, 1),
                (-self.l_f, self.w_half, 1),
            ]
        )
        header = Polygon(
            [
                (self.l_m+self.l_r, self.w_half, 1),
                (self.l, 0, 1),
                (self.l_m+self.l_r, -self.w_half, 1),
                (self.l_m+self.l_r, self.w_half, 1),
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
    def thruster_shape(self):
        w_half, l_half = self.w_t_half, self.l_t_half
        return w_half, l_half

    @property
    def thruster_outline(self) -> tuple[tuple[float, float], ...]:
        w_half, l_half = self.thruster_shape
        return (l_half, -w_half), (-l_half, -w_half), (-l_half, w_half), (l_half, w_half), (l_half, -w_half)

    @property
    def thrusters_position(self) -> list[SE2value]:
        positions = [SE2_from_xytheta((-self.l_m, -self.w_half, 0)), SE2_from_xytheta((-self.l_m, self.w_half, 0))]
        return positions

    @cached_property
    def thrusters_outline_in_body_frame(self) -> list[tuple[tuple[float, float], ...]]:
        thrusters_outline = [transform_xy(q, self.thruster_outline) for q in self.thrusters_position]
        return thrusters_outline


@dataclass(frozen=True, unsafe_hash=True)
class RocketParameters(ModelParameters):
    m_fuel: float
    """ Mass of the fuel [kg] """
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
        m_fuel=0.1,
        C_T=0.01,
        vx_limits=(kmh2ms(-7.2), kmh2ms(7.2)),
        acc_limits=(-1.0, 1.0),
        F_limits=(0.0, 2.0),
        phi_limits=(-np.deg2rad(60), np.deg2rad(60)),
        dphi_limits=(-np.deg2rad(20), np.deg2rad(20))
    ) -> "RocketParameters":
        return RocketParameters(
            m_fuel=m_fuel,
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

