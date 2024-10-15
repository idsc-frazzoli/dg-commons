from dataclasses import dataclass, replace
from decimal import Decimal
from math import sin, cos
from typing import Type, Mapping

from dg_commons import logger

import numpy as np
from frozendict import frozendict
from geometry import SE2value, SE2_from_xytheta, SO2_from_angle, SO2value, T2value
from scipy.integrate import solve_ivp
from shapely.geometry import Polygon

from dg_commons import apply_SE2_to_shapely_geo
from dg_commons.sim import ImpactLocation, IMPACT_EVERYWHERE
from dg_commons.sim.models import ModelType, ModelParameters
from dg_commons.sim.models.model_utils import apply_force_limits, apply_full_ang_vel_limits  # , apply_speed_limits
from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry, SpaceshipParameters
from dg_commons.sim.simulator_structures import SimModel


@dataclass(unsafe_hash=True, eq=True, order=True)
class SpaceshipCommands:
    thrust: float
    """ Thrust generated by engine [N]"""
    ddelta: float
    """ Angular velocity of nozzle direction change [rad/s]"""

    idx = frozendict({"thrust": 0, "ddelta": 1})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_commands(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "SpaceshipCommands") -> "SpaceshipCommands":
        if type(other) == type(self):
            return replace(
                self,
                thrust=self.thrust + other.thrust,
                ddelta=self.ddelta + other.ddelta,
            )
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "SpaceshipCommands") -> "SpaceshipCommands":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "SpaceshipCommands":
        return replace(self, thrust=self.thrust * val, ddelta=self.ddelta * val)

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "SpaceshipCommands":
        return self * (1 / val)

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.thrust, self.ddelta])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_commands() == z.size == z.shape[0], f"z vector {z} cannot initialize VehicleInputs."
        return SpaceshipCommands(thrust=z[cls.idx["thrust"]], ddelta=z[cls.idx["ddelta"]])


@dataclass(unsafe_hash=True, eq=True, order=True)
class SpaceshipState:
    x: float
    """ CoG x location [m] """
    y: float
    """ CoG y location [m] """
    psi: float
    """ Heading (yaw) [rad] """
    vx: float
    """ CoG longitudinal velocity [m/s] """
    vy: float
    """ CoG longitudinal velocity [m/s] """
    dpsi: float
    """ Heading (yaw) rate [rad/s] """
    delta: float
    """ Nozzle direction [rad] """
    m: float
    """ Mass (dry + fuel) [kg] """
    idx = frozendict({"x": 0, "y": 1, "psi": 2, "vx": 3, "vy": 4, "dpsi": 5, "delta": 6, "m": 7})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_states(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "SpaceshipState") -> "SpaceshipState":
        if type(other) == type(self):
            return replace(
                self,
                x=self.x + other.x,
                y=self.y + other.y,
                psi=self.psi + other.psi,
                vx=self.vx + other.vx,
                vy=self.vy + other.vy,
                dpsi=self.dpsi + other.dpsi,
                delta=self.delta + other.delta,
                m=self.m + other.m,
            )
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "SpaceshipState") -> "SpaceshipState":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "SpaceshipState":
        return replace(
            self,
            x=self.x * val,
            y=self.y * val,
            psi=self.psi * val,
            vx=self.vx * val,
            vy=self.vy * val,
            dpsi=self.dpsi * val,
            delta=self.delta * val,
            m=self.m * val,
        )

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "SpaceshipState":
        return self * (1 / val)

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("idx")})

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.vx, self.vy, self.dpsi, self.delta, self.m])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_states() == z.size == z.shape[0], f"z vector {z} cannot initialize SpaceshipState."
        return SpaceshipState(
            x=z[cls.idx["x"]],
            y=z[cls.idx["y"]],
            psi=z[cls.idx["psi"]],
            vx=z[cls.idx["vx"]],
            vy=z[cls.idx["vy"]],
            dpsi=z[cls.idx["dpsi"]],
            delta=z[cls.idx["delta"]],
            m=z[cls.idx["m"]],
        )


class SpaceshipModel(SimModel[SpaceshipState, SpaceshipCommands]):
    def __init__(self, x0: SpaceshipState, rg: SpaceshipGeometry, sp: SpaceshipParameters):
        self._state: SpaceshipState = x0
        """ Current state of the model"""
        self.XT: Type[SpaceshipState] = type(x0)
        """ State type"""
        self.rg: SpaceshipGeometry = rg
        """ The vehicle's geometry parameters"""
        self.sp: SpaceshipParameters = sp
        """ The vehicle parameters"""

    @classmethod
    def default(cls, x0: SpaceshipState):
        return SpaceshipModel(x0=x0, rg=SpaceshipGeometry.default(), sp=SpaceshipParameters.default())

    def update(self, commands: SpaceshipCommands, dt: Decimal):
        """
        Perform initial value problem integration
        to propagate state using actions for time dt
        """

        def _stateactions_from_array(y: np.ndarray) -> [SpaceshipState, SpaceshipCommands]:
            n_states = self._state.get_n_states()
            state = self._state.from_array(y[0:n_states])
            if self.has_collided:
                actions = SpaceshipCommands(thrust=0, ddelta=0)
            else:
                actions = SpaceshipCommands(
                    thrust=float(y[SpaceshipCommands.idx["thrust"] + n_states]),
                    ddelta=float(y[SpaceshipCommands.idx["ddelta"] + n_states]),
                )
            return state, actions

        def _dynamics(t, y):
            state0, actions = _stateactions_from_array(y=y)
            dx = self.dynamics(x0=state0, u=actions)
            du = np.zeros([len(SpaceshipCommands.idx)])
            return np.concatenate([dx.as_ndarray(), du])

        state_np = self._state.as_ndarray()
        action_np = commands.as_ndarray()
        y0 = np.concatenate([state_np, action_np])
        result = solve_ivp(fun=_dynamics, t_span=(0.0, float(dt)), y0=y0)

        if not result.success:
            raise RuntimeError("Failed to integrate ivp!")
        new_state, _ = _stateactions_from_array(result.y[:, -1])
        self._state = new_state
        return

    def dynamics(self, x0: SpaceshipState, u: SpaceshipCommands) -> SpaceshipState:
        """
        Returns state derivative for given control inputs
        # todo update dynamics
        Dynamics:
        dx/dt = vx
        dy/dt = vy
        dθ/dt = vθ
        dm/dt = -k_l*thrust
        dvx/dt = 1/m*cos(delta+θ)*thrust
        dvy/dt = 1/m*sin(delta+θ)*thrust
        dvθ/dt = -1/I*l_r*sin(delta)*thrust
        ddelta/dt = vdelta

        """
        # todo update dynamics
        thrust = apply_force_limits(u.thrust, self.sp.thrust_limits)
        ddelta = apply_full_ang_vel_limits(x0.delta, u.ddelta, self.sp)

        # set actions to zero if vehicle has no more fuel
        if x0.m <= self.sp.m_v:
            thrust = 0
            logger.warning("Vehicle has no more fuel!")

        psi = x0.psi
        dpsi = x0.dpsi
        m = x0.m
        vx = x0.vx
        vy = x0.vy
        delta = x0.delta

        dx = vx
        dy = vy
        dm = -self.sp.C_T * thrust
        dvx = 1 / m * cos(delta + psi) * thrust
        dvy = 1 / m * sin(delta + psi) * thrust
        dvpsi = -1 / self.rg.Iz * self.rg.l_r * sin(delta) * thrust
        ddelta = ddelta

        return SpaceshipState(x=dx, y=dy, psi=dpsi, vx=dvx, vy=dvy, dpsi=dvpsi, delta=ddelta, m=dm)

    def get_footprint(self) -> Polygon:
        """Returns current footprint of the rocket (mainly for collision checking)"""
        footprint = self.rg.outline_as_polygon
        transform = self.get_pose()
        footprint: Polygon = apply_SE2_to_shapely_geo(footprint, transform)
        assert footprint.is_valid
        return footprint

    def get_mesh(self) -> Mapping[ImpactLocation, Polygon]:
        footprint = self.get_footprint()
        impact_locations: Mapping[ImpactLocation, Polygon] = {
            IMPACT_EVERYWHERE: footprint,
        }
        for shape in impact_locations.values():
            assert shape.is_valid
        return impact_locations

    def get_pose(self) -> SE2value:
        return SE2_from_xytheta([self._state.x, self._state.y, self._state.psi])

    @property
    def model_geometry(self) -> SpaceshipGeometry:
        return self.rg

    def get_velocity(self, in_model_frame: bool) -> (T2value, float):
        """Returns velocity at COG"""
        vx = self._state.vx
        vy = self._state.vy
        dpsi = self._state.dpsi
        v_l = np.array([vx, vy])
        if in_model_frame:
            return v_l, dpsi
        rot: SO2value = SO2_from_angle(self._state.psi)
        v_g = rot @ v_l
        return v_g, dpsi

    def set_velocity(self, vel: T2value, dpsi: float, in_model_frame: bool):
        if not in_model_frame:
            rot: SO2value = SO2_from_angle(-self._state.psi)
            vel = rot @ vel
        self._state.vx = vel[0]
        self._state.vy = vel[1]
        self._state.dpsi = dpsi

    @property
    def model_type(self) -> ModelType:
        return self.rg.model_type

    @property
    def model_params(self) -> ModelParameters:
        return self.sp

    def get_extra_collision_friction_acc(self):
        raise NotImplementedError
