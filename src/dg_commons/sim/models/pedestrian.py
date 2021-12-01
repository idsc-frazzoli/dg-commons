from dataclasses import dataclass, replace
from functools import cached_property
from math import cos, sin
from typing import Tuple, Type, Sequence, Mapping

import numpy as np
from frozendict import frozendict
from geometry import SE2value, T2value, SE2_from_xytheta, SO2_from_angle, SO2value
from scipy.integrate import solve_ivp
from shapely import affinity
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon

from dg_commons.sim import SimModel, SimTime, ImpactLocation, IMPACT_EVERYWHERE
from dg_commons.sim.models.model_structures import ModelGeometry, PEDESTRIAN, ModelType
from dg_commons.sim.models.model_utils import apply_full_acceleration_limits
from dg_commons.sim.models.pedestrian_utils import PedestrianParameters, rotation_constraint


@dataclass(frozen=True)
class PedestrianGeometry(ModelGeometry):
    @cached_property
    def outline(self) -> Sequence[Tuple[float, float]]:
        circle = Point(0, 0).buffer(0.5)  # type(circle)=polygon
        ellipse = affinity.scale(circle, 1, 1.5)  # not sure, maybe just a circle?
        return tuple(ellipse.exterior.coords)

    @cached_property
    def outline_as_polygon(self) -> Polygon:
        return Polygon(self.outline)

    @classmethod
    def default(cls):
        return PedestrianGeometry(m=75, Iz=50, e=0.35, color="pink")


@dataclass(unsafe_hash=True, eq=True, order=True)
class PedestrianCommands:
    acc: float
    """ Acceleration [m/s^2] """
    dtheta: float
    """ rotational acceleration of the pedestrian"""
    idx = frozendict({"acc": 0, "dtheta": 1})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_commands(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "PedestrianCommands") -> "PedestrianCommands":
        if type(other) == type(self):
            return replace(self, acc=self.acc + other.acc, dtheta=self.dtheta + other.dtheta)
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "PedestrianCommands") -> "PedestrianCommands":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "PedestrianCommands":
        return replace(self, acc=self.acc * val, dtheta=self.dtheta * val)

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "PedestrianCommands":
        return self * (1 / val)

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.acc, self.dtheta])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_commands() == z.size == z.shape[0], f"z vector {z} cannot initialize VehicleInputs."
        return PedestrianCommands(acc=z[cls.idx["acc"]], dtheta=z[cls.idx["dtheta"]])


@dataclass(unsafe_hash=True, eq=True, order=True)
class PedestrianState:
    x: float
    """ x-position of pedestrian [m] """
    y: float
    """ y-position of pedestrian [m] """
    theta: float
    """ orientation [rad] """
    vx: float
    """ longitudinal speed [m/s] """
    vy: float = 0
    """ lateral speed [m/s] """
    dtheta: float = 0
    """ rot speed [rad/s] """
    idx = frozendict({"x": 0, "y": 1, "theta": 2, "vx": 3, "vy": 4, "dtheta": 5})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_states(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "PedestrianState") -> "PedestrianState":
        if type(other) == type(self):
            return replace(
                self,
                x=self.x + other.x,
                y=self.y + other.y,
                theta=self.theta + other.theta,
                vx=self.vx + other.vx,
                vy=self.vy + other.vy,
                dtheta=self.dtheta + other.dtheta,
            )
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "PedestrianState") -> "PedestrianState":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "PedestrianState":
        return replace(
            self,
            x=self.x * val,
            y=self.y * val,
            theta=self.theta * val,
            vx=self.vx * val,
            vy=self.vy * val,
            dtheta=self.dtheta * val,
        )

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "PedestrianState":
        return self * (1 / val)

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("idx")})

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.vx, self.vy, self.dtheta])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_states() == z.size == z.shape[0], f"z vector {z} cannot initialize VehicleState."
        return PedestrianState(
            x=z[cls.idx["x"]],
            y=z[cls.idx["y"]],
            theta=z[cls.idx["theta"]],
            vx=z[cls.idx["vx"]],
            vy=z[cls.idx["vy"]],
            dtheta=z[cls.idx["dtheta"]],
        )


class PedestrianModel(SimModel[SE2value, float]):
    def __init__(self, x0: PedestrianState, pg: PedestrianGeometry, pp: PedestrianParameters):
        assert isinstance(x0, PedestrianState)
        self._state: PedestrianState = x0
        """ Current state of the model"""
        self.XT: Type[PedestrianState] = type(x0)
        """ State type"""
        self.pg: PedestrianGeometry = pg
        """ The vehicle's geometry parameters"""
        self.pp: PedestrianParameters = pp
        """ Pedestrian Parameters"""

    @classmethod
    def default(cls, x0: PedestrianState):
        return PedestrianModel(x0=x0, pg=PedestrianGeometry.default(), pp=PedestrianParameters.default())

    def update(self, commands: PedestrianCommands, dt: SimTime):
        """
        Perform initial value problem integration
        to propagate state using actions for time dt
        """

        def _stateactions_from_array(y: np.ndarray) -> [PedestrianState, PedestrianCommands]:
            n_states = self.XT.get_n_states()
            state = self.XT.from_array(y[0:n_states])
            if self.has_collided:
                actions = PedestrianCommands(acc=0, dtheta=0)
            else:
                actions = PedestrianCommands(
                    acc=y[PedestrianCommands.idx["acc"] + n_states],
                    dtheta=y[PedestrianCommands.idx["dtheta"] + n_states],
                )
            return state, actions

        def _dynamics(t, y):
            state0, actions = _stateactions_from_array(y=y)
            dx = self.dynamics(x0=state0, u=actions)
            du = np.zeros([len(PedestrianCommands.idx)])
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

    def dynamics(self, x0: PedestrianState, u: PedestrianCommands) -> PedestrianState:
        """Simple double integrator with friction after collision to simulate a "bag of potato" effect"""
        dtheta = rotation_constraint(rot_velocity=u.dtheta, pp=self.pp)
        acc = apply_full_acceleration_limits(speed=x0.vx, acceleration=u.acc, p=self.pp)

        costheta, sintheta = cos(x0.theta), sin(x0.theta)
        frictionx, frictiony, frictiontheta = self.get_extra_collision_friction_acc()
        return PedestrianState(
            x=x0.vx * costheta - x0.vy * sintheta,
            y=x0.vx * sintheta + x0.vy * costheta,
            theta=dtheta,
            vx=acc + frictionx,
            vy=frictiony,
            dtheta=frictiontheta,
        )

    def get_footprint(self) -> Polygon:
        footprint = self.pg.outline_as_polygon
        transform = self.get_pose()
        matrix_coeff = transform[0, :2].tolist() + transform[1, :2].tolist() + transform[:2, 2].tolist()
        footprint = affine_transform(footprint, matrix_coeff)
        assert footprint.is_valid
        return footprint

    def get_mesh(self) -> Mapping[ImpactLocation, Polygon]:
        return {IMPACT_EVERYWHERE: self.get_footprint()}

    def get_pose(self) -> SE2value:
        return SE2_from_xytheta(xytheta=(self._state.x, self._state.y, self._state.theta))

    def get_velocity(self, in_model_frame: bool) -> (T2value, float):
        """Returns velocity at COG"""
        vx = self._state.vx
        vy = self._state.vy
        v_l = np.array([vx, vy])
        if in_model_frame:
            return v_l, self._state.dtheta
        rot: SO2value = SO2_from_angle(self._state.theta)
        v_g = rot @ v_l
        return v_g, self._state.dtheta

    def set_velocity(self, vel: T2value, omega: float, in_model_frame: bool):
        if not in_model_frame:
            rot: SO2value = SO2_from_angle(-self._state.theta)
            vel = rot @ vel
        self._state.vx = vel[0]
        self._state.vy = vel[1]
        self._state.vx = omega

    def get_geometry(self) -> PedestrianGeometry:
        return self.pg

    @property
    def model_type(self) -> ModelType:
        return PEDESTRIAN

    def get_extra_collision_friction_acc(self):
        magic_mu = 2.0
        if self.has_collided:
            frictionx = -magic_mu * self._state.vx
            frictiony = -magic_mu * self._state.vy
            frictiontheta = -magic_mu * self._state.dtheta
            return frictionx, frictiony, frictiontheta
        else:
            return 0, 0, 0
