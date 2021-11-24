import math
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Type, Mapping

import numpy as np
from frozendict import frozendict
from geometry import SE2value, SE2_from_xytheta, SO2_from_angle, SO2value, T2value
from scipy.integrate import solve_ivp
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from dg_commons.sim import logger, ImpactLocation, IMPACT_RIGHT, IMPACT_LEFT, IMPACT_BACK, IMPACT_FRONT
from dg_commons.sim.models import ModelType
from dg_commons.sim.models.model_utils import acceleration_constraint
from dg_commons.sim.models.quadrotor_structures import QuadGeometry, QuadParameters
from dg_commons.sim.models.vehicle_utils import steering_constraint
from dg_commons.sim.simulator_structures import SimModel

logger.error("Quadrotor model is not completed. It will be released in the future.")


@dataclass(unsafe_hash=True, eq=True, order=True)
class QuadCommands:
    acc: float
    """ linear acceleration [m/s^2] """
    dpsi: float
    """ Yaw rate [rad/s]"""
    idx = frozendict({"acc": 0, "dpsi": 1})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_commands(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "QuadCommands") -> "QuadCommands":
        if type(other) == type(self):
            return replace(self, acc=self.acc + other.acc, dpsi=self.dpsi + other.dpsi)
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "QuadCommands") -> "QuadCommands":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "QuadCommands":
        return replace(self, acc=self.acc * val, dpsi=self.dpsi * val)

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "QuadCommands":
        return self * (1 / val)

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.acc, self.dpsi])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_commands() == z.size == z.shape[0], f"z vector {z} cannot initialize VehicleInputs."
        return QuadCommands(acc=z[cls.idx["acc"]], dpsi=z[cls.idx["dpsi"]])


@dataclass(unsafe_hash=True, eq=True, order=True)
class QuadState:
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
    idx = frozendict({"x": 0, "y": 1, "psi": 2, "vx": 3})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_states(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "QuadState") -> "QuadState":
        if type(other) == type(self):
            return replace(
                self,
                x=self.x + other.x,
                y=self.y + other.y,
                psi=self.psi + other.psi,
                vx=self.vx + other.vx,
            )
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "QuadState") -> "QuadState":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "QuadState":
        return replace(
            self,
            x=self.x * val,
            y=self.y * val,
            psi=self.psi * val,
            vx=self.vx * val,
        )

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "QuadState":
        return self * (1 / val)

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("idx")})

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.vx, self.psi])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_states() == z.size == z.shape[0], f"z vector {z} cannot initialize QuadState."
        return QuadState(
            x=z[cls.idx["x"]],
            y=z[cls.idx["y"]],
            psi=z[cls.idx["psi"]],
            vx=z[cls.idx["vx"]],
        )


class QuadModel(SimModel[QuadState, QuadCommands]):
    def __init__(self, x0: QuadState, qg: QuadGeometry, qp: QuadParameters):
        self._state: QuadState = x0
        """ Current state of the model"""
        self.XT: Type[QuadState] = type(x0)
        """ State type"""
        self.qg: QuadGeometry = qg
        """ The vehicle's geometry parameters"""
        self.qp: QuadParameters = qp
        """ The vehicle parameters"""

    @classmethod
    def default(cls, x0: QuadState):
        return QuadModel(x0=x0, qg=QuadGeometry.default(), qp=QuadParameters.default())

    def update(self, commands: QuadCommands, dt: Decimal):
        """
        Perform initial value problem integration
        to propagate state using actions for time dt
        """

        def _stateactions_from_array(y: np.ndarray) -> [QuadState, QuadCommands]:
            n_states = self.XT.get_n_states()
            state = self.XT.from_array(y[0:n_states])
            if self.has_collided:
                actions = QuadCommands(acc=0, dpsi=0)
            else:
                actions = QuadCommands(
                    acc=y[QuadCommands.idx["acc"] + n_states], dpsi=y[QuadCommands.idx["dpsi"] + n_states]
                )
            return state, actions

        def _dynamics(t, y):
            state0, actions = _stateactions_from_array(y=y)
            dx = self.dynamics(x0=state0, u=actions)
            du = np.zeros([len(QuadCommands.idx)])
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

    def dynamics(self, x0: QuadState, u: QuadCommands) -> QuadState:
        """Kinematic bicycle model, returns state derivative for given control inputs"""
        vx = x0.vx
        dpsi = vx * math.tan(x0.psi) / self.qg.length
        vy = u.dpsi * self.qg.lr
        costh = math.cos(x0.psi)
        sinth = math.sin(x0.psi)
        xdot = vx * costh - vy * sinth
        ydot = vx * sinth + vy * costh

        dpsi = steering_constraint(x0.psi, u.dpsi, self.qp)
        acc = acceleration_constraint(x0.vx, u.acc, self.qp)
        return QuadState(x=xdot, y=ydot, psi=dpsi, vx=acc)

    def get_footprint(self) -> Polygon:
        """Returns current footprint of the vehicle (mainly for collision checking)"""
        footprint = self.qg.outline_as_polygon
        transform = self.get_pose()
        matrix_coeff = transform[0, :2].tolist() + transform[1, :2].tolist() + transform[:2, 2].tolist()
        footprint = affine_transform(footprint, matrix_coeff)
        assert footprint.is_valid
        return footprint

    def get_mesh(self) -> Mapping[ImpactLocation, Polygon]:
        footprint = self.get_footprint()
        vertices = footprint.exterior.coords[:-1]
        cxy = footprint.centroid.coords[0]
        # fixme maybe we can use triangulate from shapely inferring the side from the relative angle
        impact_locations: Mapping[ImpactLocation, Polygon] = {
            IMPACT_RIGHT: Polygon([cxy, vertices[0], vertices[3], cxy]),
            IMPACT_LEFT: Polygon([cxy, vertices[1], vertices[2], cxy]),
            IMPACT_BACK: Polygon([cxy, vertices[0], vertices[1], cxy]),
            IMPACT_FRONT: Polygon([cxy, vertices[2], vertices[3], cxy]),
        }
        for shape in impact_locations.values():
            assert shape.is_valid
        return impact_locations

    def get_pose(self) -> SE2value:
        return SE2_from_xytheta([self._state.x, self._state.y, self._state.psi])

    def get_geometry(self) -> QuadGeometry:
        return self.qg

    def get_velocity(self, in_model_frame: bool) -> (T2value, float):
        """Returns velocity at COG"""
        vx = self._state.vx
        vy = dpsi * self.qg.lr
        v_l = np.array([vx, vy])
        if in_model_frame:
            return v_l, dpsi
        rot: SO2value = SO2_from_angle(self._state.psi)
        v_g = rot @ v_l
        return v_g, dpsi

    def set_velocity(self, vel: T2value, omega: float, in_model_frame: bool):
        if not in_model_frame:
            rot: SO2value = SO2_from_angle(-self._state.psi)
            vel = rot @ vel
        self._state.vx = vel[0]
        logger.warn(
            "It is NOT possible to set the lateral and rotational velocity for this model\n"
            "Try using the dynamic model."
        )

    @property
    def model_type(self) -> ModelType:
        return self.qg.model_type

    def get_extra_collision_friction_acc(self):
        # this model is not dynamic
        pass
