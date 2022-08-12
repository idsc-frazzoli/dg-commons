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

from dg_commons.sim import ImpactLocation, IMPACT_EVERYWHERE
from dg_commons.sim.models import ModelType, ModelParameters
from dg_commons.sim.models.model_utils import apply_acceleration_limits, apply_rot_speed_constraint
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry, SpacecraftParameters
from dg_commons.sim.simulator_structures import SimModel


@dataclass(unsafe_hash=True, eq=True, order=True)
class SpacecraftCommands:
    acc_left: float
    """ linear acceleration [m/s^2] """
    acc_right: float
    """ linear acceleration [m/s^2]"""
    idx = frozendict({"acc_left": 0, "acc_right": 1})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_commands(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "SpacecraftCommands") -> "SpacecraftCommands":
        if type(other) == type(self):
            return replace(self, acc_left=self.acc_left + other.acc_left, acc_right=self.acc_right + other.acc_right)
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "SpacecraftCommands") -> "SpacecraftCommands":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "SpacecraftCommands":
        return replace(self, acc_left=self.acc_left * val, acc_right=self.acc_right * val)

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "SpacecraftCommands":
        return self * (1 / val)

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.acc_left, self.acc_right])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_commands() == z.size == z.shape[0], f"z vector {z} cannot initialize VehicleInputs."
        return SpacecraftCommands(acc_left=z[cls.idx["acc_left "]], acc_right=z[cls.idx["acc_right"]])


@dataclass(unsafe_hash=True, eq=True, order=True)
class SpacecraftState:
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
    idx = frozendict({"x": 0, "y": 1, "psi": 2, "vx": 3, "vy": 4, "dpsi": 5})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_states(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "SpacecraftState") -> "SpacecraftState":
        if type(other) == type(self):
            return replace(
                self,
                x=self.x + other.x,
                y=self.y + other.y,
                psi=self.psi + other.psi,
                vx=self.vx + other.vx,
                vy=self.vy + other.vy,
                dpsi=self.dpsi + other.dpsi,
            )
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "SpacecraftState") -> "SpacecraftState":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "SpacecraftState":
        return replace(
            self,
            x=self.x * val,
            y=self.y * val,
            psi=self.psi * val,
            vx=self.vx * val,
            vy=self.vy * val,
            dpsi=self.dpsi * val,
        )

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "SpacecraftState":
        return self * (1 / val)

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("idx")})

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.vx, self.vy, self.dpsi])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_states() == z.size == z.shape[0], f"z vector {z} cannot initialize QuadState."
        return SpacecraftState(
            x=z[cls.idx["x"]],
            y=z[cls.idx["y"]],
            psi=z[cls.idx["psi"]],
            vx=z[cls.idx["vx"]],
            vy=z[cls.idx["vy"]],
            dpsi=z[cls.idx["dpsi"]],
        )


class SpacecraftModel(SimModel[SpacecraftState, SpacecraftCommands]):
    def __init__(self, x0: SpacecraftState, sg: SpacecraftGeometry, sp: SpacecraftParameters):
        self._state: SpacecraftState = x0
        """ Current state of the model"""
        self.XT: Type[SpacecraftState] = type(x0)
        """ State type"""
        self.sg: SpacecraftGeometry = sg
        """ The vehicle's geometry parameters"""
        self.sp: SpacecraftParameters = sp
        """ The vehicle parameters"""

    @classmethod
    def default(cls, x0: SpacecraftState):
        return SpacecraftModel(x0=x0, sg=SpacecraftGeometry.default(), sp=SpacecraftParameters.default())

    def update(self, commands: SpacecraftCommands, dt: Decimal):
        """
        Perform initial value problem integration
        to propagate state using actions for time dt
        """

        def _stateactions_from_array(y: np.ndarray) -> [SpacecraftState, SpacecraftCommands]:
            n_states = self.XT.get_n_states()
            state = self.XT.from_array(y[0:n_states])
            if self.has_collided:
                actions = SpacecraftCommands(acc_left=0, acc_right=0)
            else:
                actions = SpacecraftCommands(
                    acc_left=y[SpacecraftCommands.idx["acc_left"] + n_states],
                    acc_right=y[SpacecraftCommands.idx["acc_right"] + n_states],
                )
            return state, actions

        def _dynamics(t, y):
            state0, actions = _stateactions_from_array(y=y)
            dx = self.dynamics(x0=state0, u=actions)
            du = np.zeros([len(SpacecraftCommands.idx)])
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

    def dynamics(self, x0: SpacecraftState, u: SpacecraftCommands) -> SpacecraftState:
        """Returns state derivative for given control inputs"""
        acc_lx = apply_acceleration_limits(u.acc_left, self.sp)
        acc_rx = apply_acceleration_limits(u.acc_right, self.sp)
        acc_sum = acc_lx + acc_rx
        acc_diff = acc_rx - acc_lx

        vx = x0.vx
        vy = x0.vy
        costh = math.cos(x0.psi)
        sinth = math.sin(x0.psi)
        dx = vx * costh - vy * sinth
        dy = vx * sinth + vy * costh

        ax = acc_sum + x0.vy * x0.dpsi
        ay = -x0.vx * x0.dpsi
        ddpsi = self.sg.w_half * self.sg.m / self.sg.Iz * acc_diff  # need to be saturated first
        ddpsi = apply_rot_speed_constraint(x0.dpsi, ddpsi, self.sp)
        return SpacecraftState(x=dx, y=dy, psi=x0.dpsi, vx=ax, vy=ay, dpsi=ddpsi)

    def get_footprint(self) -> Polygon:
        """Returns current footprint of the spacecraft (mainly for collision checking)"""
        footprint = self.sg.outline_as_polygon
        transform = self.get_pose()
        matrix_coeff = transform[0, :2].tolist() + transform[1, :2].tolist() + transform[:2, 2].tolist()
        footprint = affine_transform(footprint, matrix_coeff)
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
    def model_geometry(self) -> SpacecraftGeometry:
        return self.sg

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

    def set_velocity(self, vel: T2value, omega: float, in_model_frame: bool):
        if not in_model_frame:
            rot: SO2value = SO2_from_angle(-self._state.psi)
            vel = rot @ vel
        self._state.vx = vel[0]
        self._state.vy = vel[1]
        self._state.dpsi = omega

    @property
    def model_type(self) -> ModelType:
        return self.sg.model_type

    @property
    def model_params(self) -> ModelParameters:
        return self.sp

    def get_extra_collision_friction_acc(self):
        # this model is not dynamic
        pass
