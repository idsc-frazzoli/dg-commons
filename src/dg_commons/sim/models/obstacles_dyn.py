from dataclasses import replace, dataclass
from math import cos, sin
from typing import Tuple, Mapping, Type

import numpy as np
from frozendict import frozendict
from geometry import T2value, SE2value, SO2_from_angle, SO2value, SE2_from_xytheta
from scipy.integrate import solve_ivp
from shapely.geometry import Polygon

from dg_commons import U, apply_SE2_to_shapely_geo
from dg_commons.sim.sim_types import ImpactLocation, SimTime
from dg_commons.sim.simulator_structures import SimModel
from dg_commons.sim.collision_structures import IMPACT_EVERYWHERE
from dg_commons.sim.models import ModelType, DYNAMIC_OBSTACLE, ModelGeometry, ModelParameters
from dg_commons.sim.models.model_utils import apply_full_acceleration_limits, apply_rot_speed_constraint
from dg_commons.sim.models.obstacles import ObstacleGeometry, DynObstacleParameters


@dataclass(unsafe_hash=True, eq=True, order=True)
class DynObstacleCommands:
    acc_x: float
    """ Longitudinal acceleration [m/s^2] """
    acc_y: float
    """ Lateral acceleration [m/s^2] """
    acc_psi: float
    """ Rotational acceleration (yaw) [rad/s^2] """
    idx = frozendict({"acc_x": 0, "acc_y": 1, "acc_psi": 2})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_commands(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "DynObstacleCommands") -> "DynObstacleCommands":
        if type(other) == type(self):
            return replace(
                self,
                acc_x=self.acc_x + other.acc_x,
                acc_y=self.acc_y + other.acc_y,
                acc_psi=self.acc_psi + other.acc_psi,
            )
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "DynObstacleCommands") -> "DynObstacleCommands":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "DynObstacleCommands":
        return replace(self, acc_x=self.acc_x * val, acc_y=self.acc_y * val, acc_psi=self.acc_psi * val)

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "DynObstacleCommands":
        return self * (1 / val)

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.acc_x, self.acc_y, self.acc_psi])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_commands() == z.size == z.shape[0], f"z vector {z} cannot initialize VehicleInputs."
        return DynObstacleCommands(acc_x=z[cls.idx["acc_x"]], acc_y=z[cls.idx["acc_y"]], acc_psi=z[cls.idx["acc_psi"]])


@dataclass(unsafe_hash=True, eq=True, order=True)
class DynObstacleState:
    x: float
    """ CoG x location [m] """
    y: float
    """ CoG y location [m] """
    psi: float
    """ CoG heading [rad] """
    vx: float
    """ CoG longitudinal velocity [m/s] """
    vy: float
    """ CoG lateral velocity [m/s] """
    dpsi: float
    """ Rotational rate [rad/s] """
    idx = frozendict({"x": 0, "y": 1, "psi": 2, "vx": 3, "vy": 4, "dpsi": 5})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_states(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "DynObstacleState") -> "DynObstacleState":
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

    def __sub__(self, other: "DynObstacleState") -> "DynObstacleState":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "DynObstacleState":
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

    def __truediv__(self, val: float) -> "DynObstacleState":
        return self * (1 / val)

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("idx")})

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.vx, self.vy, self.dpsi])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_states() == z.size == z.shape[0], f"z vector {z} cannot initialize DynObstacleState."
        return DynObstacleState(
            x=z[cls.idx["x"]],
            y=z[cls.idx["y"]],
            psi=z[cls.idx["psi"]],
            vx=z[cls.idx["vx"]],
            vy=z[cls.idx["vy"]],
            dpsi=z[cls.idx["dpsi"]],
        )


class DynObstacleModel(SimModel[DynObstacleState, DynObstacleCommands]):
    """A dynamic obstacle"""

    def __init__(self, x0: DynObstacleState, shape: Polygon, og: ObstacleGeometry, op: DynObstacleParameters):
        """For realistic behavior it is important that the shape is centered around the origin
        that will be used as the c.o.g. for the obstacle"""
        self._state: DynObstacleState = x0
        self.XT: Type[DynObstacleState] = type(x0)
        self.og: ObstacleGeometry = og
        self.op: DynObstacleParameters = op
        assert shape.is_valid, "Shape is not valid"
        self.shape: Polygon = shape

    def update(self, commands: U, dt: SimTime):
        """
        Perform initial value problem integration
        to propagate state using actions for time dt
        """

        def _stateactions_from_array(y: np.ndarray) -> [DynObstacleState, DynObstacleCommands]:
            n_states = self.XT.get_n_states()
            state = self.XT.from_array(y[0:n_states])
            actions = DynObstacleCommands(
                acc_x=y[DynObstacleCommands.idx["acc_x"] + n_states],
                acc_y=y[DynObstacleCommands.idx["acc_y"] + n_states],
                acc_psi=y[DynObstacleCommands.idx["acc_psi"] + n_states],
            )
            return state, actions

        def _dynamics(t, y):
            state0, actions = _stateactions_from_array(y=y)
            dx = self.dynamics(x0=state0, u=actions)
            du = np.zeros([len(DynObstacleCommands.idx)])
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

    def dynamics(self, x0: DynObstacleState, u: DynObstacleCommands) -> DynObstacleState:
        vx, vy = x0.vx, x0.vy
        costh = cos(x0.psi)
        sinth = sin(x0.psi)
        xdot = vx * costh - vy * sinth
        ydot = vx * sinth + vy * costh

        acc_x = apply_full_acceleration_limits(speed=vx, acceleration=u.acc_x + x0.dpsi * vy, p=self.op)
        acc_y = apply_full_acceleration_limits(speed=vy, acceleration=u.acc_y - x0.dpsi * vx, p=self.op)
        acc_psi = apply_rot_speed_constraint(omega=x0.dpsi, domega=u.acc_psi, p=self.op)

        return DynObstacleState(x=xdot, y=ydot, psi=x0.dpsi, vx=acc_x, vy=acc_y, dpsi=acc_psi)

    def get_footprint(self) -> Polygon:
        transform = self.get_pose()
        return apply_SE2_to_shapely_geo(self.shape, transform)

    def get_pose(self) -> SE2value:
        return SE2_from_xytheta([self._state.x, self._state.y, self._state.psi])

    def get_velocity(self, in_model_frame: bool) -> (T2value, float):
        v_l = np.array([self._state.vx, self._state.vy])
        if in_model_frame:
            return v_l, self._state.dpsi
        rot: SO2value = SO2_from_angle(self._state.psi)
        v_g = rot @ v_l
        return v_g, self._state.dpsi

    def set_velocity(self, vel: T2value, omega: float, in_model_frame: bool):
        if not in_model_frame:
            rot: SO2value = SO2_from_angle(-self._state.psi)
            vel = rot @ vel

        self._state.vx = vel[0]
        self._state.vy = vel[1]
        self._state.dpsi = omega

    @property
    def model_geometry(self) -> ObstacleGeometry:
        return self.og

    def get_mesh(self) -> Mapping[ImpactLocation, Polygon]:
        footprint = self.get_footprint()
        impact_locations: Mapping[ImpactLocation, Polygon] = {IMPACT_EVERYWHERE: footprint}
        for shape in impact_locations.values():
            assert shape.is_valid
        return impact_locations

    @property
    def model_type(self) -> ModelType:
        return DYNAMIC_OBSTACLE

    @property
    def model_params(self) -> ModelParameters:
        return self.op

    def get_extra_collision_friction_acc(self) -> Tuple[float, float, float]:
        raise NotImplementedError()
