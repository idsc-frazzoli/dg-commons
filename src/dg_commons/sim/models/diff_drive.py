from dataclasses import dataclass, replace
from decimal import Decimal
from math import cos, sin
from typing import Type, Mapping

import numpy as np
from frozendict import frozendict
from geometry import SE2value, SE2_from_xytheta, T2value
from scipy.integrate import solve_ivp
from shapely.geometry import Polygon

from dg_commons import apply_SE2_to_shapely_geo, PoseState
from dg_commons.sim import logger, ImpactLocation, IMPACT_RIGHT, IMPACT_LEFT, IMPACT_BACK, IMPACT_FRONT
from dg_commons.sim.models import ModelType
from dg_commons.sim.models.diff_drive_structures import *
from dg_commons.sim.simulator_structures import SimModel


@dataclass(unsafe_hash=True, eq=True, order=True)
class DiffDriveCommands:
    omega_l: float
    """ left wheel spinning rate [rad/s]. Positive is forward."""
    omega_r: float
    """ right wheel spinning rate [rad/s]. Positive is forward. """
    idx = frozendict({"omega_l": 0, "omega_r": 1})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_commands(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "DiffDriveCommands") -> "DiffDriveCommands":
        if type(other) == type(self):
            return replace(self, omega_l=self.omega_l + other.omega_l, omega_r=self.omega_r + other.omega_r)
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "DiffDriveCommands") -> "DiffDriveCommands":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "DiffDriveCommands":
        return replace(self, omega_l=self.omega_l * val, omega_r=self.omega_r * val)

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "DiffDriveCommands":
        return self * (1 / val)

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.omega_l, self.omega_r])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_commands() == z.size == z.shape[0], f"z vector {z} cannot initialize DiffDriveInputs."
        return DiffDriveCommands(omega_l=z[cls.idx["omega_l"]], omega_r=z[cls.idx["omega_r"]])


@dataclass(unsafe_hash=True, eq=True, order=True)
class DiffDriveState(PoseState):
    """State for a bicycle model like vehicle"""

    x: float
    """ CoG x location [m] """
    y: float
    """ CoG y location [m] """
    psi: float
    """ CoG heading [rad] """
    idx = frozendict({"x": 0, "y": 1, "psi": 2})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_states(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "DiffDriveState") -> "DiffDriveState":
        if type(other) == type(self):
            return replace(self, x=self.x + other.x, y=self.y + other.y, psi=self.psi + other.psi)
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "DiffDriveState") -> "DiffDriveState":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "DiffDriveState":
        return replace(self, x=self.x * val, y=self.y * val, psi=self.psi * val)

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "DiffDriveState":
        return self * (1 / val)

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("idx")})

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_states() == z.size == z.shape[0], f"z vector {z} cannot initialize DiffDriveState."
        return DiffDriveState(x=z[cls.idx["x"]], y=z[cls.idx["y"]], psi=z[cls.idx["psi"]])


class DiffDriveModel(SimModel[DiffDriveState, DiffDriveCommands]):
    def __init__(self, x0: DiffDriveState, vg: DiffDriveGeometry, vp: DiffDriveParameters):
        self._state: DiffDriveState = x0
        """ Current state of the model"""
        self.XT: Type[DiffDriveState] = type(x0)
        """ State type"""
        self.vg: DiffDriveGeometry = vg
        """ The vehicle's geometry parameters"""
        self.vp: DiffDriveParameters = vp
        """ The vehicle parameters"""

    @classmethod
    def default(cls, x0: DiffDriveState) -> "DiffDriveModel":
        return DiffDriveModel(x0=x0, vg=DiffDriveGeometry.default(), vp=DiffDriveParameters.default())

    def update(self, commands: DiffDriveCommands, dt: Decimal):
        """
        Perform initial value problem integration
        to propagate state using actions for time dt
        """

        def _stateactions_from_array(y: np.ndarray) -> [DiffDriveState, DiffDriveCommands]:
            n_states = self.XT.get_n_states()
            state = self.XT.from_array(y[0:n_states])
            if self.has_collided:
                actions = DiffDriveCommands(omega_l=0, omega_r=0)
            else:
                actions = DiffDriveCommands.from_array(y[n_states:])
            return state, actions

        def _dynamics(t, y):
            state0, actions = _stateactions_from_array(y=y)
            dx = self.dynamics(x0=state0, u=actions)
            du = np.zeros([len(DiffDriveCommands.idx)])
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

    def dynamics(self, x0: DiffDriveState, u: DiffDriveCommands) -> DiffDriveState:
        """Kinematic bicycle model, returns state derivative for given control inputs"""

        # apply wheels constraints
        omega_l = np.clip(u.omega_l, self.vp.omega_limits[0], self.vp.omega_limits[1])
        omega_r = np.clip(u.omega_r, self.vp.omega_limits[0], self.vp.omega_limits[1])

        omega_sum = omega_l + omega_r
        costh = cos(x0.psi)
        sinth = sin(x0.psi)
        xdot = costh * omega_sum / 2
        ydot = sinth * omega_sum / 2
        dpsi = (omega_r - omega_l) / self.vg.wheelbase

        return DiffDriveState(x=xdot, y=ydot, psi=dpsi) * self.vg.wheelradius

    def get_footprint(self) -> Polygon:
        """Returns current footprint of the vehicle (mainly for collision checking)"""
        footprint = self.vg.outline_as_polygon
        transform = self.get_pose()
        return apply_SE2_to_shapely_geo(footprint, transform)

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

    @property
    def model_geometry(self) -> DiffDriveGeometry:
        return self.vg

    def get_velocity(self, in_model_frame: bool) -> (T2value, float):
        """Returns velocity at COG"""
        return [0, 0], 0

    def set_velocity(self, vel: T2value, omega: float, in_model_frame: bool):
        logger.warn(
            "It is NOT possible to set the lateral and rotational velocity for the Differential Drive model\n"
            "Try using a dynamic model."
        )

    @property
    def model_type(self) -> ModelType:
        return self.vg.vehicle_type

    @property
    def model_params(self) -> DiffDriveParameters:
        return self.vp

    def get_extra_collision_friction_acc(self):
        # this model is not dynamic
        pass
