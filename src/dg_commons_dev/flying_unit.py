from dataclasses import dataclass, replace
import numpy as np
from frozendict import frozendict
from dg_commons.sim.models.vehicle import VehicleState
import math
from scipy.integrate import solve_ivp


@dataclass(unsafe_hash=True, eq=True, order=True)
class FlyingUnit:
    x: float
    """ CoG x location [m] """
    y: float
    """ CoG y location [m] """
    theta: float
    """ CoG heading [rad] """
    vx: float
    """ CoG longitudinal velocity [m/s] """
    idx = frozendict({"x": 0, "y": 1, "theta": 2, "vx": 3})
    """ Dictionary to get correct values from numpy arrays"""

    @classmethod
    def get_n_states(cls) -> int:
        return len(cls.idx)

    def __add__(self, other: "FlyingUnit") -> "FlyingUnit":
        if type(other) == type(self):
            return replace(
                self,
                x=self.x + other.x,
                y=self.y + other.y,
                theta=self.theta + other.theta,
                vx=self.vx + other.vx,
            )
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other: "FlyingUnit") -> "FlyingUnit":
        return self + (other * -1.0)

    def __mul__(self, val: float) -> "FlyingUnit":
        return replace(
            self,
            x=self.x * val,
            y=self.y * val,
            theta=self.theta * val,
            vx=self.vx * val,
        )

    __rmul__ = __mul__

    def __truediv__(self, val: float) -> "FlyingUnit":
        return self * (1 / val)

    def __repr__(self) -> str:
        return str({k: round(float(v), 2) for k, v in self.__dict__.items() if not k.startswith("idx")})

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.vx])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_states() == z.size == z.shape[0], f"z vector {z} cannot initialize VehicleState."
        return FlyingUnit(
            x=z[cls.idx["x"]],
            y=z[cls.idx["y"]],
            theta=z[cls.idx["theta"]],
            vx=z[cls.idx["vx"]],
        )

    @classmethod
    def from_vehicle_state(cls, z: VehicleState):
        return FlyingUnit(
            x=z.x,
            y=z.y,
            theta=z.theta,
            vx=z.vx,
        )


def dynamics(x0: FlyingUnit,  dtheta: float) -> np.ndarray:
    """ Const velocity and angle variation dynamics """
    v = x0.vx
    cos_th = math.cos(x0.theta)
    sin_th = math.sin(x0.theta)

    x_dot = v * cos_th
    y_dot = v * sin_th
    theta_dot = dtheta

    return np.array([x_dot, y_dot, theta_dot])


def const_variation_integration(x0: FlyingUnit, dtheta: float, time_span: float):
    vel = x0.vx

    def _dynamics(t, y) -> np.ndarray:
        state0 = FlyingUnit(x=y[0], y=y[1], theta=y[2], vx=vel)
        dy = dynamics(x0=state0, dtheta=dtheta)
        return dy

    y0 = np.array([x0.x, x0.y, x0.theta])
    result = solve_ivp(fun=_dynamics, t_span=(0.0, float(time_span)), y0=y0)

    if not result.success:
        raise RuntimeError("Failed to integrate ivp!")
    return result.y[:, -1]
