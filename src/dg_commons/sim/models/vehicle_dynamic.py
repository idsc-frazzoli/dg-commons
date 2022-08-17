import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from frozendict import frozendict
from geometry import T2value, SO2_from_angle, SO2value

from dg_commons.sim.models import Pacejka4p, Pacejka
from dg_commons.sim.models.model_structures import TwoWheelsTypes
from dg_commons.sim.models.model_utils import apply_full_acceleration_limits
from dg_commons.sim.models.utils import G, rho
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState, VehicleModel
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import steering_constraint, VehicleParameters


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleStateDyn(VehicleState):
    vy: float = 0
    """ CoG longitudinal velocity [m/s] """
    dpsi: float = 0
    """ yaw rate """
    idx = frozendict({"x": 0, "y": 1, "psi": 2, "vx": 3, "vy": 4, "dpsi": 5, "delta": 6})
    """ Dictionary to get correct values from numpy arrays"""

    def __add__(self, other: "VehicleStateDyn") -> "VehicleStateDyn":
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
            )
        else:
            raise NotImplementedError

    def __mul__(self, val: float) -> "VehicleStateDyn":
        return replace(
            self,
            x=self.x * val,
            y=self.y * val,
            psi=self.psi * val,
            vx=self.vx * val,
            vy=self.vy * val,
            dpsi=self.dpsi * val,
            delta=self.delta * val,
        )

    def as_ndarray(self) -> np.ndarray:
        return np.array([self.x, self.y, self.psi, self.vx, self.vy, self.dpsi, self.delta])

    @classmethod
    def from_array(cls, z: np.ndarray):
        assert cls.get_n_states() == z.size == z.shape[0], f"z vector {z} cannot initialize VehicleState."
        return VehicleStateDyn(
            x=z[cls.idx["x"]],
            y=z[cls.idx["y"]],
            psi=z[cls.idx["psi"]],
            vx=z[cls.idx["vx"]],
            vy=z[cls.idx["vy"]],
            dpsi=z[cls.idx["dpsi"]],
            delta=z[cls.idx["delta"]],
        )

    def to_vehicle_state(self) -> VehicleState:
        return VehicleState(x=self.x, y=self.y, psi=self.psi, vx=self.vx, delta=self.delta)


class VehicleModelDyn(VehicleModel):
    def __init__(
        self,
        x0: VehicleStateDyn,
        vg: VehicleGeometry,
        vp: VehicleParameters,
        pacejka_front: Pacejka,
        pacejka_rear: Pacejka,
    ):
        """
        Single track dynamic model
        :param x0:
        :param vg:
        :param vp:
        """
        super(VehicleModelDyn, self).__init__(x0, vg, vp)
        # """ The vehicle's geometry parameters"""
        self.vp: VehicleParameters = vp
        """ The vehicle parameters"""
        self.vg: VehicleGeometry = vg
        """ The vehicle's geometry params"""
        self.pacejka_front: Pacejka = pacejka_front
        """ The vehicle tyre model"""
        self.pacejka_rear: Pacejka = pacejka_rear
        """ Kinematic model option"""
        self.kin: bool = False

    @classmethod
    def default_bicycle(cls, x0: VehicleStateDyn):
        return VehicleModelDyn(
            x0=x0,
            vg=VehicleGeometry.default_bicycle(),
            vp=VehicleParameters.default_bicycle(),
            pacejka_front=Pacejka4p.default_bicycle_front(),
            pacejka_rear=Pacejka4p.default_bicycle_rear(),
        )

    @classmethod
    def default_car(cls, x0: VehicleStateDyn):
        return VehicleModelDyn(
            x0=x0,
            vg=VehicleGeometry.default_car(),
            vp=VehicleParameters.default_car(),
            pacejka_front=Pacejka4p.default_car_front(),
            pacejka_rear=Pacejka4p.default_car_rear(),
        )

    @classmethod
    def default_truck(cls, x0: VehicleStateDyn):
        return VehicleModelDyn(
            x0=x0,
            vg=VehicleGeometry.default_truck(),
            vp=VehicleParameters.default_truck(),
            pacejka_front=Pacejka4p.default_truck_front(),
            pacejka_rear=Pacejka4p.default_truck_rear(),
        )

    def dynamics_kin(self, x0: VehicleStateDyn, u: VehicleCommands) -> VehicleStateDyn:
        """Kinematic bicycle model with reference point at the center of mass, used for singularity"""
        # friction model
        frictionx, frictiony, frictiontheta = self.get_extra_collision_friction_acc()

        vx = x0.vx
        dtheta = vx * math.tan(x0.delta) / self.vg.wheelbase
        vy = dtheta * self.vg.lr
        costh = math.cos(x0.psi)
        sinth = math.sin(x0.psi)
        xdot = vx * costh - vy * sinth
        ydot = vx * sinth + vy * costh

        ddelta = steering_constraint(x0.delta, u.ddelta, self.vp)
        acc = apply_full_acceleration_limits(x0.vx, u.acc*0.5, self.vp)  # address continue feasibility issue

        ddtheta = (acc * math.tan(x0.delta) + vx * ddelta / (math.cos(x0.delta)**2)) / self.vg.wheelbase
        lat_acc = ddtheta * self.vg.lr

        return VehicleStateDyn(x=xdot, y=ydot, psi=dtheta, vx=acc + frictionx, vy=lat_acc + frictiony, dpsi=ddtheta + frictiontheta, delta=ddelta)

    def dynamics(self, x0: VehicleStateDyn, u: VehicleCommands) -> VehicleStateDyn:
        """returns state derivative for given control inputs"""
        # friction model
        frictionx, frictiony, frictiontheta = self.get_extra_collision_friction_acc()

        if x0.vx < 0.1 or self.kin is True:
            self.kin = True
            dx_kin = self.dynamics_kin(x0, u)
            if abs(x0.delta) < 0.2:
                self.kin = False
            return dx_kin
        else:
            m = self.vg.m
            acc = apply_full_acceleration_limits(x0.vx, u.acc, self.vp)
            ddelta = steering_constraint(x0.delta, u.ddelta, self.vp)

            # Vertical contact forces (positive sign)
            load_transfer = self.vg.h_cog * acc
            F1_n = m * (G * self.vg.lr - load_transfer) / self.vg.wheelbase
            F2_n = m * (G * self.vg.lf + load_transfer) / self.vg.wheelbase

            # Front wheel velocities
            rot_delta = SO2_from_angle(-x0.delta)
            vel_1_tyre = rot_delta @ np.array([x0.vx, x0.vy + self.vg.lf * x0.dpsi])
            slip_angle_1 = math.atan(vel_1_tyre[1] / vel_1_tyre[0])
            # Back wheel velocities
            vel_2_tyre = np.array([x0.vx, x0.vy - self.vg.lr * x0.dpsi])
            slip_angle_2 = math.atan(vel_2_tyre[1] / vel_2_tyre[0])

            # Rolling resistance
            F_rr_f = -self.vg.c_rr_f * F1_n * np.sign(vel_1_tyre[0])
            F_rr_r = -self.vg.c_rr_r * F2_n * np.sign(vel_2_tyre[0])

            # Torque splitting
            Facc1_tyre, Facc2_tyre = self.get_torque_split(m * acc)
            Facc1_tyre += F_rr_f
            Facc2_tyre += F_rr_r

            # Front wheel forces (hyp: only braking or positive acceleration)
            if abs(vel_1_tyre[0]) > 0.1:
                Facc1_tyre = np.clip(Facc1_tyre, -F1_n * self.pacejka_front.D, F1_n * self.pacejka_front.D)
            else:
                Facc1_tyre = np.clip(Facc1_tyre, 0, F1_n * self.pacejka_front.D)

            F1y_sat = F1_n * self.pacejka_front.D * math.sqrt(1 - (Facc1_tyre / (F1_n * self.pacejka_front.D)) ** 2)
            F1y_tyre = np.clip(F1_n * self.pacejka_front.evaluate(-slip_angle_1), -F1y_sat, F1y_sat)
            F1 = rot_delta.T @ np.array([Facc1_tyre, F1y_tyre])

            # Back wheel forces (hyp: only braking or positive acceleration)
            if abs(vel_2_tyre[0]) > 0.1:
                Facc2_tyre = np.clip(Facc2_tyre, -F2_n * self.pacejka_rear.D, F2_n * self.pacejka_front.D)
            else:
                Facc2_tyre = np.clip(Facc2_tyre, 0, F2_n * self.pacejka_rear.D)

            F2y_sat = F2_n * self.pacejka_rear.D * math.sqrt(1 - (Facc2_tyre / (F2_n * self.pacejka_rear.D)) ** 2)
            F2y_tyre = np.clip(F2_n * self.pacejka_rear.evaluate(-slip_angle_2), -F2y_sat, F2y_sat)
            F2 = np.array([Facc2_tyre, F2y_tyre])

            # Drag Force
            F_drag = -0.5 * x0.vx * self.vg.a_drag * self.vg.c_drag * rho**2

            # longitudinal acceleration
            acc_x = (F1[0] + F2[0] + F_drag) / m + x0.dpsi * x0.vy
            # lateral acceleration
            acc_y = (F1[1] + F2[1]) / m - x0.dpsi * x0.vx
            # yaw acceleration
            ddtheta = (F1[1] * self.vg.lf - F2[1] * self.vg.lr) / self.vg.Iz

            # kinematic model
            costh = math.cos(x0.psi)
            sinth = math.sin(x0.psi)
            xdot = x0.vx * costh - x0.vy * sinth
            ydot = x0.vx * sinth + x0.vy * costh

            return VehicleStateDyn(
                x=xdot,
                y=ydot,
                psi=x0.dpsi,
                vx=acc_x + frictionx,
                vy=acc_y + frictiony,
                dpsi=ddtheta + frictiontheta,
                delta=ddelta,
            )

    def get_velocity(self, in_model_frame: bool) -> (T2value, float):
        self._state: VehicleStateDyn
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
        self._state.dtheta = omega

    def get_torque_split(self, Facc: float) -> Sequence[float]:
        """Returns split of acceleration force to be applied on front and rear wheel"""
        if Facc <= 0:
            # we partition acc 60% front 40% rear while braking
            return Facc * 0.6, Facc * 0.4
        else:
            if self.vg.vehicle_type in TwoWheelsTypes:
                # only rear for acc on bicycles-like
                return 0, Facc
            else:
                # assumes 4WD car
                return Facc * 0.5, Facc * 0.5

    def get_extra_collision_friction_acc(self):
        magic_mu = 3.0 if self.model_type in TwoWheelsTypes else 1.0
        if self.has_collided:  # and self.model_type in TwoWheelsTypes:
            frictionx = -magic_mu * self._state.vx
            frictiony = -magic_mu * self._state.vy
            frictiontheta = -magic_mu * self._state.dpsi
            return frictionx, frictiony, frictiontheta
        else:
            return 0, 0, 0

    @property
    def model_geometry(self) -> VehicleGeometry:
        return self.vg
