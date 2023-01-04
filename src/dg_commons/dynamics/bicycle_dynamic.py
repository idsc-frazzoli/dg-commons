import math
from dataclasses import replace
from decimal import Decimal as D
from typing import FrozenSet, Mapping, TypeVar

import numpy as np
from scipy.integrate import solve_ivp

from dg_commons import Timestamp, U, X
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters

__all__ = ["BicycleDynamics"]

SR = TypeVar("SR")


# todo this is temporary


class BicycleDynamics:
    def __init__(self, vg: VehicleGeometry, vp: VehicleParameters):
        self.vg: VehicleGeometry = vg
        self.vp: VehicleParameters = vp
        """ The vehicle parameters"""

    def all_actions(self) -> FrozenSet[U]:
        pass

    def successors(self, x: VehicleState, u0: VehicleCommands, dt: D = None) -> Mapping[VehicleCommands, VehicleState]:
        """For each state, returns a dictionary U -> Possible Xs"""
        # todo
        pass

    def successor(self, x0: VehicleState, u: VehicleCommands, dt: Timestamp) -> VehicleState:
        """Perform Euler forward integration to propagate state using actions for time dt.
        This method is very inaccurate for integration steps above 0.1[s]"""
        dt = float(dt)
        # input constraints
        acc = float(np.clip(u.acc, self.vp.acc_limits[0], self.vp.acc_limits[1]))
        ddelta = float(np.clip(u.ddelta, -self.vp.ddelta_max, self.vp.ddelta_max))

        state_rate = self.dynamics(x0, replace(u, acc=acc, ddelta=ddelta))
        x0 += state_rate * dt

        # state constraints
        vx = float(np.clip(x0.vx, self.vp.vx_limits[0], self.vp.vx_limits[1]))
        delta = float(np.clip(x0.delta, -self.vp.delta_max, self.vp.delta_max))

        new_state = replace(x0, vx=vx, delta=delta)
        return new_state

    def successor_ivp(self, x0: VehicleState, u: VehicleCommands, dt: Timestamp) -> VehicleState:
        """
        Perform initial value problem integration to propagate state using actions for time dt
        """

        def _stateactions_from_array(y: np.ndarray) -> [VehicleState, VehicleCommands]:
            n_states = VehicleState.get_n_states()
            state = VehicleState.from_array(y[0:n_states])
            actions = VehicleCommands(
                acc=y[VehicleCommands.idx["acc"] + n_states], ddelta=y[VehicleCommands.idx["ddelta"] + n_states]
            )
            return state, actions

        def _dynamics(t, y):
            state0, actions = _stateactions_from_array(y=y)
            dx = self.dynamics(x0=state0, u=actions)
            du = np.zeros([len(VehicleCommands.idx)])
            return np.concatenate([dx.as_ndarray(), du])

        state_np = x0.as_ndarray()
        input_np = u.as_ndarray()
        y0 = np.concatenate([state_np, input_np])
        result = solve_ivp(fun=_dynamics, t_span=(0.0, float(dt)), y0=y0)

        if not result.success:
            raise RuntimeError("Failed to integrate ivp!")
        new_state, _ = _stateactions_from_array(result.y[:, -1])
        return new_state

    def constrained_successor_ivp(self, x0: VehicleState, u: VehicleCommands, dt: Timestamp) -> VehicleState:
        """Perform initial value problem integration to propagate state using actions for time dt.
        State constraints are applied."""
        total_time = float(dt)

        acc = float(np.clip(u.acc, self.vp.acc_limits[0], self.vp.acc_limits[1]))
        ddelta = float(np.clip(u.ddelta, -self.vp.ddelta_max, self.vp.ddelta_max))
        u = replace(u, acc=acc, ddelta=ddelta)

        v_init = x0.vx
        delta_init = x0.delta

        if acc == 0:
            v_limit = None
        elif acc > 0:
            v_limit = self.vp.vx_limits[1]
        else:
            v_limit = self.vp.vx_limits[0]

        # Time to reach the limit
        if v_limit is not None:
            t_v_limit = (v_limit - v_init) / acc
        else:
            t_v_limit = total_time

        if ddelta == 0:
            delta_limit = None
        elif ddelta > 0:
            delta_limit = self.vp.delta_max
        else:
            delta_limit = -self.vp.delta_max

        # Time to reach the limit
        if delta_limit is not None:
            t_delta_limit = (delta_limit - delta_init) / ddelta
        else:
            t_delta_limit = total_time

        if t_v_limit >= total_time and t_delta_limit >= total_time:
            return self.successor_ivp(x0, u, dt)
        elif t_v_limit < total_time and t_delta_limit >= total_time:
            x_temp = self.successor_ivp(x0, u, t_v_limit)
            return self.successor_ivp(x_temp, replace(u, acc=0.0), total_time - t_v_limit)
        elif t_v_limit >= total_time and t_delta_limit < total_time:
            x_temp = self.successor_ivp(x0, u, t_delta_limit)
            return self.successor_ivp(x_temp, replace(u, ddelta=0.0), total_time - t_delta_limit)
        else:
            if t_v_limit < t_delta_limit:
                x_temp = self.successor_ivp(x0, u, t_v_limit)
                u = replace(u, acc=0.0)
                x_temp = self.successor_ivp(x_temp, u, t_delta_limit - t_v_limit)
                return self.successor_ivp(x_temp, replace(u, ddelta=0.0), total_time - t_delta_limit)
            else:
                x_temp = self.successor_ivp(x0, u, t_delta_limit)
                u = replace(u, ddelta=0.0)
                x_temp = self.successor_ivp(x_temp, u, t_v_limit - t_delta_limit)
                return self.successor_ivp(x_temp, replace(u, acc=0.0), total_time - t_v_limit)

    def dynamics(self, x0: VehicleState, u: VehicleCommands) -> VehicleState:
        """Get rate of change of states for given control inputs"""

        dx = x0.vx
        dtheta = dx * math.tan(x0.delta) / self.vg.wheelbase
        dy = dtheta * self.vg.lr
        costh = math.cos(x0.psi)
        sinth = math.sin(x0.psi)
        xdot = dx * costh - dy * sinth
        ydot = dx * sinth + dy * costh
        x_rate = VehicleState(x=xdot, y=ydot, psi=dtheta, vx=u.acc, delta=u.ddelta)
        return x_rate

    def get_shared_resources(self, x: X) -> FrozenSet[SR]:
        pass
