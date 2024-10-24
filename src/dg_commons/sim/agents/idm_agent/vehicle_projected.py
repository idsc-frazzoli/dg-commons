from dataclasses import dataclass

import numpy as np
from frozendict import frozendict

from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn


@dataclass(unsafe_hash=True, eq=True, order=True)
class VehicleStatePrj(VehicleStateDyn):
    # The intersection point
    int_point: tuple[float, float] = (0, 0)
    idx = frozendict({"x": 0, "y": 1, "psi": 2, "vx": 3, "vy": 4, "dpsi": 5, "delta": 6, "int_point": 7})

    @classmethod
    def from_vehicle_state_dyn(cls, state_dyn: VehicleStateDyn, int_point: np.ndarray):
        return cls(
            x=state_dyn.x,
            y=state_dyn.y,
            psi=state_dyn.psi,
            vx=state_dyn.vx,
            vy=state_dyn.vy,
            dpsi=state_dyn.dpsi,
            delta=state_dyn.delta,
            int_point=tuple(int_point),
        )

    @classmethod
    def from_vehicle_state(cls, state: VehicleState, int_point: np.ndarray):
        return cls(
            x=state.x,
            y=state.y,
            psi=state.psi,
            vx=state.vx,
            vy=0.0,
            dpsi=0.0,
            delta=state.delta,
            int_point=tuple(int_point),
        )

    def to_vehicle_state(self) -> VehicleState:
        return VehicleState(x=self.x, y=self.y, psi=self.psi, vx=self.vx, delta=self.delta)
