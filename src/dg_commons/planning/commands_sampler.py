from dataclasses import dataclass
from decimal import Decimal
from itertools import product
from typing import Set, List, Callable, Optional

import numpy as np

from dg_commons import Timestamp, logger, LinSpaceTuple
from dg_commons.planning.trajectory import Trajectory
from dg_commons.planning.trajectory_generator_abc import TrajGenerator
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands
from dg_commons.sim.models.vehicle_utils import VehicleParameters

__all__ = ["CommandsSampler", "CommandsSamplerParam"]


@dataclass
class CommandsSamplerParam:
    dt: Decimal
    n_steps: int
    acc: LinSpaceTuple
    steer_rate: LinSpaceTuple

    def __post_init__(self):
        assert isinstance(self.dt, Decimal)
        # todo in the future more n_steps
        assert self.n_steps == 1

    @classmethod
    def from_vehicle_parameters(
        cls, dt: Decimal, n_steps: int, n_acc: int, n_steer_rate: int, vp: VehicleParameters
    ) -> "CommandsSamplerParam":
        """
        :param dt:
        :param n_steps:
        :param n_acc:
        :param n_steer_rate:
        :param vp:
        :return:
        """
        vel_linspace = (vp.vx_limits[0], vp.vx_limits[1], n_acc)
        steer_linspace = (-vp.delta_max, vp.delta_max, n_steer_rate)
        return CommandsSamplerParam(dt=dt, n_steps=n_steps, acc=vel_linspace, steer_rate=steer_linspace)


class CommandsSampler(TrajGenerator):
    def __init__(
        self,
        param: CommandsSamplerParam,
        vehicle_dynamics: Callable[[VehicleState, VehicleCommands, Timestamp], VehicleState],
        vehicle_param: VehicleParameters,
    ):
        super(CommandsSampler, self).__init__(vehicle_dynamics=vehicle_dynamics, vehicle_param=vehicle_param)
        self._param: Optional[CommandsSamplerParam] = None
        self.update_params(param)

    def generate(self, x0: VehicleState) -> Set[Trajectory]:
        acc_samples, dsteer_samples = self.generate_samples()
        trajectories = set()
        for acc, steer_rate in product(acc_samples, dsteer_samples):
            timestamps = [
                Decimal(0),
            ]
            states = [
                x0,
            ]
            next_state = x0
            cmds = VehicleCommands(acc=acc, ddelta=steer_rate)
            next_state = self.vehicle_dynamics(next_state, cmds, float(self._param.dt))
            timestamps.append(self._param.dt)
            states.append(next_state)
            trajectories.add(Trajectory(timestamps=timestamps, values=states))
        logger.info(f"{type(self).__name__}:Generated {len(trajectories)}")
        return trajectories

    def update_params(self, param: CommandsSamplerParam):
        assert self.vehicle_param.acc_limits[0] <= param.acc[0] <= param.acc[1] <= self.vehicle_param.acc_limits[1]
        assert (
            -self.vehicle_param.ddelta_max
            <= param.steer_rate[0]
            <= param.steer_rate[1]
            <= self.vehicle_param.ddelta_max
        )
        self._param = param

    def generate_samples(self) -> (List, List):
        acc_samples = np.linspace(*self._param.acc)
        dsteer_samples = np.linspace(*self._param.steer_rate)
        return acc_samples, dsteer_samples
