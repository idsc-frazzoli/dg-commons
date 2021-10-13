from dataclasses import replace
from functools import partial
from typing import List, Optional, Type, Mapping, Set, Iterator

import numpy as np
from geometry import xytheta_from_SE2
from networkx import DiGraph

from dg_commons import SE2Transform, PlayerName, X
from dg_commons.seq.sequence import DgSampledSequence, iterate_with_dt
from dg_commons.sim.models import extract_pose_from_state
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands

__all__ = ["Trajectory", "JointTrajectories", "TrajectoryGraph", "commands_plan_from_trajectory"]


class Trajectory(DgSampledSequence[VehicleState]):
    """Container for a trajectory as a sampled sequence"""

    @property
    def XT(self) -> Type[X]:
        return VehicleState

    def as_path(self) -> List[SE2Transform]:
        """Returns cartesian coordinates (SE2) of transition states"""
        return [SE2Transform(p=np.array([x.x, x.y]), theta=x.theta) for x in self.values]

    def apply_SE2transform(self, transform: SE2Transform):
        def _applySE2(x: VehicleState, t: SE2Transform) -> VehicleState:
            pose = extract_pose_from_state(x)
            new_pose = np.dot(t.as_SE2(), pose)
            xytheta = xytheta_from_SE2(new_pose)
            return replace(x, x=xytheta[0], y=xytheta[1], theta=xytheta[2])

        f = partial(_applySE2, t=transform)
        return self.transform_values(f=f, YT=VehicleState)

    def is_connectable(self, other: "Trajectory", tol=1e-3) -> bool:
        """
        Any primitive whose initial state's velocity and steering angle are equal to those of the current primitive is
        deemed connectable.

        :param other: the motion primitive to which the connectivity is examined
        """
        diff = self.at(self.get_end()) - other.at(other.get_end())
        return abs(diff.vx) < tol and abs(diff.delta) < tol

    def __add__(self, other: Optional["Trajectory"]) -> "Trajectory":
        assert self.is_connectable(other)
        # todo
        pass

    def upsample(self, n: int) -> "Trajectory":
        """"""
        # todo
        pass


JointTrajectories = Mapping[PlayerName, Trajectory]


def commands_plan_from_trajectory(trajectory: Trajectory) -> DgSampledSequence[VehicleCommands]:
    timestamps = []
    commands = []
    for t0, _, dt, v0, v1 in iterate_with_dt(trajectory):
        timestamps.append(t0)
        commands.append(VehicleCommands(acc=(v1.vx - v0.vx) / dt, ddelta=(v1.delta - v0.delta) / dt))
    return DgSampledSequence[VehicleCommands](timestamps, commands)


class TrajectoryGraph(DiGraph):
    # https://networkx.org/documentation/stable/reference/algorithms/dag.html
    pass
    # todo missing sampling time to have proper trajectories

    # todo
    def get_all_trajectories(self) -> Set[Trajectory]:
        pass

    def iterate_all_trajectories(self) -> Iterator[Trajectory]:
        pass
