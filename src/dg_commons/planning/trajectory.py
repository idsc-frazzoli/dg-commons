from dataclasses import replace
from functools import partial
from itertools import product
from typing import List, Optional, Type, Mapping, Set, Iterator, Tuple

import numpy as np
from geometry import xytheta_from_SE2
from networkx import DiGraph, is_directed_acyclic_graph, all_simple_paths, descendants, has_path, shortest_path

from dg_commons import SE2Transform, PlayerName, X
from dg_commons.seq.sequence import DgSampledSequence, iterate_with_dt, Timestamp
from dg_commons.sim.models import extract_pose_from_state
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands

__all__ = ["Trajectory", "JointTrajectories", "TrajectoryGraph", "commands_plan_from_trajectory", "TimedVehicleState"]


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
        if self.is_empty():
            return True
        diff = self.at(self.get_end()) - other.at(other.get_start())
        return abs(diff.vx) < tol and abs(diff.delta) < tol

    def is_empty(self):
        return len(self.timestamps) == 0 and len(self.values) == 0

    def __add__(self, other: Optional["Trajectory"]) -> "Trajectory":
        assert self.is_connectable(other)
        values = list(self.values)
        timestamps=list(self.timestamps)
        if self.is_empty():
            other_val = list(other.values)
            other_timestamps = (list(other.timestamps))
        else:
            other_val = list(other.values[1:])
            other_timestamps = (list(other.timestamps[1:]))
        values.append(other_val)
        timestamps.append(other_timestamps)
        return Trajectory(values=values[0], timestamps=timestamps[0])

    def upsample(self, n: int) -> "Trajectory":
        """
        Add n points between each subsequent point in the original trajectory by interpolation
        """
        timestamps = list(self.timestamps)
        up_values: List[VehicleState] = []
        up_timestamps: List[Timestamp] = []

        for t_previous, t_next in zip(timestamps, timestamps[1:]):
            up_timestamps.append(t_previous)
            up_values.append(self.at(t_previous))
            assert t_next > t_previous
            dt = (t_next - t_previous) / (n + 1)
            for i in range(1, n + 1):
                up_timestamps.append(i * dt + t_previous)
                up_values.append(self.at_interp(i * dt + t_previous))

        # append last element of original sequence
        up_timestamps.append(timestamps[-1])
        up_values.append(self.at(timestamps[-1]))

        return Trajectory(values=up_values, timestamps=up_timestamps)


JointTrajectories = Mapping[PlayerName, Trajectory]


def commands_plan_from_trajectory(trajectory: Trajectory) -> DgSampledSequence[VehicleCommands]:
    timestamps = []
    commands = []
    for t0, _, dt, v0, v1 in iterate_with_dt(trajectory):
        timestamps.append(t0)
        commands.append(VehicleCommands(acc=(v1.vx - v0.vx) / dt, ddelta=(v1.delta - v0.delta) / dt))
    return DgSampledSequence[VehicleCommands](timestamps, commands)


TimedVehicleState = Tuple[Timestamp, VehicleState]


class TrajectoryGraph(DiGraph):
    # https://networkx.org/documentation/stable/reference/algorithms/dag.html
    # pass

    def add_node(self, timed_state: TimedVehicleState, **attr):
        super(TrajectoryGraph, self).add_node(node_for_adding=timed_state, **attr)

    def check_node(self, node: TimedVehicleState):
        if node not in self.nodes:
            raise ValueError(f"{node} not in graph!")

    def add_edge(self, states: Trajectory, transition: Trajectory, **attr):
        source, target = states.at(states.get_start()), states.at(states.get_end())
        start_time, end_time = states.get_start(), states.get_end()
        self.check_node(node=(start_time, source))
        attr["transition"] = transition
        if target not in self.nodes:
            self.add_node(timed_state=(end_time, target), gen=self.nodes.get((start_time, source))["gen"] + 1)

        super(TrajectoryGraph, self).add_edge(u_of_edge=(start_time, source), v_of_edge=(end_time, target), **attr)
        return
        # self.trajectories[(source, target)] = trajectory

    def get_all_trajectories(self) -> Set[Trajectory]:
        assert is_directed_acyclic_graph(self)

        trajectories = set()

        roots = [node for node, degree in self.in_degree() if degree == 0]
        assert len(roots) == 1
        source = roots[0]
        leaves = [node for node, degree in self.out_degree() if degree == 0]

        for target in leaves:
            trajectories.add(self.get_trajectory(source=source, target=target))
        return trajectories

    def get_trajectory(self, source: TimedVehicleState, target: TimedVehicleState) -> Trajectory:
        self.check_node(source)
        self.check_node(target)
        if not has_path(G=self, source=source, target=target):
            raise ValueError(f"No path exists between {source, target}!")

        nodes = shortest_path(G=self, source=source, target=target)
        traj: Trajectory = Trajectory(values=[], timestamps=[])
        for node1, node2 in zip(nodes[:-1], nodes[1:]):
            traj += self.get_edge_data(u=node1, v=node2)["transition"]
        return traj

    def iterate_all_trajectories(self) -> Iterator[Trajectory]:
        all_traj = self.get_all_trajectories()
        # todo
        pass
