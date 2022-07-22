from dataclasses import replace
from functools import partial
from typing import List, Type, Mapping, Set, Iterator, Tuple

import numpy as np
from geometry import xytheta_from_SE2
from networkx import DiGraph, is_directed_acyclic_graph, has_path, shortest_path

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
        return [SE2Transform(p=np.array([x.x, x.y]), theta=x.psi) for x in self.values]

    def apply_SE2transform(self, transform: SE2Transform):
        def _applySE2(x: VehicleState, t: SE2Transform) -> VehicleState:
            pose = extract_pose_from_state(x)
            new_pose = np.dot(t.as_SE2(), pose)
            xytheta = xytheta_from_SE2(new_pose)
            return replace(x, x=xytheta[0], y=xytheta[1], psi=xytheta[2])

        f = partial(_applySE2, t=transform)
        return self.transform_values(f=f, YT=VehicleState)

    def is_connectable(self, other: "Trajectory", tol: float = 1e-3) -> bool:
        """
        Any primitive whose initial state's velocity and steering angle are equal to those of the current primitive is
        deemed connectable.
        """
        if self.is_empty() or other.is_empty():
            return True
        diff = self.at(self.get_end()) - other.at(other.get_start())
        return abs(diff.vx) < tol and abs(diff.delta) < tol

    def is_empty(self):
        return len(self.timestamps) == 0 and len(self.values) == 0

    def merge(self, other: "Trajectory") -> "Trajectory":
        """It is checked that speed and steering angle are consistent
        between the last state of a trajectory and the first state of the other trajectory."""
        assert self.is_connectable(other)
        values = list(self.values)
        timestamps = list(self.timestamps)
        if self.is_empty():
            other_val = list(other.values)
            other_timestamps = list(other.timestamps)
        else:
            other_val = list(other.values[1:])
            other_timestamps = list(other.timestamps[1:])
        values = values + other_val
        timestamps = timestamps + other_timestamps
        return Trajectory(values=values, timestamps=timestamps)

    # def scalar_multiply(self, scalar: float) -> "Trajectory":
    #     """
    #     Multiply all values of a trajectory by a scalar. x, y, theta, vx and delta will each be
    #     multiplied by the same scalar.
    #     """
    #     values = [value * scalar for value in self.values]
    #     return Trajectory(values=values, timestamps=self.timestamps)
    #
    # def __add__(self, other: "Trajectory") -> "Trajectory":
    #     """
    #     Sum two trajectories, value by value.
    #     """
    #     assert self.timestamps == other.timestamps, "The timestamps must be equal sum the values of two trajectories."
    #     values = []
    #     for i, _ in enumerate(self.timestamps):
    #         values.append(self.values[i] + other.values[i])
    #
    #     return Trajectory(values=values, timestamps=self.timestamps)
    #
    # def __sub__(self, other: "Trajectory") -> "Trajectory":
    #     """
    #     Subtract one trajectory from another one, value by value.
    #     """
    #     return self + other.scalar_multiply(scalar=-1.0)

    def merge_unsafe(self, other: "Trajectory", tol=1e-3) -> "Trajectory":
        """Only checks that timestamps between the end and start of the trajectories to merge are consistent."""
        if not self.is_empty():
            assert (
                abs(self.timestamps[-1] - other.timestamps[0]) <= tol
            ), "End and start timestamps of the trajectories to merge are not consistent."
        if self.is_empty():
            other_val = list(other.values)
            other_timestamps = list(other.timestamps)
        else:
            other_val = list(other.values[1:])
            other_timestamps = list(other.timestamps[1:])

        values = list(self.values) + other_val
        timestamps = list(self.timestamps) + other_timestamps
        return Trajectory(values=values, timestamps=timestamps)

    def upsample(self, n: int) -> "Trajectory":
        """
        Add n points between each subsequent point in the original trajectory by interpolation
        # todo this method can be moved to be a method of the generic DGSampledSequence
        """
        timestamps = list(self.timestamps)
        up_values: List[VehicleState] = []
        up_timestamps: List[Timestamp] = []

        for t_previous, t_next in zip(timestamps, timestamps[1:]):
            up_timestamps.append(t_previous)
            up_values.append(self.at(t_previous))
            dt = (t_next - t_previous) / (n + 1)
            for i in range(1, n + 1):
                ts = i * dt + t_previous
                up_timestamps.append(ts)
                up_values.append(self.at_interp(ts))

        # append last element of original sequence
        up_timestamps.append(timestamps[-1])
        up_values.append(self.at(timestamps[-1]))

        return Trajectory(values=up_values, timestamps=up_timestamps)

    # def squared_error(self, other: "Trajectory") -> float:
    #     """
    #     Compute the average mean squared error for x, y, theta, vx, delta between two trajectories.
    #     Then compute the average across these 5 averages.
    #     """
    #     assert self.timestamps == other.timestamps, "The timestamps must be equal to compute squared error."
    #     diff = self - other
    #     x_squared = sum([value.x * value.x for value in diff.values]) / len(diff.values)
    #     y_squared = sum([value.y * value.y for value in diff.values]) / len(diff.values)
    #     theta_squared = sum([value.psi * value.psi for value in diff.values]) / len(diff.values)
    #     vx_squared = sum([value.vx * value.vx for value in diff.values]) / len(diff.values)
    #     delta_squared = sum([value.delta * value.delta for value in diff.values]) / len(diff.values)
    #
    #     return (x_squared + y_squared + theta_squared + vx_squared + delta_squared) / 5.0


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

    def get_all_transitions(self) -> Set[Trajectory]:
        """
        Return all possible transitions on a graph. Transitions are the upsampled states between two nodes,
        stored ad edge data.
        """
        assert is_directed_acyclic_graph(self)

        trajectories = set()

        roots = [node for node, degree in self.in_degree() if degree == 0]
        assert len(roots) == 1
        source = roots[0]
        leaves = [node for node, degree in self.out_degree() if degree == 0]

        for target in leaves:
            # trajectories.add(self.get_trajectory(source=source, target=target))
            trajectories.add(self.get_transition(source=source, target=target))

        return trajectories

    def get_all_trajectories(self) -> Set[Trajectory]:
        """
        Return all possible trajectories stored in the graph nodes.
        """
        assert is_directed_acyclic_graph(self)

        trajectories = set()

        roots = [node for node, degree in self.in_degree() if degree == 0]
        assert len(roots) == 1
        source = roots[0]
        leaves = [node for node, degree in self.out_degree() if degree == 0]

        for target in leaves:
            trajectories.add(self.get_trajectory(source=source, target=target))

        return trajectories

    def commands_on_trajectory(self, trajectory: Trajectory) -> DgSampledSequence:
        """
        Retrieve sequence of commands stored as edge data, when moving on graph on a certain trajectory.
        :param trajectory: Trajectory along which to search.
        """

        source = (trajectory.timestamps[0], trajectory.values[0])
        target = (trajectory.timestamps[-1], trajectory.values[-1])

        self.check_node(source)
        self.check_node(target)
        if not has_path(G=self, source=source, target=target):
            raise ValueError(f"No path exists between {source, target}!")

        nodes = shortest_path(G=self, source=source, target=target)
        commands = []
        timestamps = []
        for node1, node2 in zip(nodes[:-1], nodes[1:]):
            commands.append(self.get_edge_data(u=node1, v=node2)["commands"])
            timestamps.append(node1[0])
        return DgSampledSequence[VehicleCommands](values=commands, timestamps=timestamps)

    def get_transition(self, source: TimedVehicleState, target: TimedVehicleState) -> Trajectory:
        """
        Compute the shortest path between source and target nodes and return as trajectory of vehicle states.
        Merging trajectories is done by only checking timestamps consistency, since the trajectories generated
        are not always continuous (usually not).
        :param source: source node
        :param target: target node
        """
        self.check_node(source)
        self.check_node(target)
        if not has_path(G=self, source=source, target=target):
            raise ValueError(f"No path exists between {source, target}!")

        nodes = shortest_path(G=self, source=source, target=target)
        traj: Trajectory = Trajectory(values=[], timestamps=[])
        for node1, node2 in zip(nodes[:-1], nodes[1:]):
            traj = traj.merge_unsafe(self.get_edge_data(u=node1, v=node2)["transition"])
        return traj

    def get_trajectory(self, source: TimedVehicleState, target: TimedVehicleState):
        """
        Get state stored in each node on shortest path from source to target.
        :param source: source node
        :param target: target node
        """
        self.check_node(source)
        self.check_node(target)
        if not has_path(G=self, source=source, target=target):
            raise ValueError(f"No path exists between {source, target}!")

        nodes = shortest_path(G=self, source=source, target=target)
        states = []
        timestamps = []
        for node in nodes:
            timestamps.append(node[0])
            states.append(node[1])
        return Trajectory(values=states, timestamps=timestamps)

    def iterate_all_trajectories(self) -> Iterator[Trajectory]:
        # todo
        pass
