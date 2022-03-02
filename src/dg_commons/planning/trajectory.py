from dataclasses import replace
from functools import partial
from itertools import product
from typing import List, Optional, Type, Mapping, Set, Iterator

import numpy as np
from geometry import xytheta_from_SE2
from networkx import DiGraph, is_directed_acyclic_graph, all_simple_paths, descendants, has_path, shortest_path

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

    def add_node(self, state: VehicleState, **attr):
        super(TrajectoryGraph, self).add_node(node_for_adding=state, **attr)

    def check_node(self, node: VehicleState):
        if node not in self.nodes:
            raise ValueError(f"{node} not in graph!")

    def add_edge(self, trajectory: Trajectory, **attr):
        source, target = trajectory.at(trajectory.get_start()), trajectory.at(trajectory.get_end())
        self.check_node(node=source)
        attr["transition"] = trajectory
        if target not in self.nodes:
            self.add_node(state=target, gen=self.nodes[source]["gen"] + 1)

        super(TrajectoryGraph, self).add_edge(u_of_edge=source, v_of_edge=target, **attr)
        #self.trajectories[(source, target)] = trajectory


    def get_max_gen(self):
        gens = [node ["gen"] for node in self.nodes]
        return max(gens)

    # todo
    def get_all_trajectories(self) -> Set[Trajectory]:
        assert is_directed_acyclic_graph(self)
        ret = set()

        max_gen = self.get_max_gen()
        start_nodes = [node for node in self.nodes if node["gen"] == 0]
        end_nodes = [node for node in self.nodes if node["gen"] == max_gen]
        all_paths = []
        for (start, end) in product(start_nodes, end_nodes):
            all_paths.append(all_simple_paths(self, start, end))
        return all_paths




        pass

    def get_trajectory(self, source: VehicleState, target: VehicleState) -> Trajectory:
        self.check_node(source)
        self.check_node(target)
        if not has_path(G=self, source=source, target=target):
            raise ValueError(f"No path exists between {source, target}!")

        nodes = shortest_path(G=self, source=source, target=target)
        traj: List[Trajectory] = [] # todo how to fix this
        for node1, node2 in zip(nodes[:-1], nodes[1:]):
            traj += self.get_edge_data(u=node1, v=node2)["transition"]
            #traj.append(self.get_trajectory_edge(source=node1, target=node2))
        #return Trajectory(values=traj, lane=self.lane, goal=self.goal)
        return traj[0]


    def iterate_all_trajectories(self) -> Iterator[Trajectory]:
        all_traj = self.get_all_trajectories()
        # todo
        pass




def get_all_trajectories(self, source: VehicleState) -> Set[Trajectory]:
    if source not in self.nodes:
        raise ValueError(f"Source node ({source}) not in graph!")

    successors = [self.get_trajectory_edge(source=source, target=target) for target in self.successors(source)]
    return frozenset(successors)