from typing import List, Set

from dg_commons.planning import Trajectory


class Node:
    """
    Class for nodes used in search-based motion planners. The class was adapted to dg-commons
    from the original code from CommonRoad (Cyber-Physical Systems Group, Technical University of Munich):
    https://gitlab.lrz.de/tum-cps/commonroad-search/-/blob/master/SMP/motion_planner/node.py
    """

    def __init__(self, list_trajectories: List[Trajectory],
                 set_primitives: Set[Trajectory], depth_tree: int):
        """
        Initialization of class Node.
        """
        # list of paths of motion primitives
        self.list_paths = [traj.as_path() for traj in list_trajectories]
        # list of trajectories of motion primitives
        self.list_trajectories = list_trajectories
        # list of motion primitives
        self.set_primitives = set_primitives
        # depth of the node
        self.depth_tree = depth_tree

    def get_successors(self) -> Set[Trajectory]:
        """
        Returns all possible successor primitives of the current primitive (node).
        """
        return self.set_primitives


class PriorityNode(Node):
    """
    Class for nodes with priorities used in the motion planners. The class was adapted to dg-commons
    from the original code from CommonRoad (Cyber-Physical Systems Group, Technical University of Munich):
    https://gitlab.lrz.de/tum-cps/commonroad-search/-/blob/master/SMP/motion_planner/node.py
    """

    def __init__(self, list_trajectories: List[Trajectory],
                 set_primitives: Set[Trajectory], depth_tree: int,
                 priority: float):
        """
        Initialization of class PriorityNode.
        """
        super().__init__(list_trajectories, set_primitives, depth_tree)
        # priority/cost of the node
        self.priority = priority
