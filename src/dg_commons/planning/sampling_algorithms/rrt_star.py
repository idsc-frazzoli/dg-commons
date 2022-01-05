import math
from typing import List, Optional

import numpy as np
from commonroad.planning.planning_problem import PlanningProblem

from dg_commons import SE2Transform
from dg_commons.planning import PlanningGoal
from dg_commons.planning.sampling_algorithms.node import StarNode
from dg_commons.planning.sampling_algorithms.rrt import RRT
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.scenarios import DgScenario


class RRTStar(RRT):
    def __init__(self, scenario: DgScenario, planningProblem: PlanningProblem,
                 initial_vehicle_state: VehicleState, goal: PlanningGoal, goal_state: VehicleState,
                 max_iter: int, goal_sample_rate: int, expand_dis: float, path_resolution: float,
                 connect_circle_dist: float, search_until_max_iter: bool, seed: int):
        super().__init__(scenario=scenario, planningProblem=planningProblem,
                         initial_vehicle_state=initial_vehicle_state, goal=goal, goal_state=goal_state,
                         max_iter=max_iter, goal_sample_rate=goal_sample_rate,
                         expand_dis=expand_dis, path_resolution=path_resolution, seed=seed)

        self.connect_circle_dist = connect_circle_dist
        self.search_until_max_iter = search_until_max_iter

    def planning(self) -> Optional[List[SE2Transform]]:
        self.node_list = [StarNode(SE2Transform(p=np.array([self.state_initial.x, self.state_initial.y]),
                                                theta=self.state_initial.theta))]
        for i in range(self.max_iter):
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd,
                                  self.expand_dis)
            near_node = self.node_list[nearest_ind]
            new_node.cost = near_node.cost + math.hypot(new_node.pose.p[0] - near_node.pose.p[0],
                                                        new_node.pose.p[1] - near_node.pose.p[1])

            if not self.check_collision(new_node):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            if ((not self.search_until_max_iter)
                    and new_node):  # if reaches goal
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node: StarNode, near_inds: List[int]) -> Optional[StarNode]:
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node
            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node, self.expand_dis)
            if t_node and not self.check_collision(t_node):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node, self.expand_dis)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self) -> Optional[int]:
        dist_to_goal_list = [
            self.calc_dist_to_goal(n) for n in self.node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node, self.expand_dis)
            if not self.check_collision(t_node):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node: StarNode):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.pose.p[0] - new_node.pose.p[0]) ** 2 + (node.pose.p[1] - new_node.pose.p[1]) ** 2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        return near_inds

    def rewire(self, new_node: StarNode, near_inds: List[int]) -> None:
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree
                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.
        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node, self.expand_dis)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            collision = self.check_collision(edge_node)
            improved_cost = near_node.cost > edge_node.cost

            if not collision and improved_cost:
                near_node.pose = edge_node.pose
                near_node.cost = edge_node.cost
                near_node.path = edge_node.path
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node: StarNode, to_node: StarNode) -> float:
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node: StarNode) -> None:

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)
