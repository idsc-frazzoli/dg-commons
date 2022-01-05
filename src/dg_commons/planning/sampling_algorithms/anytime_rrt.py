import random
from typing import Optional, List, Tuple

import numpy as np
from commonroad.planning.planning_problem import PlanningProblem

from dg_commons import SE2Transform, logger
from dg_commons.planning import PlanningGoal
from dg_commons.planning.sampling_algorithms.node import Node, Tree, AnyNode
from dg_commons.planning.sampling_algorithms.rrt import RRT
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.scenarios import DgScenario


class AnytimeRRT(RRT):
    def __init__(self, scenario: DgScenario, planningProblem: PlanningProblem,
                 initial_vehicle_state: VehicleState, goal: PlanningGoal, goal_state: VehicleState,
                 max_iter: int, goal_sample_rate: int, expand_dis: float, path_resolution: float,
                 search_until_max_iter: bool, seed: int, expand_iter: int):
        super().__init__(scenario=scenario, planningProblem=planningProblem,
                         initial_vehicle_state=initial_vehicle_state, goal=goal, goal_state=goal_state,
                         max_iter=max_iter, goal_sample_rate=goal_sample_rate,
                         expand_dis=expand_dis, path_resolution=path_resolution, seed=seed)

        self.search_until_max_iter = search_until_max_iter
        self.tree = Tree(root_node=AnyNode(pose=SE2Transform(p=np.array([self.state_initial.x, self.state_initial.y]),
                                                             theta=self.state_initial.theta), id='0'))
        self.path = None
        self.expand_iter = expand_iter
        self.number_nodes = 0

    def planning(self) -> Optional[List[SE2Transform]]:
        """
        rrt path planning
        animation: flag for animation on or off
        """
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node_from_tree(self.tree.tree, rnd_node)
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if not self.check_collision(new_node):
                self.number_nodes += 1
                new_node.id = str(self.number_nodes)
                self.tree.set_child(parent=nearest_node, child=new_node)
                self.tree.insert(new_node)

            if self.calc_dist_to_goal(self.tree.last_node()) <= self.expand_dis:
                final_node = self.steer(self.tree.last_node(), self.goal_node,
                                        self.expand_dis)
                if not self.check_collision(final_node):
                    self.number_nodes += 1
                    final_node.id = str(self.number_nodes)
                    self.tree.set_child(parent=self.tree.last_node(), child=final_node)
                    self.tree.insert(final_node)
                    self.path = self.tree.find_best_path(final_node)
                    return self.path.path

        return None  # cannot find path

    def get_random_node(self) -> AnyNode:
        node_id = self.tree.last_id_node() + 1
        if random.randint(0, 100) > self.goal_sample_rate:
            sample_point = self.sample_point()
            rnd = AnyNode(pose=SE2Transform(p=np.array([sample_point.x, sample_point.y]),
                                            theta=random.uniform(0, 2 * np.pi)),
                          id=str(node_id))
        else:  # goal point sampling
            rnd = AnyNode(SE2Transform(p=np.array([self.goal_state.x, self.goal_state.y]),
                                       theta=self.goal_state.theta), id=str(node_id))
        return rnd

    @staticmethod
    def get_nearest_node_from_tree(tree: dict, rnd_node: AnyNode) -> AnyNode:
        nearest_node = tree['0']
        d_min = (nearest_node.pose.p[0] - rnd_node.pose.p[0]) ** 2 + (nearest_node.pose.p[1] - rnd_node.pose.p[1]) ** 2
        for key, node in tree.items():
            d = (node.pose.p[0] - rnd_node.pose.p[0]) ** 2 + (node.pose.p[1] - rnd_node.pose.p[1]) ** 2
            if d < d_min:
                nearest_node = node
                d_min = d

        return nearest_node

    def check_path_valid(self) -> bool:
        if self.path:
            for node in self.path.nodes:
                if self.check_collision(node):
                    self.tree.set_invalid_node(node)
                    self.tree.invalid_childs(node)
                    return False

            return True
        else:
            return False

    def remove_driven_nodes(self, current_pose: SE2Transform) -> None:
        id_list = [self.get_min_dist_point(n, current_pose) for n in self.path.nodes]
        d_list = [d[1] for d in id_list]
        min_d = d_list.index(min(d_list))
        min_idx_node = id_list[min_d][0]
        node = self.path.nodes[min_d]
        self.tree.set_new_root_node(node, current_pose, min_idx_node)

        # self.path = self.tree.find_best_path(self.path.nodes[-1])



    @staticmethod
    def get_min_dist_point(node: AnyNode, pose: SE2Transform) -> Tuple[int, float]:
        path = node.path
        d = [(p.p[0] - pose.p[0]) ** 2 + (p.p[1] - pose.p[1]) ** 2 for p in path]
        minind = d.index(min(d))

        return minind, d[minind]

    def replanning(self, current_pose: SE2Transform):
        self.remove_driven_nodes(current_pose)

        if self.check_path_valid():
            return self.path.path
        else:
            if self.calc_dist_to_goal(self.tree.last_node()) <= self.expand_dis:
                final_node = self.steer(self.tree.last_node(), self.goal_node,
                                        self.expand_dis)
                if not self.check_collision(final_node):
                    self.number_nodes += 1
                    final_node.id = str(self.number_nodes)
                    self.tree.set_child(parent=self.tree.last_node(), child=final_node)
                    self.tree.insert(final_node)
                    self.path = self.tree.find_best_path(final_node)
                    return self.path.path

        return None

    def expand_tree(self):
        for i in range(self.expand_iter):
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node_from_tree(self.tree.tree, rnd_node)
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if not self.check_collision(new_node):
                self.number_nodes += 1
                new_node.id = str(self.number_nodes)
                self.tree.set_child(parent=nearest_node, child=new_node)
                self.tree.insert(new_node)
