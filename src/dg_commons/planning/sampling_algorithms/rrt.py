import math
import random
from abc import ABC
from copy import deepcopy
from typing import List, Tuple, Optional

import numpy as np
from commonroad.planning.planning_problem import PlanningProblem

from dg_commons import SE2Transform, PlayerName
from dg_commons.planning import PlanningGoal
from dg_commons.planning.sampling_algorithms.base_class import SamplingBaseClass
from dg_commons.planning.sampling_algorithms.node import Node
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.scenarios import DgScenario


class RRT(SamplingBaseClass, ABC):
    """
        Class for RRT planning
        """

    def __init__(self, player_name: PlayerName, scenario: DgScenario, planningProblem: PlanningProblem,
                 initial_vehicle_state: VehicleState, goal: PlanningGoal, goal_state: VehicleState,
                 max_iter: int, goal_sample_rate: int, expand_dis: float, path_resolution: float, seed: int):
        super().__init__(player_name=player_name, scenario=scenario, planningProblem=planningProblem,
                         initial_vehicle_state=initial_vehicle_state, goal=goal, goal_state=goal_state, seed=seed)
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_node = Node(SE2Transform(p=np.array([self.goal_state.x, self.goal_state.y]),
                                            theta=self.goal_state.theta))

    def planning(self) -> Optional[List[SE2Transform]]:
        """
        rrt path planning
        animation: flag for animation on or off
        """

        self.node_list = [Node(SE2Transform(p=np.array([self.state_initial.x, self.state_initial.y]),
                                            theta=self.state_initial.theta))]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if not self.check_collision(new_node):
                self.node_list.append(new_node)

            if self.calc_dist_to_goal(self.node_list[-1]) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.goal_node,
                                        self.expand_dis)
                if not self.check_collision(final_node):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # cannot find path

    def steer(self, from_node: Node, to_node: Node, extend_length: float):
        new_node = deepcopy(from_node)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        new_node.path.append(new_node.pose)

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node_x = new_node.pose.p[0]
            new_node_y = new_node.pose.p[1]
            new_node_x += self.path_resolution * math.cos(theta)
            new_node_y += self.path_resolution * math.sin(theta)
            new_node.pose = SE2Transform(p=np.array([new_node_x, new_node_y]), theta=new_node.pose.theta)
            new_node.path.append(new_node.pose)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path.append(to_node.pose)
            new_node.pose = to_node.pose

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind: int) -> List[SE2Transform]:
        path = [self.goal_node.pose]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            for p in reversed(node.path):
                path.append(p)
            node = node.parent
        path.append(node.pose)

        return list(reversed(path))

    def calc_dist_to_goal(self, final_node: Node):
        dx = final_node.pose.p[0] - self.goal_state.x
        dy = final_node.pose.p[1] - self.goal_state.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            sample_point = self.sample_point()
            rnd = Node(SE2Transform(p=np.array([sample_point.x, sample_point.y]), theta=random.uniform(0, 2*np.pi)))
        else:  # goal point sampling
            rnd = self.goal_node
        return rnd

    @staticmethod
    def get_nearest_node_index(node_list: List[Node], rnd_node: Node) -> int:
        dlist = [(node.pose.p[0] - rnd_node.pose.p[0]) ** 2 + (node.pose.p[1] - rnd_node.pose.p[1]) ** 2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def calc_distance_and_angle(from_node: Node, to_node: Node) -> Tuple[float, float]:
        dp = to_node.pose.p - from_node.pose.p
        d = math.hypot(dp[0], dp[1])
        theta = math.atan2(dp[1], dp[0])
        return d, theta

    def check_collision(self, node: Node) -> bool:
        """Check collisions of the planned trajectory with the environment
        :param trajectory: The planned trajectory
        :return: True if at least one collision happened, False otherwise"""
        if node is None:
            return True
        env_obstacles = self.sim_context.dg_scenario.strtree_obstacles
        collision = False
        for pose in node.path:
            x0_p1 = VehicleStateDyn(x=pose.p[0], y=pose.p[1], theta=pose.theta,
                                    vx=0.0, delta=0.0)
            p_model = VehicleModelDyn.default_car(x0_p1)
            footprint = p_model.get_footprint()
            assert footprint.is_valid
            p_shape = footprint
            items = env_obstacles.query_items(p_shape)
            for idx in items:
                candidate = self.sim_context.dg_scenario.static_obstacles[idx]
                if p_shape.intersects(candidate.shape):
                    collision = True
        return collision
