import copy

import numpy as np

from dg_commons import SE2Transform, PlayerName
from dg_commons.planning import PlanningGoal
from dg_commons.planning.sampling_algorithms import dubins_path
from dg_commons.planning.sampling_algorithms.node import StarNode
from dg_commons.planning.sampling_algorithms.rrt import RRT
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn
from dg_commons.sim.scenarios import DgScenario


class RRTDubins(RRT):
    def __init__(self, player_name: PlayerName, scenario: DgScenario,
                 initial_vehicle_state: VehicleState, goal: PlanningGoal, goal_state: VehicleState,
                 max_iter: int, goal_sample_rate: int, expand_dis: float, path_resolution: float, path_length: float,
                 curvature: float, goal_yaw_th: float, goal_xy_th: float, search_until_max_iter: bool, seed: int):
        super().__init__(player_name=player_name, scenario=scenario,
                         initial_vehicle_state=initial_vehicle_state, goal=goal, goal_state=goal_state,
                         max_iter=max_iter, goal_sample_rate=goal_sample_rate,
                         expand_dis=expand_dis, path_resolution=path_resolution, seed=seed)

        self.curvature = curvature
        self.goal_yaw_th = np.deg2rad(goal_yaw_th)
        self.goal_xy_th = goal_xy_th
        self.search_until_max_iter = search_until_max_iter
        self.path_length = path_length

    def planning(self):
        self.node_list = [StarNode(SE2Transform(p=np.array([self.state_initial.x, self.state_initial.y]),
                                                theta=self.state_initial.theta))]
        for i in range(self.max_iter):
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.constrained_steer(self.node_list[nearest_ind], rnd, self.expand_dis)

            if not self.check_collision(new_node):
                self.node_list.append(new_node)

            if (not self.search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)
        else:
            print("Cannot find path")

        return None

    def steer(self, from_node: StarNode, to_node: StarNode, extend_length: float):

        path, mode, course_lengths = \
            dubins_path.dubins_path_planning(
                from_node.pose.p[0], from_node.pose.p[1], from_node.pose.theta,
                to_node.pose.p[0], to_node.pose.p[1], to_node.pose.theta, self.curvature, step_size=self.path_resolution)

        if len(path) <= 1:  # cannot find a dubins path
            return None

        new_node = copy.deepcopy(from_node)
        new_node.pose = SE2Transform(p=np.array([path[-1].p[0], path[-1].p[1]]),
                                     theta=path[-1].theta)

        new_node.path = path
        new_node.cost += sum([abs(c) for c in course_lengths])
        new_node.parent = from_node

        return new_node

    def constrained_steer(self, from_node: StarNode, to_node: StarNode, extend_length: float):
        path, mode, course_lengths = dubins_path.dubins_path_planning(
            from_node.pose.p[0], from_node.pose.p[1], from_node.pose.theta,
            to_node.pose.p[0], to_node.pose.p[1], to_node.pose.theta, self.curvature, step_size=self.path_resolution)

        if len(path) <= 1:  # cannot find a dubins path
            return None
        path_idx = int(self.path_length/self.path_resolution)

        new_node = copy.deepcopy(from_node)
        if path_idx < len(path) - 1:
            path, mode, course_lengths = dubins_path.dubins_path_planning(
                from_node.pose.p[0], from_node.pose.p[1], from_node.pose.theta,
                path[path_idx].p[0], path[path_idx].p[1], path[path_idx].theta, self.curvature,
                step_size=self.path_resolution)
        new_node.pose = SE2Transform(p=np.array([path[-1].p[0], path[-1].p[1]]),
                                     theta=path[-1].theta)
        new_node.path = path
        new_node.cost += sum([abs(c) for c in course_lengths])
        new_node.parent = from_node

        return new_node

    def reached_goal(self, node: StarNode):
        x0_p1 = VehicleStateDyn(x=node.pose.p[0], y=node.pose.p[1], theta=node.pose.theta,
                                vx=0.0, delta=0.0)
        if self.goal.is_fulfilled(x0_p1):
            return True
        return False

    def search_best_goal_node(self):

        # goal_indexes = []
        # for (i, node) in enumerate(self.node_list):
        #     if self.calc_dist_to_goal(node) <= self.goal_xy_th:
        #         goal_indexes.append(i)
        #
        # # angle check
        # final_goal_indexes = []
        # for i in goal_indexes:
        #     if abs(self.node_list[i].pose.theta - self.goal_node.pose.theta) <= self.goal_yaw_th:
        #         final_goal_indexes.append(i)
        final_goal_indexes = []
        for (i, node) in enumerate(self.node_list):
            if self.reached_goal(node):
                final_goal_indexes.append(i)

        if not final_goal_indexes:
            return None

        min_cost = min([self.node_list[i].cost for i in final_goal_indexes])
        for i in final_goal_indexes:
            if self.node_list[i].cost == min_cost:
                return i

        return None
