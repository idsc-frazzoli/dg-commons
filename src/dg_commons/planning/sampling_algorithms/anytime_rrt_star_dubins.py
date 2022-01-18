from typing import Optional, List

import numpy as np

from dg_commons import SE2Transform, PlayerName
from dg_commons.planning import PlanningGoal
from dg_commons.planning.sampling_algorithms import dubins_path
from dg_commons.planning.sampling_algorithms.anytime_rrt_star import AnytimeRRTStar
from dg_commons.planning.sampling_algorithms.node import AnyNode
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn
from dg_commons.sim.scenarios import DgScenario


class AnytimeRRTStarDubins(AnytimeRRTStar):
    def __init__(self, player_name: PlayerName, scenario: DgScenario,
                 initial_vehicle_state: VehicleState, goal: PlanningGoal, goal_state: VehicleState,
                 max_iter: int, goal_sample_rate: int, expand_dis: float, path_resolution: float, path_length,
                 search_until_max_iter: bool, seed: int, curvature: float, expand_iter: int,
                 connect_circle_dist: float):
        super().__init__(player_name=player_name, scenario=scenario,
                         initial_vehicle_state=initial_vehicle_state, goal=goal, goal_state=goal_state,
                         max_iter=max_iter, goal_sample_rate=goal_sample_rate,
                         expand_dis=expand_dis, path_resolution=path_resolution, seed=seed,
                         search_until_max_iter=search_until_max_iter, expand_iter=expand_iter,
                         connect_circle_dist=connect_circle_dist)
        self.curvature = curvature
        self.path_length = path_length

    def planning(self) -> Optional[List[SE2Transform]]:
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_node = self.tree.get_nearest_node_from_tree(rnd_node.pose.p)
            new_node = self.constrained_steer(nearest_node, rnd_node)

            if not self.check_collision(new_node):
                near_nodes = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_nodes)
                if node_with_updated_parent:
                    self.number_nodes += 1
                    node_with_updated_parent.id = str(self.number_nodes)
                    self.tree.insert(node_with_updated_parent)
                    self.tree.set_child(parent=node_with_updated_parent.parent, child=node_with_updated_parent)
                    self.rewire(node_with_updated_parent, near_nodes, SE2Transform(p=[self.state_initial.x,
                                                                                      self.state_initial.y],
                                                                                   theta=self.state_initial.theta))
                else:
                    self.number_nodes += 1
                    new_node.id = str(self.number_nodes)
                    self.tree.insert(new_node)
                    self.tree.set_child(parent=nearest_node, child=new_node)

            if (not self.search_until_max_iter) and new_node:  # check reaching the goal
                last_node = self.search_best_goal_node()
                if last_node:
                    self.path = self.tree.find_best_path(last_node)
                    return self.path.path

        print("reached max iteration")

        last_node = self.search_best_goal_node()
        if last_node:
            self.path = self.tree.find_best_path(last_node)
            return self.path.path
        else:
            print("Cannot find path")
        return None  # cannot find path

    def replanning(self, current_pose: SE2Transform):
        self.remove_driven_nodes(current_pose)
        if self.check_path_valid():
            return self.path.path
        else:
            self.tree.remove()
            last_node = self.search_best_goal_node()
            if last_node:
                self.path = self.tree.find_best_path(last_node)
                if self.check_path_valid():
                    return self.path.path
                self.tree.remove()
        self.path = None
        return None

    def reached_goal(self, node: AnyNode):
        x0_p1 = VehicleStateDyn(x=node.pose.p[0], y=node.pose.p[1], theta=node.pose.theta,
                                vx=0.0, delta=0.0)
        if self.goal.is_fulfilled(x0_p1):
            return True
        return False

    def search_best_goal_node(self):
        final_goal_nodes = []
        for node in self.tree.tree.values():
            if self.reached_goal(node):
                final_goal_nodes.append(node)

        if not final_goal_nodes:
            return None

        min_cost = min([node.cost for node in final_goal_nodes])
        for node in final_goal_nodes:
            if node.cost == min_cost:
                return node

    def steer(self, from_node: AnyNode, to_node: AnyNode, extend_length: float):

        path, mode, course_lengths = \
            dubins_path.dubins_path_planning(
                from_node.pose.p[0], from_node.pose.p[1], from_node.pose.theta,
                to_node.pose.p[0], to_node.pose.p[1], to_node.pose.theta, self.curvature,
                step_size=self.path_resolution)

        if len(path) <= 1:  # cannot find a dubins path
            return None

        new_node = AnyNode(pose=SE2Transform(p=np.array([path[-1].p[0], path[-1].p[1]]),
                                             theta=path[-1].theta), id='new')
        new_node.path = path
        new_node.cost += sum([abs(c) for c in course_lengths])
        new_node.parent = from_node

        return new_node

    def constrained_steer(self, from_node: AnyNode, to_node: AnyNode):
        path, mode, course_lengths = dubins_path.dubins_path_planning(
            from_node.pose.p[0], from_node.pose.p[1], from_node.pose.theta,
            to_node.pose.p[0], to_node.pose.p[1], to_node.pose.theta, self.curvature, step_size=self.path_resolution)

        if len(path) <= 1:  # cannot find a dubins path
            return None
        path_idx = int(self.path_length / self.path_resolution)

        if path_idx < len(path) - 1:
            path, mode, course_lengths = dubins_path.dubins_path_planning(
                from_node.pose.p[0], from_node.pose.p[1], from_node.pose.theta,
                path[path_idx].p[0], path[path_idx].p[1], path[path_idx].theta, self.curvature,
                step_size=self.path_resolution)
        new_node = AnyNode(pose=SE2Transform(p=np.array([path[-1].p[0], path[-1].p[1]]),
                                             theta=path[-1].theta), id='new')
        new_node.path = path
        new_node.cost += sum([abs(c) for c in course_lengths])
        new_node.parent = from_node

        return new_node

    def expand_tree(self, current_pose: SE2Transform):
        for i in range(self.expand_iter):
            rnd_node = self.get_random_node()
            nearest_node = self.tree.get_nearest_node_from_tree(rnd_node.pose.p)
            new_node = self.constrained_steer(nearest_node, rnd_node)
            if not self.check_collision(new_node):
                near_nodes = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_nodes)
                if node_with_updated_parent:
                    self.number_nodes += 1
                    node_with_updated_parent.id = str(self.number_nodes)
                    self.tree.insert(node_with_updated_parent)
                    self.tree.set_child(parent=node_with_updated_parent.parent, child=node_with_updated_parent)
                    self.rewire(node_with_updated_parent, near_nodes, current_pose)
                else:
                    self.number_nodes += 1
                    new_node.id = str(self.number_nodes)
                    self.tree.insert(new_node)
                    self.tree.set_child(parent=nearest_node, child=new_node)
