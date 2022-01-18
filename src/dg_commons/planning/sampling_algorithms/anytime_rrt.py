import random
from decimal import Decimal
from typing import Optional, List, Tuple

import numpy as np

from dg_commons import SE2Transform, PlayerName
from dg_commons.planning import PlanningGoal
from dg_commons.planning.sampling_algorithms.node import Node, Tree, AnyNode
from dg_commons.planning.sampling_algorithms.rrt import RRT
from dg_commons.sim import SimObservations
from dg_commons.sim.models import Pacejka4p
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.scenarios import DgScenario


class AnytimeRRT(RRT):
    def __init__(self, player_name: PlayerName, scenario: DgScenario,
                 initial_vehicle_state: VehicleState, goal: PlanningGoal, goal_state: VehicleState,
                 max_iter: int, goal_sample_rate: int, expand_dis: float, path_resolution: float,
                 search_until_max_iter: bool, seed: int, expand_iter: int):
        super().__init__(player_name=player_name, scenario=scenario,
                         initial_vehicle_state=initial_vehicle_state, goal=goal, goal_state=goal_state,
                         max_iter=max_iter, goal_sample_rate=goal_sample_rate,
                         expand_dis=expand_dis, path_resolution=path_resolution, seed=seed)

        self.search_until_max_iter = search_until_max_iter
        self.tree = Tree(root_node=AnyNode(pose=SE2Transform(p=np.array([self.state_initial.x, self.state_initial.y]),
                                                             theta=self.state_initial.theta), id='0'))
        self.path = None
        self.expand_iter = expand_iter
        self.number_nodes = 0
        self.sim_observation: SimObservations = SimObservations(players={}, time=Decimal(0.0))

    def planning(self) -> Optional[List[SE2Transform]]:
        """
        rrt path planning
        animation: flag for animation on or off
        """
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            # nearest_node = self.get_nearest_node_from_tree(self.tree.tree, rnd_node)
            nearest_node = self.tree.get_nearest_node_from_tree(rnd_node.pose.p.tolist())
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
                if node.id in self.tree.tree:
                    if self.check_collision(node):
                        self.tree.set_invalid_node(node)
                        self.tree.invalid_childs(node)
                        return False

            return True
        else:
            return False

    def remove_driven_nodes(self, current_pose: SE2Transform) -> None:
        if self.path is not None:
            id_list = [self.get_min_dist_point(n, current_pose) for n in self.path.nodes]
            d_list = [d[1] for d in id_list]
            min_d = d_list.index(min(d_list))
            min_idx_node = id_list[min_d][0]
            node = self.path.nodes[min_d]
            if node.id != "0":
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
            self.tree.remove()
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

    def expand_tree(self, current_pose: SE2Transform):
        for i in range(self.expand_iter):
            rnd_node = self.get_random_node()
            nearest_node = self.tree.get_nearest_node_from_tree(rnd_node.pose.p.tolist())
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if not self.check_collision(new_node):
                self.number_nodes += 1
                new_node.id = str(self.number_nodes)
                self.tree.set_child(parent=nearest_node, child=new_node)
                self.tree.insert(new_node)

    def check_collision(self, node: Node) -> bool:
        """Check collisions of the planned trajectory with the environment
        :param trajectory: The planned trajectory
        :return: True if at least one collision happened, False otherwise"""
        if node is None:
            return True
        env_obstacles = self.sim_context.dg_scenario.strtree_obstacles
        dynamic_obstacles = [obs_val for obs_key, obs_val in self.sim_observation.players.items() if
                             obs_key != self.player_name]
        collision = False
        for pose in node.path:
            x0_p1 = VehicleStateDyn(x=pose.p[0], y=pose.p[1], theta=pose.theta,
                                    vx=0.0, delta=0.0)
            # p_model = VehicleModelDyn.default_car(x0_p1)
            # footprint = p_model.get_footprint()
            vm = VehicleModelDyn(x0=x0_p1, vg=VehicleGeometry.default_car(w_half=0.9+0.1, lf=1.7+0.1, lr=1.7+0.1),
                                 vp=VehicleParameters.default_car(),
                                 pacejka_front=Pacejka4p.default_car_front(),
                                 pacejka_rear=Pacejka4p.default_car_rear(),)
            footprint = vm.get_footprint()
            # f_bounds = footprint.bounds
            # delta_increase = 0.05
            # p_shape = Polygon(((f_bounds[0] - delta_increase, f_bounds[1] - delta_increase),
            #                    (f_bounds[2] + delta_increase, f_bounds[1] - delta_increase),
            #                    (f_bounds[2] + delta_increase, f_bounds[3] + delta_increase),
            #                    (f_bounds[0] - delta_increase, f_bounds[3] + delta_increase)))
            p_shape = footprint
            assert p_shape.is_valid
            items = env_obstacles.query_items(p_shape)
            for idx in items:
                candidate = self.sim_context.dg_scenario.static_obstacles[idx]
                if p_shape.intersects(candidate.shape):
                    collision = True
            for do in dynamic_obstacles:
                if do == self.player_name:
                    continue
                do_shape = do.occupancy
                if do_shape.intersects(p_shape):
                    collision = True
        return collision
