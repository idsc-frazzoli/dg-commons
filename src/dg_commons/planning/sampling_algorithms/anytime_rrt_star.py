import math
from typing import Optional, List

from dg_commons import SE2Transform, PlayerName
from dg_commons.planning import PlanningGoal
from dg_commons.planning.sampling_algorithms.anytime_rrt import AnytimeRRT
from dg_commons.planning.sampling_algorithms.node import AnyNode
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.scenarios import DgScenario


class AnytimeRRTStar(AnytimeRRT):
    def __init__(self, player_name: PlayerName, scenario: DgScenario,
                 initial_vehicle_state: VehicleState, goal: PlanningGoal, goal_state: VehicleState,
                 max_iter: int, goal_sample_rate: int, expand_dis: float, path_resolution: float,
                 search_until_max_iter: bool, seed: int, expand_iter: int, connect_circle_dist: float):
        super().__init__(player_name=player_name, scenario=scenario,
                         initial_vehicle_state=initial_vehicle_state, goal=goal, goal_state=goal_state,
                         max_iter=max_iter, goal_sample_rate=goal_sample_rate,
                         expand_dis=expand_dis, path_resolution=path_resolution, seed=seed,
                         search_until_max_iter=search_until_max_iter, expand_iter=expand_iter)
        self.connect_circle_dist = connect_circle_dist

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
            new_node.cost = nearest_node.cost + math.hypot(new_node.pose.p[0] - nearest_node.pose.p[0],
                                                           new_node.pose.p[1] - nearest_node.pose.p[1])

            if not self.check_collision(new_node):
                near_nodes = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_nodes)
                if node_with_updated_parent:
                    self.number_nodes += 1
                    node_with_updated_parent.id = str(self.number_nodes)
                    self.tree.insert(node_with_updated_parent)
                    self.tree.set_child(parent=node_with_updated_parent.parent, child=node_with_updated_parent)
                    self.rewire(node_with_updated_parent, near_nodes, self.state_initial.as_ndarray()[:-1])
                else:
                    self.number_nodes += 1
                    new_node.id = str(self.number_nodes)
                    self.tree.insert(new_node)
                    self.tree.set_child(parent=nearest_node, child=new_node)

            if self.calc_dist_to_goal(self.tree.last_node()) <= self.expand_dis:
                final_node = self.steer(self.tree.last_node(), self.goal_node,
                                        self.expand_dis)
                if not self.check_collision(final_node):
                    self.number_nodes += 1
                    final_node.id = str(self.number_nodes)
                    self.tree.insert(final_node)
                    self.tree.set_child(parent=self.tree.last_node(), child=final_node)
                    self.path = self.tree.find_best_path(final_node)
                    return self.path.path

        return None  # cannot find path

    def find_near_nodes(self, new_node: AnyNode) -> List[AnyNode]:
        nnode = len(self.tree.tree) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        near_nodes = [node for node in self.tree.tree.values() if
                      (node.pose.p[0] - new_node.pose.p[0]) ** 2 + (node.pose.p[1] - new_node.pose.p[1]) ** 2 <= r ** 2]
        return near_nodes

    def choose_parent(self, new_node: AnyNode, near_nodes: List[AnyNode]) -> Optional[AnyNode]:
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
        if not near_nodes:
            return None

        # search nearest cost in near_inds
        costs = []
        for near_node in near_nodes:
            t_node = self.steer(near_node, new_node, self.expand_dis)
            if t_node and not self.check_collision(t_node):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            return None

        min_near_node = near_nodes[costs.index(min_cost)]
        new_node = self.steer(min_near_node, new_node, self.expand_dis)
        new_node.cost = min_cost

        return new_node

    def calc_new_cost(self, from_node: AnyNode, to_node: AnyNode) -> float:
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def rewire(self, new_node: AnyNode, near_nodes: List[AnyNode], current_pose: SE2Transform) -> None:
        for near_node in near_nodes:
            if near_node.id == "0":
                continue
            if near_node.id == new_node.parent.id:
                continue
            near_node_children_id = [n.id for n in near_node.children.copy()]
            if new_node.id in near_node_children_id:
                continue

            edge_node = self.steer(new_node, near_node, self.expand_dis)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            collision = self.check_collision(edge_node)
            improved_cost = near_node.cost > edge_node.cost

            if not collision and improved_cost:
                self.tree.rewire(near_node.id, edge_node)

    def expand_tree(self, current_pose: SE2Transform):
        for i in range(self.expand_iter):
            rnd_node = self.get_random_node()
            # nearest_node = self.get_nearest_node_from_tree(self.tree.tree, rnd_node)
            nearest_node = self.tree.get_nearest_node_from_tree(rnd_node.pose.p.tolist())
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            new_node.cost = nearest_node.cost + math.hypot(new_node.pose.p[0] - nearest_node.pose.p[0],
                                                           new_node.pose.p[1] - nearest_node.pose.p[1])

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
