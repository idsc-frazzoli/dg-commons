import math
from dg_commons_dev.planning.rrt import RRT, RRTParams
from shapely.geometry import Point, Polygon, LineString
from dg_commons_dev.planning.rrt_utils.utils import Node
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from dg_commons_dev.planning.rrt_utils.sampling import uniform_sampling, BaseBoundaries, RectangularBoundaries
from dg_commons_dev.planning.rrt_utils.steering import straight_to
from dg_commons_dev.planning.rrt_utils.nearest_neighbor import distance_cost, naive
from shapely.geometry.base import BaseGeometry
from dg_commons_dev.planning.rrt_utils.goal_region import GoalRegion


@dataclass
class RRTStarParams(RRTParams):
    path_resolution: float = 0.5
    """ Resolution of points on path """
    max_iter: int = 1500
    """ Maximal number of iterations """
    goal_sample_rate: float = 5
    """ Rate at which, on average, the goal position is sampled in % """
    sampling_fct: Callable[[BaseBoundaries, GoalRegion, float, Tuple[float, float]], Node] = uniform_sampling
    """ 
    Sampling function: takes sampling boundaries, goal node, goal sampling rate and returns a sampled node
    """
    steering_fct: Callable[[Node, Node, float, float, float], Node] = straight_to
    """ 
    Steering function: takes two nodes (start and goal); Maximum distance; Resolution of path; Max curvature
    and returns the new node fulfilling the passed requirements
    """
    distance_meas: Callable[[Node, Node, float], float] = distance_cost
    """ 
    Formulation of a distance between two nodes, in general not symmetric: from first node to second
    """
    max_distance: float = 3
    """ 
    Maximum distance between a node and its nearest neighbor wrt distance_meas
    """
    max_distance_to_goal: float = 3
    """ Max distance to goal """
    nearest_neighbor_search: Callable[[Node, List[Node], Callable[[Node, Node, float], float], float], int] = naive
    """ 
    Method for nearest neighbor search. Searches for the nearest neighbor to a node through a list of nodes wrt distance
    function.
    """
    connect_circle_dist: float = 5.0
    """ Radius of near neighbors is proportional to this one, only used in RRT star """
    max_curvature: float = 0.2
    """ Maximal curvature in Dubin curve, only used with Dubin curves """


class RRTStar(RRT):
    """
    Class for RRT Star planning
    """
    REF_PARAMS: dataclass = RRTStarParams

    def __init__(self, params: RRTStarParams = RRTStarParams()):
        """
        Parameters set up
        @param params: RRT star parameters
        """
        super().__init__(params)

    def planning(self, start: Node, goal: GoalRegion, obstacle_list: List[BaseGeometry],
                 sampling_bounds: BaseBoundaries, search_until_max_iter: bool = False,
                 limit_angles: Tuple[float, float] = (-math.pi, math.pi),
                 min_distance: float = 0.1) -> Optional[List[Node]]:
        """
        RRT Star planning
        @param start: Starting node
        @param goal: Goal region
        @param obstacle_list: List of shapely objects representing obstacles
        @param sampling_bounds: Boundaries in the samples space
        @param search_until_max_iter: flag for whether to search until max_iter
        @param limit_angles: Sampling space for angle
        @param min_distance: minimal distance to keep from obstacles
        @return: sequence of nodes corresponding to path found or None if no path was found
        """
        self.start = start
        self.end = goal
        self.boundaries = sampling_bounds
        self.obstacle_list = obstacle_list
        self.can_reach_end = []
        self.node_list = [self.start]
        self.dist_from_obstacles = min_distance

        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd_node = self.sampling_fct(self.boundaries, self.end, self.goal_sample_rate, limit_angles)
            nearest_ind = self.nearest(rnd_node, self.node_list, self.distance_meas, self.curvature)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steering_fct(nearest_node, rnd_node, self.expand_dis, self.path_resolution, self.curvature)
            new_node.cost = self.calc_new_cost(nearest_node, new_node)

            if self.check_collision(new_node, self.obstacle_list):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)
                self.update_nodes_to_end()

                if (not search_until_max_iter) and self.can_reach_end:
                    self.search_best_goal_node()
                    return self.generate_final_course()

        print("reached max iteration")

        if self.can_reach_end:
            self.search_best_goal_node()
            return self.generate_final_course()

        return None

    def choose_parent(self, new_node: Node, near_inds: List[int]) -> Optional[Node]:
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
        @param new_node: Randomly generated node with a path from its neared point.
        @param near_inds: Indices of the nodes that are near new_node
        @return: New node with the chosen parent. None if it was not possible to compute
        """
        if not near_inds:
            return None

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steering_fct(near_node, new_node, self.expand_dis, self.path_resolution, self.curvature)
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steering_fct(self.node_list[min_ind], new_node, self.expand_dis,
                                     self.path_resolution, self.curvature)
        new_node.cost = min_cost

        return new_node

    def find_near_nodes(self, new_node: Node) -> List[int]:
        """
        Defines a circle centered on new_node
        Returns all nodes of the three that are inside this ball
        @param new_node: New randomly generated node, without collisions between its nearest node
        @return: List with the indices of the nodes inside the ball of radius r
        """
        n_node = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(n_node) / n_node))
        r = min(r, self.expand_dis)

        dist_list = [self.distance_meas(node, new_node, self.curvature) for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node: Node, near_inds: List[int]) -> None:
        """
        For each node in near_inds, this will check if it is cheaper to
        arrive to them from new_node.
        In such a case, this will re-assign the parent of the nodes in
        near_inds to new_node.
        @param new_node: Node randomly added which can be joined to the tree
        @param near_inds: A list of indices of the self.new_node which contains nodes within a circle of a given radius
        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steering_fct(new_node, near_node, self.expand_dis, self.path_resolution, self.curvature)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.path_x = edge_node.path_x
                near_node.path_y = edge_node.path_y
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node: Node, to_node: Node) -> float:
        """
        Distance from from_node to to_node + distance from start to from_node
        @param from_node: starting node
        @param to_node: target node
        @return: distance
        """
        d = self.distance_meas(to_node, from_node, self.curvature)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node: Node) -> None:
        """
        Update all costs affected by the structure change
        @param parent_node: Node, whose cost has changed
        """
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


def main(gx=6.0, gy=10.0):
    """ Dummy example """

    obstacle_list = [Polygon(((5, 4), (5, 6), (10, 6), (10, 4), (5, 4))), LineString([Point(0, 8), Point(5, 8)])]
    bounds: BaseBoundaries = RectangularBoundaries((-2, 15, -2, 15))
    rrt = RRTStar()
    rrt.planning(start=Node(0, 0), goal=GoalRegion(Node(gx, gy), 0, 0.5, 0.5, 0.5),
                 sampling_bounds=bounds, obstacle_list=obstacle_list, search_until_max_iter=False)
    rrt.plot_results(create_animation=True)


if __name__ == '__main__':
    main()
