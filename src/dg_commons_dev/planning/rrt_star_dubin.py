import math
import numpy as np
from dg_commons_dev.planning.rrt_star import RRTStar, RRTStarParams
from shapely.geometry import Point, Polygon, LineString
from dg_commons_dev.planning.rrt_utils.utils import Node
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from dg_commons_dev.planning.rrt_utils.sampling import uniform_sampling, RectangularBoundaries
from dg_commons_dev.planning.rrt_utils.steering import dubin_curves
from dg_commons_dev.planning.rrt_utils.nearest_neighbor import naive, dubin_distance_cost
from dg_commons_dev.planning.rrt_utils.sampling import BaseBoundaries
from shapely.geometry.base import BaseGeometry
from dg_commons_dev.planning.rrt_utils.goal_region import GoalRegion


@dataclass
class RRTStarDubinParams(RRTStarParams):
    path_resolution: float = 0.1
    """ Resolution of points on path """
    max_iter: int = 500
    """ Maximal number of iterations """
    goal_sample_rate: float = 10
    """ Rate at which, on average, the goal position is sampled in % """
    sampling_fct: Callable[[BaseBoundaries, GoalRegion, float, Tuple[float, float]], Node] = uniform_sampling
    """ 
    Sampling function: takes sampling boundaries, goal node, goal sampling rate and returns a sampled node
    """
    steering_fct: Callable[[Node, Node, float, float, float], Node] = dubin_curves
    """ 
    Steering function: takes two nodes (start and goal); Maximum distance; Resolution of path; Max curvature
    and returns the new node fulfilling the passed requirements
    """
    distance_meas: Callable[[Node, Node, ...], float] = dubin_distance_cost
    """ 
    Formulation of a distance between two nodes, in general not symmetric: from second node to first
    """
    max_distance: float = 3
    """ 
    Maximum distance between a node and its nearest neighbor wrt distance_meas
    """
    max_distance_to_goal: float = 3
    """ Max distance to goal """
    nearest_neighbor_search: Callable[[Node, List[Node], Callable[[Node, Node], float]], int] = naive
    """ 
    Method for nearest neighbor search. Searches for the nearest neighbor to a node through a list of nodes wrt distance
    function.
    """
    connect_circle_dist: float = 50.0
    """ Radius of near neighbors is proportional to this one """
    max_curvature: float = 0.2
    """ Maximal curvature in Dubin curve """


class RRTStarDubins(RRTStar):
    """
    Class for RRT planning with Dubins path
    """
    REF_PARAMS: dataclass = RRTStarDubinParams

    def __init__(self, params: RRTStarDubinParams = RRTStarDubinParams()):
        """
        Parameters set up
        @param params: RRT Dubin parameters
        """
        super().__init__(params)

    def planning(self, start: Node, goal: GoalRegion, obstacle_list: List[BaseGeometry],
                 sampling_bounds: BaseBoundaries, search_until_max_iter: bool = False,
                 limit_angles: Tuple[float, float] = (-math.pi, math.pi),
                 min_distance: float = 0.1) -> Optional[List[Node]]:
        """
        RRT Dubin planning
        @param start: Starting node
        @param goal: Goal node
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
            rnd = self.sampling_fct(self.boundaries, self.end, self.goal_sample_rate, limit_angles)
            nearest_ind = self.nearest(rnd, self.node_list, self.distance_meas, self.curvature)
            new_node = self.steering_fct(self.node_list[nearest_ind], rnd, self.expand_dis,
                                         self.path_resolution, self.curvature)
            new_node.cost = self.calc_new_cost(self.node_list[nearest_ind], new_node)

            if self.check_collision(new_node, self.obstacle_list):
                near_indexes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indexes)
                if new_node:
                    self.node_list.append(new_node)
                    self.update_nodes_to_end()
                    self.rewire(new_node, near_indexes)

            if (not search_until_max_iter) and self.can_reach_end:
                self.search_best_goal_node()
                return self.generate_final_course()

        print("reached max iteration")

        if self.can_reach_end:
            self.search_best_goal_node()
            return self.generate_final_course()
        else:
            print("Cannot find path")

        return None


def main():
    """ Dummy example """
    obstacle_list = [Polygon(((5, 4), (5, 6), (10, 6), (10, 4), (5, 4))), LineString([Point(0, 8), Point(5, 8)])]

    # Set Initial parameters
    start = Node(0.0, 0.0, np.deg2rad(0.0))
    goal = Node(10.0, 10.0, np.deg2rad(0.0))
    rand_area = RectangularBoundaries((-2, 15, -2, 15))

    rrt = RRTStarDubins()
    rrt.planning(start, GoalRegion(goal, goal.yaw, 1, 1, 0.5), obstacle_list, rand_area, search_until_max_iter=False)
    rrt.plot_results(create_animation=True)


if __name__ == '__main__':
    main()
