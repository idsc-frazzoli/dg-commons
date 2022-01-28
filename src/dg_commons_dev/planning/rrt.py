import math
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from typing import List, Callable, Optional, Tuple, Dict
from shapely.geometry.base import BaseGeometry
from dataclasses import dataclass
from dg_commons_dev.planning.rrt_utils.sampling import uniform_sampling
from dg_commons_dev.planning.rrt_utils.steering import straight_to
from dg_commons_dev.planning.rrt_utils.nearest_neighbor import distance_cost, naive
from dg_commons_dev.planning.rrt_utils.sampling import RectangularBoundaries, BaseBoundaries
from dg_commons_dev.planning.rrt_utils.utils import Node, move_vehicle
from dg_commons.sim.models.vehicle import VehicleGeometry
from dg_commons_dev.planning.planner_base import Planner
from dg_commons_dev.utils import BaseParams
from dg_commons_dev.planning.rrt_utils.dubins_path_planning import plot_arrow
from dg_commons_dev.planning.rrt_utils.goal_region import GoalRegion


@dataclass
class RRTParams(BaseParams):
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
    vehicle_geom: VehicleGeometry = VehicleGeometry.default_car()
    """ vehicle geometry """
    connect_circle_dist: float = 50.0
    """ Radius of near neighbors is proportional to this one, only used in RRT star """
    max_curvature: float = 0.2
    """ Maximal curvature in Dubin curve, only used with Dubin curves """

    def __post_init__(self):
        assert 0 < self.path_resolution
        assert 0 < self.max_iter
        assert 0 <= self.goal_sample_rate <= 100
        assert 0 < self.max_distance
        assert 0 < self.max_distance_to_goal
        assert 0 <= self.connect_circle_dist
        assert 0 <= self.max_curvature


class RRT(Planner):
    """
    Class for RRT planning
    """
    REF_PARAMS: dataclass = RRTParams

    def __init__(self, params: RRTParams = RRTParams()):
        """
        Parameters set up
        @param params: RRT parameters
        """
        self.start: Node = Node(0, 0)
        """ Start node """
        self.end: GoalRegion = GoalRegion(Node(0, 0, 0), 0, 0, 0)
        """ Goal node """
        self.node_list: List[Node] = []
        """ List of all reachable sampled nodes """
        self.can_reach_end: List[int] = []
        """ List of indices of nodes that can reach the goal node without collisions """
        self.path: List[Node] = []
        """ List of nodes connecting start and goal """
        self.obstacle_list: List[BaseGeometry] = []
        """ List of obstacles present in the scene """
        self.boundaries = None
        self.dist_from_obstacles = 0.1
        """ Minimal distance to keep from obstacles"""

        self.expand_dis: float = params.max_distance
        self.max_dist_to_goal: float = params.max_distance_to_goal
        self.path_resolution: float = params.path_resolution
        self.goal_sample_rate: float = params.goal_sample_rate
        self.max_iter: int = params.max_iter

        self.steering_fct: Callable[[Node, Node, float, float, float], Node] = params.steering_fct
        self.sampling_fct: Callable[[BaseBoundaries, GoalRegion, float, Tuple[float, float]], Node] = params.sampling_fct
        self.distance_meas: Callable[[Node, Node], float] = params.distance_meas
        self.nearest: Callable[[Node, List[Node], Callable[[Node, Node], float]], int] = params.nearest_neighbor_search

        self.vg: VehicleGeometry = params.vehicle_geom
        self.curvature = params.max_curvature
        self.connect_circle_dist = params.connect_circle_dist

    def planning(self, start: Node, goal: GoalRegion, obstacle_list: List[BaseGeometry],
                 sampling_bounds: BaseBoundaries, search_until_max_iter: bool = False,
                 limit_angles: Tuple[float, float] = (-math.pi, math.pi),
                 min_distance: float = 0.1) -> Optional[List[Node]]:
        """
        RRT planning
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
        self.node_list = []
        self.can_reach_end = []
        self.obstacle_list = obstacle_list
        self.node_list = [self.start]
        self.boundaries = sampling_bounds
        self.dist_from_obstacles = min_distance

        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd_node = self.sampling_fct(sampling_bounds, self.end, self.goal_sample_rate, limit_angles)
            nearest_ind = self.nearest(rnd_node, self.node_list, self.distance_meas, self.curvature)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steering_fct(nearest_node, rnd_node, self.expand_dis, self.path_resolution, self.curvature)
            new_node.cost = self.calc_new_cost(nearest_node, new_node)

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)
                self.update_nodes_to_end()

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

    def calc_new_cost(self, from_node: Node, to_node: Node) -> float:
        """
        Compute new cost to go of to_node considering previous node plus edge cost
        @param from_node: Nearest node
        @param to_node: Final node
        @return: Cost
        """
        cost = self.distance_meas(to_node, from_node, self.curvature)

        return from_node.cost + cost

    def generate_final_course(self) -> List[Node]:
        """
        Generate list of nodes composing the path
        @return: The generated list
        """
        path: List[Node] = []

        node = self.end.goal_node
        while node.parent:
            for (ix, iy) in zip(reversed(node.path_x), reversed(node.path_y)):
                temp_node: Node = Node(ix, iy)
                path.append(temp_node)
            node = node.parent
        path.append(self.start)
        path.reverse()

        for i in range(len(path) - 1):
            x, y = path[i].x, path[i].y
            x_next, y_next = path[i + 1].x, path[i + 1].y
            path[i].yaw = math.atan2(y_next - y, x_next - x)

        path[len(path) - 1].yaw = path[len(path) - 2].yaw
        self.path = path

        return path

    def check_collision(self, node: Node, obstacle_list: List[BaseGeometry]) -> bool:
        """
        Check if the connection between the passed node and its parent causes a collision
        @param node: the node considered
        @param obstacle_list: list of shapely obstacles
        @return: whether it is causing a collision
        """
        if node is None:
            return False
        return_val: bool = True

        assert len(node.path_x) == len(node.path_y)
        n = len(node.path_x)
        positions = []
        for i in range(n):
            positions.append((node.path_x[i], node.path_y[i]))
        car_path = LineString(positions).buffer(self.vg.width/2 + self.dist_from_obstacles)
        for obs in obstacle_list:
            if car_path.intersects(obs):
                return_val = False
                break

        return return_val

    def update_nodes_to_end(self) -> None:
        """
        Update list of indices of nodes that can reach the goal based on last node added to the list of nodes.
        """
        new_node = self.node_list[-1]
        idx = len(self.node_list) - 1

        if self.end.inside(new_node):
            self.can_reach_end.append(idx)
        else:
            to_end = self.steering_fct(new_node, self.end.goal_node,
                                       self.expand_dis, self.path_resolution, self.curvature)
            if to_end:
                end_reached = self.end.inside(to_end)
                no_collision = self.check_collision(to_end, self.obstacle_list)
                if end_reached and no_collision:
                    self.can_reach_end.append(idx)

    def search_best_goal_node(self) -> None:
        """
        Return index of the best-last node of the path to the goal. From that, the whole path can be constructed.
        @return: The index
        """
        temp_nodes = []
        for idx in self.can_reach_end:
            node = self.node_list[idx]
            if self.end.inside(node):
                temp = node
            else:
                temp = self.steering_fct(node, self.end.goal_node, self.expand_dis, self.path_resolution,
                                         self.curvature)
            temp_nodes.append(temp)

        min_cost = min([end.cost for end in temp_nodes])
        for end in temp_nodes:
            if end.cost == min_cost:
                self.end.goal_node = end
                return

    def get_length(self) -> float:
        """
        @return: Returns the enlarged length of the vehicle
        """
        return self.vg.length

    def get_width(self) -> float:
        """
        @return: Returns the enlarged width of the vehicle
        """
        return self.vg.width + 2 * self.dist_from_obstacles

    def plot_results(self, create_animation: bool = False) -> None:
        """
        Tool to plot and save the results
        """
        assert len(self.path) != 0

        plt.clf()
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for ob in self.obstacle_list:
            if isinstance(ob, Polygon):
                x, y = ob.exterior.xy
                plt.plot(x, y)
            if isinstance(ob, LineString):
                plt.plot(*ob.xy)

        plt.plot(self.start.x, self.start.y, "xr")
        x, y = self.end.polygon.exterior.xy
        plt.plot(x, y, 'b')
        angle_polygon = self.end.angle_polygon()
        if angle_polygon:
            x, y = angle_polygon.exterior.xy
            plt.plot(x, y, 'y')
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)

        plt.plot([node.x for node in self.path], [node.y for node in self.path], '-r')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')

        enlargement_f: float = 0.1
        min_enlargement: float = 10
        x_lim = self.boundaries.x_bounds()
        delta = enlargement_f * (x_lim[1] - x_lim[0]) + min_enlargement
        x_lim = (x_lim[0] - delta, x_lim[1] + delta)
        plt.xlim(x_lim)

        y_lim = self.boundaries.y_bounds()
        delta = enlargement_f * (y_lim[1] - y_lim[0]) + min_enlargement
        y_lim = (y_lim[0] - delta, y_lim[1] + delta)
        plt.ylim(y_lim)

        if self.start.is_yaw_considered and self.end.goal_node.is_yaw_considered:
            plot_arrow(self.start.x, self.start.y, self.start.yaw)
            plot_arrow(self.end.goal_node.x, self.end.goal_node.y, self.end.goal_node.yaw)
        plt.savefig("test")

        if create_animation:

            from matplotlib.animation import FuncAnimation
            plt.style.use('seaborn-pastel')

            fig = plt.figure()
            ax = plt.axes(xlim=x_lim, ylim=y_lim)
            plt.gca().set_aspect('equal', adjustable='box')
            lines = []
            line, = ax.plot([], [], lw=3)
            lines.append(line)
            line, = ax.plot([], [], lw=3)
            lines.append(line)
            line, = ax.plot([], [], lw=3)
            lines.append(line)
            for _ in self.obstacle_list:
                line, = ax.plot([], [], lw=3)
                lines.append(line)

            n = len(lines)

            def init():
                lines[0].set_data([], [])
                lines[1].set_data([], [])
                lines[2].set_data([node.x for node in self.path], [node.y for node in self.path])
                for i, obs in enumerate(self.obstacle_list):
                    idx = 3 + i
                    if isinstance(obs, Polygon):
                        x, y = obs.exterior.xy
                        lines[idx].set_data(x, y)
                    if isinstance(obs, LineString):
                        lines[idx].set_data(*obs.xy)
                return lines[:n]

            def animate(i):
                node = self.path[i]
                node_next = self.path[i + 1]

                p1 = (node.x, node.y)
                p2 = (node_next.x, node_next.y)
                vehicle = move_vehicle(self.vg, p1, p2)
                x, y = vehicle.exterior.xy
                lines[0].set_data(x, y)

                vehicle = move_vehicle(self.vg, p1, p2, (1, 1))
                x, y = vehicle.exterior.xy
                lines[1].set_data(x, y)
                return lines[:n]

            anim = FuncAnimation(fig, animate, init_func=init,
                                 frames=len(self.path) - 1, interval=20, blit=True)

            anim.save('car_moving.gif', writer='imagemagick')

    def plot_pseudo_results(self, create_animation: bool = False) -> None:
        """
        Tool to plot and save the results
        """
        plt.clf()
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for ob in self.obstacle_list:
            if isinstance(ob, Polygon):
                x, y = ob.exterior.xy
                plt.plot(x, y)
            if isinstance(ob, LineString):
                plt.plot(*ob.xy)

        plt.plot(self.start.x, self.start.y, "xr")
        x, y = self.end.polygon.exterior.xy
        plt.plot(x, y, 'b')
        angle_polygon = self.end.angle_polygon()
        if angle_polygon:
            x, y = angle_polygon.exterior.xy
            plt.plot(x, y, 'y')
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')

        enlargement_f: float = 0.1
        min_enlargement: float = 10
        x_lim = self.boundaries.x_bounds()
        delta = enlargement_f * (x_lim[1] - x_lim[0]) + min_enlargement
        x_lim = (x_lim[0] - delta, x_lim[1] + delta)
        plt.xlim(x_lim)

        y_lim = self.boundaries.y_bounds()
        delta = enlargement_f * (y_lim[1] - y_lim[0]) + min_enlargement
        y_lim = (y_lim[0] - delta, y_lim[1] + delta)
        plt.ylim(y_lim)

        if self.start.is_yaw_considered and self.end.goal_node.is_yaw_considered:
            plot_arrow(self.start.x, self.start.y, self.start.yaw)
            plot_arrow(self.end.goal_node.x, self.end.goal_node.y, self.end.goal_node.yaw)
        plt.savefig("test")


def main(gx=6.0, gy=10.0):
    """ Dummy example """

    obstacle_list = [Polygon(((5, 4), (5, 6), (10, 6), (10, 4), (5, 4))), LineString([Point(0, 8), Point(5, 8)])]
    bounds: BaseBoundaries = RectangularBoundaries((-2, 15, -2, 15))
    rrt = RRT()
    rrt.planning(start=Node(0, 0), goal=GoalRegion(Node(gx, gy), 0, 0.5, 0.5, 0.5),
                 sampling_bounds=bounds, obstacle_list=obstacle_list, search_until_max_iter=False)
    rrt.plot_results(create_animation=True)


if __name__ == '__main__':
    main()
