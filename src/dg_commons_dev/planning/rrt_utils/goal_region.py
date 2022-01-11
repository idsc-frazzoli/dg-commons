import numpy as np
from dg_commons_dev.planning.rrt_utils.sampling import PolygonBoundaries
from dg_commons_dev.planning.rrt_utils.utils import Node
from typing import Tuple, Optional
import random
from dg_commons_dev.planning.rrt_utils.dubins_path_planning import mod2pi
import math
from geometry import SE2_from_translation_angle, SE2value
from dg_commons.geo import SE2_apply_T2
from shapely.geometry import Polygon, Point


class GoalRegion:
    """
    Class for defining a goal region
    """
    def __init__(self, goal_node: Node, region_orientation: float,
                 tol_x: float, tol_y: float, tol_theta: float = None):
        self.goal_node: Node = goal_node
        self.transformation: SE2value = SE2_from_translation_angle(np.array([goal_node.x, goal_node.y]),
                                                                   region_orientation)

        x_min, x_max, y_min, y_max = -tol_x, tol_x, -tol_y, tol_y
        pos1 = tuple(SE2_apply_T2(self.transformation, np.array([x_min, y_min])))
        pos2 = tuple(SE2_apply_T2(self.transformation, np.array([x_min, y_max])))
        pos3 = tuple(SE2_apply_T2(self.transformation, np.array([x_max, y_max])))
        pos4 = tuple(SE2_apply_T2(self.transformation, np.array([x_max, y_min])))
        self.polygon: Polygon = Polygon((pos1, pos2, pos3, pos4, pos1))
        self._tol_region: PolygonBoundaries = PolygonBoundaries(self.polygon)

        if tol_theta and goal_node.is_yaw_considered:
            self._theta_region: Tuple[float, float] = (mod2pi(goal_node.yaw - tol_theta),
                                                       mod2pi(goal_node.yaw + tol_theta))
        else:
            self._theta_region = None

    def sample_node(self) -> Node:
        """
        Sample a node from goal region
        @return: sampled node
        """
        x, y = self._tol_region.random_sampling()

        if self._theta_region:
            theta_min, theta_max = self._theta_region
            if theta_max < theta_min:
                theta = random.uniform(self._theta_region[0] - 2 * math.pi, self._theta_region[1])
            else:
                theta = random.uniform(self._theta_region[0], self._theta_region[1])
            return Node(x, y, theta)
        else:
            return Node(x, y)

    def inside(self, other: Node) -> bool:
        """
        Check if node is inside tolerance region
        @param other: node to check
        @return: True if inside, false otherwise
        """
        point: Point = Point(other.x, other.y)
        in_xy: bool = self.polygon.contains(point)

        if self._theta_region:
            if not other.yaw:
                return False

            theta_min, theta_max = self._theta_region
            o_theta = mod2pi(other.yaw)

            if theta_max < theta_min:
                in_theta = (theta_min <= o_theta <= 2 * math.pi) or (0 <= o_theta <= theta_max)
            else:
                in_theta = (theta_min <= o_theta <= theta_max)
            return in_xy and in_theta
        else:
            return in_xy

    def angle_polygon(self) -> Optional[Polygon]:
        """
        Returns a polygon (triangle) representing the theta tolerance
        @return: the polygon
        """
        if self._theta_region:
            length = 3
            pos = np.array([self.goal_node.x, self.goal_node.y])
            pos1 = tuple(pos)
            pos2 = tuple(length * np.array([math.cos(self._theta_region[0]), math.sin(self._theta_region[0])]) + pos)
            pos3 = tuple(length * np.array([math.cos(self._theta_region[1]), math.sin(self._theta_region[1])]) + pos)
            return Polygon((pos1, pos2, pos3, pos1))
        else:
            return None
