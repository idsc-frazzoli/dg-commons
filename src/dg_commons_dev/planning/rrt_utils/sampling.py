from dg_commons_dev.planning.rrt_utils.utils import Node
import math
from abc import ABC, abstractmethod
import random
from typing import Tuple
from shapely.geometry import Polygon, Point


class BaseBoundaries(ABC):
    """
    Base class for 2D bounded sampling
    """

    @abstractmethod
    def random_sampling(self) -> Tuple[float, float]:
        """
        Sample uniformly at random inside the boundaries, returns (x, y)
        """
        pass

    @abstractmethod
    def x_bounds(self) -> Tuple[float, float]:
        """ Returns x_bounds of the sampling space """
        pass

    @abstractmethod
    def y_bounds(self) -> Tuple[float, float]:
        """ Returns y bounds of the sampling space """
        pass


class RectangularBoundaries(BaseBoundaries):
    """
    Class for 2D bounded sampling inside a rectangle
    """

    def __init__(self, boundaries: Tuple[float, float, float, float]):
        self.x_min, self.x_max = boundaries[0], boundaries[1]
        self.y_min, self.y_max = boundaries[2], boundaries[3]

    def x_bounds(self) -> Tuple[float, float]:
        """
        @return: tuple with x boundaries: (min, max)
        """
        return self.x_min, self.x_max

    def y_bounds(self) -> Tuple[float, float]:
        """
        @return: tuple with y boundaries: (min, max)
        """
        return self.y_min, self.y_max

    def random_sampling(self) -> Tuple[float, float]:
        """
        Sample uniformly at random inside the rectangle
        @return: x and y position
        """
        x: float = random.uniform(self.x_min, self.x_max)
        y: float = random.uniform(self.y_min, self.y_max)
        return x, y


class PolygonBoundaries(BaseBoundaries):
    """
    Class for 2D bounded sampling inside a polygon
    """

    def __init__(self, boundaries: Polygon):
        self.polygon: Polygon = boundaries
        min_x, min_y, max_x, max_y = self.polygon.bounds
        self.x_b = (min_x, max_x)
        self.y_b = (min_y, max_y)

    def random_sampling(self) -> Tuple[float, float]:
        """
        Sample uniformly at random inside the polygon
        @return: x and y position
        """
        point = None
        while not point:
            sh_point = Point(random.uniform(*self.x_b), random.uniform(*self.y_b))
            if self.polygon.contains(sh_point):
                point = (sh_point.x, sh_point.y)
        return point

    def x_bounds(self) -> Tuple[float, float]:
        """
        @return: tuple with x boundaries: (min, max)
        """
        return self.x_b

    def y_bounds(self) -> Tuple[float, float]:
        """
        @return: tuple with y boundaries: (min, max)
        """
        return self.y_b


def uniform_sampling(boundaries: BaseBoundaries, goal_region, goal_sample_rate: float,
                     limit_angles: Tuple[float, float] = (-math.pi, math.pi)) -> Node:
    """
    Sample a random node uniformly from available space.
    @param boundaries: limits the area, from which the sample is taken
    @param goal_region: goal region
    @param goal_sample_rate: how often, on average, should the goal pose be sampled in %
    @param limit_angles: Sampling space for angle
    @return: A node with the sampled position (and sampled orientation if requested).
    """
    goal_node: Node = goal_region.goal_node

    if random.randint(0, 100) > goal_sample_rate:
        pose: Tuple[float, float] = boundaries.random_sampling()

        if goal_node.is_yaw_considered:
            angle = random.uniform(*limit_angles)
            pose += (angle, )
        rnd = Node(*pose)
    else:
        rnd = goal_region.sample_node()
    return rnd
