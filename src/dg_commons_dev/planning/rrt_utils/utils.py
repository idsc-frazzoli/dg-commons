from typing import List, Optional, Tuple, Union
import math
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from dg_commons.sim.models.vehicle import VehicleGeometry


class Node:
    """
    RRT Node
    """

    def __init__(self, x: float, y: float, theta: Optional[float] = None):
        self.x: float = x
        self.y: float = y
        self._yaw: Optional[float] = theta

        self.path_x: List[float] = []
        self.path_y: List[float] = []
        self.path_yaw: List[float] = []
        self.parent: Optional["Node"] = None
        self.cost: float = 0

        self.is_yaw_considered: bool = self.yaw is not None

    @property
    def yaw(self) -> float:
        return self._yaw

    @yaw.setter
    def yaw(self, val: float):
        """
        Yaw setter
        @param val: yaw value
        """
        self._yaw = val
        self.is_yaw_considered = True

    def pose(self) -> Union[Tuple[float, float, float], Tuple[float, float]]:
        """
        @return: The goal, that might be a tuple (x, y) or a triplet (x, y, yaw) if yaw is considered
        """
        if self.is_yaw_considered:
            return self.x, self.y, self.yaw
        else:
            return self.x, self.y

    def same(self, other: "Node") -> bool:
        """
        Check if two node are equal
        @param other: Other Node
        @return:
        """
        def close(val1, val2, tol=10e-6):
            return abs(val1 - val2) < tol

        same_x_y: bool = close(self.x, other.x) and close(self.y, other.y)
        if self.is_yaw_considered:
            # TODO: fix
            return (self.yaw, other.yaw) and same_x_y
        else:
            return same_x_y


def calc_distance_and_angle(from_node: Node, to_node: Node) -> Tuple[float, float]:
    """
    Compute magnitude and orientation of vector from from_node to to_node
    @param from_node: starting node
    @param to_node: target node
    @return: distance, angle
    """
    dx = to_node.x - from_node.x
    dy = to_node.y - from_node.y
    d = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    return d, theta


def move_vehicle(vg: VehicleGeometry, p1: Tuple[float, float], p2: Tuple[float, float],
                 enlargement: Tuple[float, float] = (1.0, 1.0)) -> Polygon:
    """
    Create polygon with car dimension and path pose: p1 and p2 are two subsequent point on the path.
    @param vg: Vehicle geometry, important here are front and rear axle distances
    @param p1: Point on path
    @param p2: Subsequent point on path
    @param enlargement: Factor to enlarge vehicle dimension to be more conservative (length, width)
    @return: Generated polygon
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = math.atan2(dy, dx) - math.pi / 2
    step_size = math.hypot(dx, dy)

    y_min, y_max = - enlargement[0] * vg.lr, enlargement[0] * vg.lf + step_size
    x_min, x_max = - enlargement[1] * vg.width / 2, enlargement[1] * vg.width / 2

    shape = Polygon(((x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)))
    shape: Polygon = rotate(shape, angle=ang, use_radians=True)
    shape: Polygon = translate(shape, xoff=p1[0], yoff=p1[1])

    return shape
