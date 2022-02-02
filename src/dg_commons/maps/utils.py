import numpy as np
from typing import List
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet

__all__ = ['interpolate2d']


def interpolate2d(fraction: float, points: [[float, float], [float, float]]) -> [float, float]:
    """
    Simple interpolation between two points in 2D.
    """
    assert 0.0 <= fraction <= 1.0, "You can only interpolate with a fraction between 0.0 and 1.0"
    x_new = fraction * (points[1][0] - points[0][0]) + points[0][0]
    y_new = fraction * (points[1][1] - points[0][1]) + points[0][1]
    return x_new, y_new


def interpolate1d(fraction: float, points: [float, float]) -> float:
    """
    Simple interpolation between two points in 1D.
    """
    assert 0.0 <= fraction <= 1.0, "You can only interpolate with a fraction between 0.0 and 1.0"
    x_new = fraction * (points[1] - points[0]) + points[0]
    return x_new


# fixme: find input and return types
def get_intermediate_points(previous_beta: float, beta: float, distance_vector: np.ndarray,
                            points_left: np.ndarray, points_right: np.ndarray) -> np.ndarray:
    delta = 1.0 / float(len(distance_vector))
    remainder = beta % delta
    remainder = remainder / delta
    divisor = beta // delta
    previous_divisor = previous_beta // delta
    intermediate_points = []
    if divisor == previous_divisor:
        pass
    elif divisor > previous_divisor:
        # account for case where there may be more than one pair of points
        for point_idx in range(int(divisor - previous_divisor)):
            point = (points_left[divisor + point_idx], points_right[divisor + point_idx],)
            intermediate_points.append(point)

    else:
        raise ValueError("divisor can only be greater or equal than previous_divisor.")

    intermediate_points = np.asarray(intermediate_points)
    return intermediate_points


def get_beta_in_distance_vector(pos: float, points: np.ndarray):
    """
    Compute the progression factor beta in the "distance space".
    :param pos: query point for which to find beta in the distance vector
    :param points: Distance vector
    """
    delta = 1.0 / float(len(points))
    # handle special cases at beginning or end of distance vector
    if pos <= points[0]:
        beta = 0.0
    elif pos >= points[-1]:
        beta = 1.0
    # case when point is inside distance vector
    else:
        idx_before = np.argmin(points < pos) - 1
        idx_after = np.argmax(points > pos)
        fraction = (pos - points[idx_before]) / (points[idx_after] - points[idx_before])
        beta = fraction * delta + idx_before * delta
    return beta


def get_point_from_beta_2d(beta: float, points: np.ndarray):
    """
    Get point from a vector of points, with beta defined as the progression in the distance vector.
    :param beta: progression in distance vector
    :param points: vector of points
    """
    delta = 1.0 / float(len(points))
    remainder = beta % delta
    remainder = remainder / delta
    divisor = beta // delta
    # handle special cases
    if beta == 1.0:
        # point = points[int(divisor)-1]
        point = points[-1]
        pos = point[0], point[1]
    elif beta <= delta:
        pos = interpolate2d(fraction=remainder,
                            points=[points[0], points[1]])
    elif beta + delta >= 1.0:
        pos = interpolate2d(fraction=remainder,
                            points=[points[-2], points[-1]])
    # standard case
    else:
        pos = interpolate2d(fraction=remainder,
                            points=[points[int(divisor)], points[int(divisor + 1)]])
    return pos


"""def make_polygons_from_lanelet(point_list: List[tuple[tuple[float, float]]]):
    Make list of polygons from a list of points.
    For every tuple, first element is left vertex and second element is right vertex
    polygons = []

    for idx, points in enumerate(point_list[:-1]):
        polygon_vertices = points + point_list[idx + 1]
        polygon_vertices = [vert for vert in polygon_vertices]
        current_poly = Polygon(polygon_vertices)  # check mutability
        polygons.append(current_poly)

    return polygons"""


# currently not used
def get_lanelet_center_coordinates(lanelet_network: LaneletNetwork, lanelet_id: int) -> np.ndarray:
    center = lanelet_network.find_lanelet_by_id(lanelet_id).polygon.center
    return center
