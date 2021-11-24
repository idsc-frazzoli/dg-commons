from geometry import SE2_from_xytheta
from matplotlib import pyplot as plt
from numpy import deg2rad
from shapely.geometry import Polygon, LinearRing

from dg_commons import apply_SE2_to_shapely_geo
from dg_commons.maps.shapely_viz import ShapelyViz


def polygon_star(x: float, y: float, p: float, t: float) -> Polygon:
    """
    Returns a polygon with a 4 spikes star shape.
    :param x: center of the star
    :param y: center of the star
    :param p: extension of points aligned with x-y axis
    :param t: extension of points aligned with diagonals
    :return:
    """
    points = []
    for i in (-1, 1):
        points.append((x, y + i * p))
        points.append((x + i * t, y + i * t))
        points.append((x + i * p, y))
        points.append((x + i * t, y - i * t))

    return Polygon(points)


if __name__ == "__main__":
    poly = polygon_star(15, 8, 5, 1.5)
    transform = SE2_from_xytheta((3, 3, deg2rad(0)))
    viz = ShapelyViz()
    arena_size = 100
    boundary = LinearRing([[0, 0], [arena_size, 0], [arena_size, arena_size], [0, arena_size]])
    viz.add_shape(boundary, color="r")
    for i in range(10):
        poly = apply_SE2_to_shapely_geo(poly, transform)
        viz.add_shape(poly, facecolor="gold", edgecolor="r")
    viz.ax.autoscale()
    viz.ax.set_aspect("equal")
    plt.show()
