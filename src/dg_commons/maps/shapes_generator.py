import math
import random
from typing import Tuple, Sequence

import numpy as np

Points = Sequence[Tuple[float, float]]


def create_star_polygon(x: float, y: float, p: float, t: float) -> Points:
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

    return points


def create_random_starshaped_polygon(
    ctr_x: float, ctr_y: float, avg_radius: float, irregularity: float, spikiness: float, n_vertices: int
) -> Points:
    """Start with the centre of the polygon at ctrX, ctrY, then creates the polygon by sampling points on a circle around the centre.
    Random noise is added by varying the angular spacing between sequential points,
    and by varying the radial distance of each point from the centre.

    :param ctr_x, ctr_y: - coordinates of the "centre" of the polygon
    :param avg_radius: - in px, the average radius of this polygon, this roughly controls how large the polygon is, really only useful for order of magnitude.
    :param irregularity: - [0,1] indicating how much variance there is in the angular spacing of vertices. [0,1] will map to [0, 2pi/numberOfVerts]
    :param spikiness: - [0,1] indicating how much variance there is in each vertex from the circle of radius aveRadius. [0,1] will map to [0, avg_radius]
    :param n_vertices: - self-explanatory
    :return : a list of vertices, in CCW order.
    """

    irregularity = np.clip(irregularity, 0, 1) * 2 * math.pi / n_vertices
    spikiness = np.clip(spikiness, 0, 1) * avg_radius

    # generate n angle steps
    angleSteps = []
    lower = (2 * math.pi / n_vertices) - irregularity
    upper = (2 * math.pi / n_vertices) + irregularity
    sum = 0
    for i in range(n_vertices):
        tmp = random.uniform(lower, upper)
        angleSteps.append(tmp)
        sum += tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2 * math.pi)
    for i in range(n_vertices):
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(n_vertices):
        r_i = np.clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        x = ctr_x + r_i * math.cos(angle)
        y = ctr_y + r_i * math.sin(angle)
        points.append((int(x), int(y)))
        angle = angle + angleSteps[i]

    return points
