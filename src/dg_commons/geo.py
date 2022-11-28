from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import numpy as np
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object
from geometry import (
    SE2value,
    translation_from_SE2,
    SE2,
    T2value,
    translation_angle_from_SE2,
    SE2_from_translation_angle,
    linear_angular_from_se2,
)
from shapely.affinity import affine_transform
from shapely.geometry.base import BaseGeometry
from commonroad_dc.pycrcc import Polygon as CommonRoadPolygon
from shapely.geometry import Polygon, LineString

"""
Some of these structures and operations are are taken from `duckietown-world` with minor modifications
"""


@dataclass
class PoseState:
    x: float
    y: float
    psi: float


def norm_between_SE2value(p0: SE2value, p1: SE2value, ord=None) -> float:
    t0 = translation_from_SE2(p0)
    t1 = translation_from_SE2(p1)
    return np.linalg.norm(t0 - t1, ord=ord)


def SE2_interpolate(q0: SE2value, q1: SE2value, alpha: float) -> SE2value:
    """Interpolate on SE2 manifold"""
    alpha = float(alpha)
    v = SE2.algebra_from_group(SE2.multiply(SE2.inverse(q0), q1))
    vi = v * alpha
    q = np.dot(q0, SE2.group_from_algebra(vi))
    return q


def SE2_apply_T2(q: SE2value, p: T2value) -> T2value:
    """Apply translation to SE2value"""
    pp = [p[0], p[1], 1]
    res = np.dot(q, pp)
    return res[:2]


def relative_pose(base: SE2value, pose: SE2value) -> SE2value:
    assert isinstance(base, np.ndarray), base
    assert isinstance(pose, np.ndarray), pose
    return np.dot(np.linalg.inv(base), pose)


def get_distance_SE2(q0: SE2value, q1: SE2value) -> float:
    # fixme test
    g = SE2.multiply(SE2.inverse(q0), q1)
    v = SE2.algebra_from_group(g)
    linear, angular = linear_angular_from_se2(v)
    return np.linalg.norm(linear)


class SE2Transform:
    def __init__(self, p: Sequence[float], theta: float):
        self.p: np.ndarray = np.array(p, dtype="float64")
        self.theta: float = float(theta)

    def __repr__(self):
        d = np.rad2deg(self.theta)
        return "SE2Transform(%s,%.1f)" % (self.p.tolist(), d)

    @classmethod
    def identity(cls) -> "SE2Transform":
        return SE2Transform([0.0, 0.0], 0.0)

    @classmethod
    def from_SE2(cls, q: SE2value) -> "SE2Transform":
        """From a matrix"""
        translation, angle = translation_angle_from_SE2(q)
        return SE2Transform(translation, angle)

    @classmethod
    def from_PoseState(cls, x: PoseState) -> "SE2Transform":
        """From a PoseState object"""
        return SE2Transform([x.x, x.y], x.psi)

    def as_SE2(self) -> SE2value:
        M = SE2_from_translation_angle(self.p, self.theta)
        return M


def apply_SE2_to_shapely_geo(shapely_geometry: BaseGeometry, se2_value: SE2value) -> BaseGeometry:
    """Apply SE2 transform to shapely geometry"""
    coeffs = [se2_value[0, 0], se2_value[0, 1], se2_value[1, 0], se2_value[1, 1], se2_value[0, 2], se2_value[1, 2]]
    return affine_transform(shapely_geometry, coeffs)


def transform_xy(q: np.ndarray, points: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float]]:
    """Transform a list of points according to q"""
    points_array = np.array([(x, y, 1) for x, y in points]).T
    points = q @ points_array
    x = points[0, :]
    y = points[1, :]
    return tuple(zip(x, y))


def sPolygon2crPolygon(shapely_polygon: Union[Polygon, LineString]) -> CommonRoadPolygon:
    """Convert a shapely polygon to a CommonRoad polygon
    Interface available at
        https://gitlab.lrz.de/tum-cps/commonroad-drivability-checker/-/blob/master/cpp/collision/src/narrowphase/polygon.cc
    Numbers for triangulation taken from
        https://gitlab.lrz.de/tum-cps/commonroad-drivability-checker/-/blob/master/cpp/collision/include/collision/plugins/triangulation/triangulate.h
    Careful in triangulating with duplicate vertices which might give segfault in triangle library, see:
        https://github.com/drufat/triangle/issues/2#issuecomment-583812662 and comment here:
        https://gitlab.lrz.de/tum-cps/commonroad-drivability-checker/-/blob/master/commonroad_dc/collision/collision_detection/scenario.py
    """
    if isinstance(shapely_polygon, LineString):
        shapely_polygon = shapely_polygon.buffer(0.01)
    vertices = np.array(list(zip(*shapely_polygon.exterior.xy)))
    if all(np.equal(vertices[0], vertices[-1])):
        vertices = vertices[:-1]
    # cr_poly = CommonRoadPolygon(vertices, 0.125, 5)
    cr_poly = CommonRoadPolygon(vertices, [])
    return cr_poly
