import time
from math import pi
from typing import List, Tuple, Mapping, Dict

import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork
from geometry import T2value, SO2value, SO2_from_angle, SE2value
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.geometry.base import BaseGeometry
from toolz import remove

from dg_commons import X, PlayerName, logger
from dg_commons.maps.lanes import DgLanelet, DgLanePose
from dg_commons.sim.models.model_structures import ModelGeometry


class CollisionException(Exception):
    pass


_rot90: SO2value = SO2_from_angle(pi / 2)


def velocity_of_P_given_A(vel: T2value, omega: float, vec_ap: T2value) -> T2value:
    """Compute velocity of point P given velocity at A, rotational velocity of the rigid body and vector AP"""
    # rotate by 90 to be equivalent to cross product omega x r_ap
    return vel + omega * (_rot90 @ vec_ap)


def _find_intersection_points(a_shape: Polygon, b_shape: BaseGeometry) -> List[Tuple[float, float]]:
    """#todo"""
    int_shape = a_shape.intersection(b_shape)
    if isinstance(int_shape, Polygon):
        points = list(int_shape.exterior.coords[:-1])
    elif isinstance(int_shape, MultiPolygon):
        int_shape: Polygon = int_shape.minimum_rotated_rectangle
        logger.warn(
            f"Found multiple contact points between two geometries, collision resolution might not be accurate. "
            f"Use a smaller physics step for improved accuracy."
        )
        points = list(int_shape.exterior.coords[:-1])
    elif isinstance(int_shape, LineString):
        points = list(int_shape.coords)
    else:
        raise CollisionException(f"Intersection shape is not a polygon: {int_shape}")

    def is_contained_in_aorb(p) -> bool:
        shapely_point = Point(p).buffer(1.0e-9)
        return a_shape.contains(shapely_point) or b_shape.contains(shapely_point)

    points = list(remove(is_contained_in_aorb, points))
    if not len(points) == 2:
        from matplotlib import pyplot as plt

        plt.figure()
        plt.plot(*a_shape.exterior.xy, "b")
        if isinstance(b_shape, Polygon):
            plt.plot(*b_shape.exterior.xy, "r")
        else:
            plt.plot(*b_shape.xy, "r")
        for p in points:
            plt.plot(*p, "o")
        plt.savefig(f"coll_debug{time.time()}.png")
        raise CollisionException(f"At the moment collisions with {len(points)} intersecting points are not supported")
    return points


def get_impact_point_direction(state: X, impact_point: Point) -> float:
    """returns the impact point angle wrt to the vehicle"""
    # Direction of Force (DOF) -> vector that goes from car center to impact point
    abs_angle_dof = np.arctan2(impact_point.y - state.y, impact_point.x - state.x)
    car_heading: float = state.psi
    return abs_angle_dof - car_heading


def compute_impact_geometry(a: Polygon, b: BaseGeometry) -> (np.ndarray, Point):
    """
    This computes the normal of impact between models a and b
    :param a: Polygon object
    :param b: Polygon object
    :return: normal of impact and the impact point
    """
    assert not a.touches(b)
    intersecting_points = _find_intersection_points(a, b)
    impact_point = LineString(intersecting_points).interpolate(0.5, normalized=True)
    first, second = intersecting_points
    dxdy_surface = (second[0] - first[0], second[1] - first[1])
    normal = np.array([-dxdy_surface[1], dxdy_surface[0]])
    normal /= np.linalg.norm(normal)
    r_ap = np.array(impact_point.coords[0]) - np.array(a.centroid.coords[0])
    if np.dot(r_ap, normal) < 0:
        # rotate by 180 if pointing into the inwards of A
        normal *= -1
    return normal, impact_point


def compute_impulse_response(
    n: np.ndarray, vel_ab: np.ndarray, r_ap: np.ndarray, r_bp: np.ndarray, a_geom: ModelGeometry, b_geom: ModelGeometry
) -> float:
    """
    The impulse J is defined in terms of force F and time period ∆t
    J = F*∆t = ma*∆t = m *∆v/∆t *∆t = m*∆v
    :param n:             Vector onto which to project rel_v (normally, n or t)
    :param vel_ab:          Relative velocity between a and b
    :param r_ap:            Vector from CG of a to collision point P
    :param r_bp:            Vector from CG of b to collision point P
    :param a_geom:          Geometry of model a
    :param b_geom:          Geometry of model b
    :return:
    """
    # Restitution coefficient -> represents the "bounciness" of the vehicle
    e = min(a_geom.e, b_geom.e)
    j = -(1 + e) * np.dot(vel_ab, n)
    rot_part = (np.cross(r_ap, n) ** 2 / a_geom.Iz) + (np.cross(r_bp, n) ** 2 / b_geom.Iz)
    j /= 1 / a_geom.m + 1 / b_geom.m + rot_part
    return j


def velocity_after_collision(n: np.ndarray, velocity: np.ndarray, m: float, j: float) -> np.ndarray:
    """
    This computes the velocity after the collision based on the impulse resolution method
    :param n:           normal of impact
    :param velocity:   velocity right before the collision
    :param m:           vehicle mass
    :param j:           impulse scalar
    :return: velocity after the impulse has been applied
    """
    return velocity + (j * n) / m


def rot_velocity_after_collision(r: np.ndarray, n: np.ndarray, omega: np.ndarray, Iz: float, j: float) -> float:
    """
    This computes the velocity after the collision based on the impulse resolution method
    :param r: Contact vector
    :param n: Normal of impact
    :param omega: rot velocity before impact
    :param Iz: rotational inertia
    :param j: Impulse
    :return:
    """
    return omega + np.cross(r, j * n) / Iz


def kinetic_energy(velocity: np.ndarray, m: float) -> float:
    """
    This computes the kinetic energy lost in the collision as 1/2*m*(vf^2-vi^2)
    :param velocity:   velocity right before the collision
    :param m:     mass of the object
    :return:
    """
    # todo also rotational components?
    return 0.5 * m * np.linalg.norm(velocity) ** 2


def check_who_is_at_fault(
    p_poses: Mapping[PlayerName, SE2value], impact_point: Point, lanelet_network: LaneletNetwork
) -> Mapping[PlayerName, bool]:
    """
    This functions checks who is at fault in a collision.
    First_check who was in an illegal state, if you were in an illegal state you are at fault.
    You are in an illegal state if your pose is illegal for every
    :param p_poses:
    :param impact_point: #fixme this could be made into a np.array, we do not use any shapely stuff atm
    :param lanelet_network:
    :return:
    """
    # first_check who is in an illegal state, if you are in an illegal state you are at fault
    lanesid_at_impact = lanelet_network.find_lanelet_by_position([np.array([impact_point.x, impact_point.y])])[0]
    dglanes_at_impact = [
        DgLanelet.from_commonroad_lanelet(lanelet_network.find_lanelet_by_id(id)) for id in lanesid_at_impact
    ]
    who_is_at_fault: Dict[PlayerName, bool] = {p: False for p in p_poses}
    for p, p_pose in p_poses.items():
        dglane_poses = [dglane.lane_pose_from_SE2_generic(p_pose) for dglane in dglanes_at_impact]
        illegal_poses = [_is_illegal_lanepose(pose) for pose in dglane_poses]
        if all(illegal_poses):
            who_is_at_fault[p] = True

    if not all(who_is_at_fault.values()):
        # if you are not in an illegal state you are at fault if also...
        # todo coming from the right
        pass
        # get_impact_point_direction()

    return who_is_at_fault


def _is_illegal_lanepose(dglanepose: DgLanePose) -> bool:
    condition = not dglanepose.correct_direction or not dglanepose.inside
    return condition
