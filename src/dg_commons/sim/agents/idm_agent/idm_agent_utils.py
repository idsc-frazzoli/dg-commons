import random
from dataclasses import replace
from functools import lru_cache
from typing import Optional

import numpy as np
from commonroad.scenario.intersection import Intersection
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork
from geometry import translation_angle_scale_from_E2
from geometry.types import SE2value, T2value
from numpy.typing import NDArray
from shapely.geometry import Polygon

from dg_commons import X
from dg_commons.geo import relative_pose
from dg_commons.maps.lanes import DgLanelet, LaneCtrPoint
from dg_commons.sim import PlayerObservations, TModelParameters
from dg_commons.sim.models import extract_pose_from_state
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn
from .vehicle_projected import VehicleStatePrj

"""
Part 1
Geometry-related functions
"""


def compute_approx_curvatures(
    ctrl_points: NDArray[np.float64], lane_lengths: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Computes approximated curvatures from control points.
    [Number of curvatures] = [Number of control points] - 2.
    The curvature is computed as the change of unit tangent vector divided by the change of arc length.

    :param ctrl_points: the list of control points
    :param lane_lengths: the list of lengths between control points
    :return: array of curvatures
    """
    lane_vectors = np.diff(ctrl_points, axis=0)
    lane_vector_norms = np.linalg.norm(lane_vectors, axis=1)
    unit_lane_vectors = lane_vectors / np.concatenate(
        (lane_vector_norms.reshape(-1, 1), lane_vector_norms.reshape(-1, 1)), axis=1
    )
    diff_lane_angles = np.arccos(np.clip(np.diag(unit_lane_vectors[:-1] @ unit_lane_vectors[1:].T), -1, 1))
    approx_curvatures = diff_lane_angles / (lane_lengths[:-1] + lane_lengths[1:])

    return approx_curvatures


def smooth_lanelet(ref_lane: DgLanelet, threshold_curvature: float = 1.0, abs_tol: float = 1e-10) -> DgLanelet:
    """Smooths lanelet by removing control points which cause irregularities in curvatures.

    :param ref_lane: the reference lanelet
    :param threshold_curvature: the threshold of curvature, defaults to 1.0
    :param abs_tol: the absolute tolerance for removing duplicate control points, defaults to 1e-10
    :return: the smoothed lanelet
    """
    ctrl_points = list(ref_lane.control_points)
    new_control_points = []
    smooth_control_points = []
    lane_lengths = np.array(ref_lane.get_lane_lengths())

    # There could be some duplicate control points.
    # Remove them before computing curvatures.
    ctrl_points = np.array([ctrl_point.q.p for ctrl_point in ctrl_points])
    lane_vectors = np.diff(ctrl_points, axis=0)
    lane_vector_norms = np.linalg.norm(lane_vectors, axis=1)
    indices = np.nonzero(lane_vector_norms < abs_tol)[0]
    ctrl_points = np.delete(ctrl_points, indices + 1, axis=0)
    lane_lengths = np.delete(lane_lengths, indices)
    # Deleting elements destructively is dangerous...
    for i, control_point in enumerate(ref_lane.control_points):
        if i - 1 not in indices:
            new_control_points.append(control_point)

    if ctrl_points.size < 3:
        return []

    curvatures = compute_approx_curvatures(ctrl_points, lane_lengths)
    indices = np.nonzero(curvatures > threshold_curvature)[0]

    for i, control_point in enumerate(new_control_points):
        if i - 1 not in indices:
            smooth_control_points.append(control_point)

    return DgLanelet(smooth_control_points)


def compute_ref_lane_polygon(ctrl_points: list[LaneCtrPoint]) -> Polygon:
    """Computes the polygon of reference DgLanelet from its control points.

    :param ctrl_points: control points of the DgLanelet
    :return: the polygon
    """

    def _ctrl_point2vertex(ctrl_point: LaneCtrPoint) -> NDArray[np.float64]:
        center = ctrl_point.q.p
        r = ctrl_point.r
        theta = ctrl_point.q.theta + np.pi / 2
        dxdy = r * np.array([np.cos(theta), np.sin(theta)])
        left_vertex = center + dxdy
        right_vertex = center - dxdy

        return np.array([left_vertex, right_vertex])

    left_right_vertices = np.array([_ctrl_point2vertex(ctrl_point) for ctrl_point in ctrl_points])
    polygon_vertices = np.concatenate((left_right_vertices[:, 0, :], np.flip(left_right_vertices[:, 1, :], 0)))

    return Polygon(polygon_vertices)


@lru_cache(maxsize=128)
def state2beta(ref_lane: DgLanelet, p_state: X) -> float:
    """Maps player state to beta along the lane.

    :param ref_lane: reference DgLanelet
    :param p_state: player state
    :return: beta along the lane
    """
    position = np.array([p_state.x, p_state.y])
    beta, _ = ref_lane.find_along_lane_closest_point_fast(position, tol=1e-3)
    return beta


def state2beta_duo(ref_lane: DgLanelet, p_state: X) -> tuple[float, Optional[float]]:
    """Maps player state to beta duo along the lane.
    If the player is a projected player, (beta, beta of the intersection point) is returned.
    If the player is a normal player, (beta, None) is returned.

    :param ref_lane: reference DgLanelet
    :param p_state: player state
    :return: beta duo
    """
    beta = state2beta(ref_lane, p_state)

    if isinstance(p_state, VehicleStatePrj):
        int_point = np.array(p_state.int_point)
        int_beta, _ = ref_lane.find_along_lane_closest_point_fast(int_point, tol=1e-3)

        return beta, int_beta
    else:
        return beta, None


def compute_gap(polygon_lead: Polygon, polygon: Polygon) -> float:
    """Computes the bumper-to-bumper gap between the leading vehicle and this vehicle.
    Currently, this is referred to as the minimum distance between two polygons.

    :param polygon_lead: polygon of the leading vehicle
    :param polygon: polygon of this vehicle
    :return: minimum distance between two polygons
    """
    return polygon_lead.distance(polygon)


def inside_ref_lane_from_obs(ref_lane: DgLanelet, ref_lane_polygon: Polygon, player_obs: PlayerObservations) -> bool:
    """Checks whether a player is in the reference lane.
    Note that if the reference lane polygon contains the vehicle polygon, there are no intersection points as well.

    :param ref_lane: reference DgLanelet
    :param ref_lane_polygon: Polygon generated from the DgLanelet
    :param player_obs: player observations
    :return: whether a player is in the reference lane
    """
    if ref_lane.is_inside_from_T2value(np.array([player_obs.state.x, player_obs.state.y])):
        return True
    elif player_obs.occupancy is None:
        return False
    else:
        return ref_lane_polygon.intersects(player_obs.occupancy)


"""
Part 2
Policy-related functions
"""


def compute_low_speed_intervals(
    ref_lane: DgLanelet, braking_dist: float = 20.0, max_allowed_curvature: float = 0.1, abs_tol: float = 1e-10
) -> list[list[float, float]]:
    """Computes the low speed intervals from the list of control points of the reference lanelet.
    The intervals are computed in order to make sure the vehicle brakes before turning.
    A braking is considered to be necessary if the curvature of the reference lanelet > max_allowed_curvature.
    The return value is a list of lists of floats: [[start1, end1], [start2, end2], ...].
    Both "start" and "end" are progress along the lane.

    :param ref_lane: the list of control points of the reference lanelet
    :param braking_dist: the braking distance (in meters) before the turn, defaults to 20.0
    :param max_allowed_curvature: the max allowed curvature such that a braking is not necessary, defaults to 0.1
    :param abs_tol: the absolute tolerance for removing duplicate control points.
    :return: the list of low speed intervals
    """
    ctrl_points = ref_lane.control_points
    lane_lengths = np.array(ref_lane.get_lane_lengths())

    # There could be some duplicate control points.
    # Remove them before computing curvatures.
    ctrl_points = np.array([ctrl_point.q.p for ctrl_point in ctrl_points])
    lane_vectors = np.diff(ctrl_points, axis=0)
    lane_vector_norms = np.linalg.norm(lane_vectors, axis=1)
    indices = np.nonzero(lane_vector_norms < abs_tol)[0]
    ctrl_points = np.delete(ctrl_points, indices + 1, axis=0)
    lane_lengths = np.delete(lane_lengths, indices)

    if ctrl_points.size < 3:
        return []

    curvatures = compute_approx_curvatures(ctrl_points, lane_lengths)
    indices = np.nonzero(curvatures > max_allowed_curvature)

    if indices[0].size == 0:
        return []

    progs = np.cumsum(lane_lengths)[indices]
    raw_intervals = [[prog - braking_dist, prog] for prog in progs]

    # Union the overlapped intervals
    intervals = [raw_intervals[0]]
    for interval in raw_intervals:
        if interval[0] <= intervals[-1][1]:
            intervals[-1][1] = interval[1]
        else:
            intervals.append(interval)

    return intervals


def find_lanelet_ids_from_obs(lanelet_network: LaneletNetwork, player_obs: PlayerObservations) -> frozenset[int]:
    """Returns the ids of Lanelet objects which the player is on.
    The position of the player is used to find the Lanelets.
    If no Lanelets are found, an empty set is returned.
    :param lanelet_network: a network of Lanelets
    :param player_obs: player observations
    :raises RuntimeError: a player cannot be at multiple positions at the same time
    :return: the ids of Lanelet objects
    """
    state = player_obs.state
    position = np.array([state.x, state.y])

    lanelets = lanelet_network.find_lanelet_by_position([position])

    # lanelets: list[list[int]]
    # Since we only query the network about one position, we should get only one list of integers.
    # If lanelets[0] is empty, no Lanelet is found.
    if len(lanelets) != 1:
        raise RuntimeError(f"Player @ {position} -- too many positions given.")
    elif not lanelets[0]:
        return frozenset()
    else:
        return frozenset(lanelets[0])


def find_best_lanelet_from_obs(lanelet_network: LaneletNetwork, player_obs: PlayerObservations) -> Optional[Lanelet]:
    """Finds the best Lanelet from player observations.
    The best Lanelet is the Lanelet on which the vehicle has minimum absolute relative heading.
    In other words, we prefer the Lanelet which the vehicle is aligned with.
    If no Lanelets are found, None is returned.

    :param lanelet_network: a network of Lanelets
    :param player_obs: player observations
    :return: the best Lanelet
    """

    def _dg_lanelet2heading(dg_lanelet: DgLanelet, pose: SE2value, tol: float = 1e-3) -> float:
        # Avoid computing lane pose because it could be expensive.
        p, _, _ = translation_angle_scale_from_E2(pose)
        _, pose0 = dg_lanelet.find_along_lane_closest_point_fast(p, tol)
        rel = relative_pose(pose0, pose)
        _, relative_heading, _ = translation_angle_scale_from_E2(rel)
        return abs(relative_heading)

    lanelet_ids = find_lanelet_ids_from_obs(lanelet_network, player_obs)

    if not lanelet_ids:
        return None

    lanelet_ids = list(lanelet_ids)
    lanelet_ids.sort()
    lanelets = [lanelet_network.find_lanelet_by_id(id) for id in lanelet_ids]
    dg_lanelets = [DgLanelet.from_commonroad_lanelet(lanelet) for lanelet in lanelets]
    pose = extract_pose_from_state(player_obs.state)
    lanelet_ids2headings = dict(
        zip(
            lanelet_ids,
            map(
                lambda dg_lanelet: _dg_lanelet2heading(dg_lanelet, pose),
                dg_lanelets,
            ),
        )
    )
    # Here we sort the dictionary by value (i.e. relative heading).
    # The best Lanelet is the Lanelet on which the vehicle has minimum absolute relative heading.
    lanelet_id = sorted(lanelet_ids2headings.items(), key=lambda item: item[1])[0][0]
    return lanelet_network.find_lanelet_by_id(lanelet_id)


def predict_dg_lanelet_from_obs(
    lanelet_network: LaneletNetwork,
    player_obs: PlayerObservations,
    max_length: float,
    seed: Optional[int] = None,
) -> Optional[DgLanelet]:
    """Predicts the DgLanelet the player is likely to take from the observations.
    To achieve this, we first find the best Lanelet.
    Then we get a list of merged Lanelets by merging successors from the best Lanelet.
    Finally, we randomly choose one Lanelet from the list and generate the predicted DgLanelet.
    This random choice is somewhat reasonable as a human driver could only guess the trajectory of other vehicles from
    the Lanelet.
    The driver will never know the reference Lanelets of other vehicles.

    :param lanelet_network: a network of Lanelets
    :param player_obs: player observations
    :param max_length: max length of the predicted DgLanelet
    :param seed: the random seed
    :return: the predicted DgLanelet
    """
    best_lanelet = find_best_lanelet_from_obs(lanelet_network, player_obs)

    if best_lanelet is None:
        return None

    merged_lanelets, _ = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
        best_lanelet, lanelet_network, max_length
    )

    if not merged_lanelets:
        return None

    random.seed(a=seed)
    predicted_merged_lanelet = random.choice(merged_lanelets)
    return DgLanelet.from_commonroad_lanelet(predicted_merged_lanelet)


def find_intersections(lanelet_network: LaneletNetwork, player_obs: PlayerObservations) -> dict[int, Intersection]:
    """Finds intersections on the current Lanelets which the agent is on.

    :param lanelet_network: a network of Lanelets
    :param player_obs: player's observations
    :return: a dictionary that maps lanelet_id to Intersection object
    """
    lanelet_ids = find_lanelet_ids_from_obs(lanelet_network, player_obs)
    lanelets2intersections = lanelet_network.map_inc_lanelets_to_intersections

    return {_id: lanelets2intersections.get(_id) for _id in lanelet_ids}


def compute_projected_obs(
    player_obs: PlayerObservations, path_int: T2value, predicted_dg_lanelet: DgLanelet, my_dg_lanelet: DgLanelet
) -> PlayerObservations:
    """Computes the player observations after the projection.
    The distance from the player to the intersection point is preserved during the projection.

    :param player_obs: original player observations
    :param path_int: intersection point of two paths
    :param predicted_dg_lanelet: predicted DgLanelet of another player
    :param my_dg_lanelet: DgLanelet of the agent
    :return: projected player observations
    """
    path_int_along_lane = predicted_dg_lanelet.along_lane_from_T2value(path_int, fast=True)
    path_int_along_my_lane = my_dg_lanelet.along_lane_from_T2value(path_int, fast=True)
    state = player_obs.state
    player_along_lane = predicted_dg_lanelet.along_lane_from_T2value(np.array([state.x, state.y]), fast=True)
    player_along_my_lane = path_int_along_my_lane - (path_int_along_lane - player_along_lane)

    projected_pose = my_dg_lanelet.center_point_fast_SE2Transform(
        my_dg_lanelet.beta_from_along_lane(player_along_my_lane)
    )
    projected_x, projected_y = projected_pose.p
    projected_psi = projected_pose.theta
    # player_obs.occupancy should be unchanged.
    # player_obs.state.vx/delta/idx should be unchanged.
    projected_state = replace(player_obs.state, x=projected_x, y=projected_y, psi=projected_psi)

    if isinstance(projected_state, VehicleStateDyn):
        projected_state_prj = VehicleStatePrj.from_vehicle_state_dyn(projected_state, int_point=path_int)
    elif isinstance(projected_state, VehicleState):
        projected_state_prj = VehicleStatePrj.from_vehicle_state(projected_state, int_point=path_int)
    else:
        Warning("Unsupported type of vehicle state")

    projected_obs = replace(player_obs, state=projected_state_prj)

    return projected_obs


"""
Part 3
Miscellaneous
"""


def clip_cmd_value(cmd_value: float, limits: tuple[float, float]) -> float:
    """Clips the command value within the limits.
    # todo just use numpy?
    :param cmd_value: the command value
    :param limits: the limits
    :return: the clipped command value
    """
    return min(max(cmd_value, limits[0]), limits[1])


def apply_speed_constraint(speed: float, acceleration: float, p: TModelParameters):
    """Enforces acceleration limits if the maximum speed is reached"""
    if (speed <= p.vx_limits[0] and acceleration < 0) or (speed >= p.vx_limits[1] and acceleration > 0):
        acceleration = 0
    return acceleration
