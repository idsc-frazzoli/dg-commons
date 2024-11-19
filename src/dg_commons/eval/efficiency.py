from __future__ import annotations

from math import sqrt
from typing import Optional

import numpy as np
from commonroad.scenario.lanelet import Lanelet, LaneletNetwork

from dg_commons import X, U, DgSampledSequence, iterate_with_dt
from dg_commons.geo import PoseState, SE2Transform
from dg_commons.maps.lanes import DgLanelet
from dg_commons.seq.sequence import Timestamp
from dg_commons.sim.goals import RefLaneGoal


def distance_traveled(states: DgSampledSequence[X]) -> float:
    dist: float = 0
    for it in iterate_with_dt(states):
        dist += sqrt((it.v1.x - it.v0.x) ** 2 + (it.v1.y - it.v0.y) ** 2)
    return dist


def actuation_effort(commands: DgSampledSequence[U]) -> float:
    pass


def time_goal_lane_reached(
    lanelet_network: LaneletNetwork,
    goal_lane: RefLaneGoal,
    states: DgSampledSequence[X],
    pos_tol: float = 0.8,
    heading_tol: float = 0.08,
) -> float | None:
    reached_time = None
    for idx, state in enumerate(states.values):
        reached = desired_lane_reached(lanelet_network, goal_lane, state, pos_tol, heading_tol)
        if reached:
            reached_time = float(states.timestamps[idx])
            break
    return reached_time


def desired_lane_reached(
    lanelet_network: LaneletNetwork, goal_lane: RefLaneGoal, state: X, pos_tol: float, heading_tol: float
) -> bool:
    """

    :param state: the last ego state in the simulation
    :param goal_lane: the desired lane
    :return: True if the ego vehicle has reached the goal lane or any of its successors. Reached means the vehicle
    center is close to the lane center and the heading is aligned with the lane.
    """
    ego_posestate = PoseState(x=state.x, y=state.y, psi=state.psi)
    ego_pose = SE2Transform.from_PoseState(ego_posestate)
    ref_lane = goal_lane.ref_lane  # dglanelet
    lane_pose = ref_lane.lane_pose_from_SE2Transform(ego_pose)
    while not lane_pose.along_inside:
        if np.abs(lane_pose.along_lane) < 1.0 or np.abs(lane_pose.along_lane - ref_lane.get_lane_length()) < 1.0:
            # not inside the lane but close enough
            break
        if lane_pose.along_after:
            ref_lane = get_successor_dglane(lanelet_network, ref_lane)
            if ref_lane is not None:
                lane_pose = ref_lane.lane_pose_from_SE2Transform(ego_pose)
            else:
                break
        if lane_pose.along_before:
            ref_lane = get_predecessor_dglane(lanelet_network, ref_lane)
            if ref_lane is not None:
                lane_pose = ref_lane.lane_pose_from_SE2Transform(ego_pose)
            else:
                break

    if goal_lane.is_fulfilled(state):
        return True
    # vehicle still on the road and is inside the desired lane, check its pose
    if (
        lane_pose.lateral_inside
        and lane_pose.distance_from_center < pos_tol
        and abs(lane_pose.relative_heading) < heading_tol
    ):
        return True
    else:
        return False


def get_lanelet_from_dglanelet(lanelet_network: LaneletNetwork, dglanelet: DgLanelet) -> Lanelet:
    ref_point = dglanelet.control_points[1].q.p
    lane_id = lanelet_network.find_lanelet_by_position([ref_point])[0][0]
    lanelet = lanelet_network.find_lanelet_by_id(lane_id)
    return lanelet


def get_successor_lane(lanelet_network: LaneletNetwork, cur_lane: Lanelet | DgLanelet) -> Optional[Lanelet]:
    # note: only one successor is considered for now
    if isinstance(cur_lane, DgLanelet):
        cur_lane = get_lanelet_from_dglanelet(lanelet_network, cur_lane)
    if len(cur_lane.successor) > 0:
        return lanelet_network.find_lanelet_by_id(cur_lane.successor[0])
    else:
        return None


def get_successor_dglane(lanelet_network: LaneletNetwork, cur_lane: Lanelet | DgLanelet) -> Optional[DgLanelet]:
    suc_lanelet = get_successor_lane(lanelet_network, cur_lane)
    if suc_lanelet is None:
        return None
    else:
        return DgLanelet.from_commonroad_lanelet(suc_lanelet)


def get_predecessor_lane(lanelet_network: LaneletNetwork, cur_lane: Lanelet | DgLanelet) -> Optional[Lanelet]:
    # note: only one predecessor is considered for now
    if isinstance(cur_lane, DgLanelet):
        cur_lane = get_lanelet_from_dglanelet(lanelet_network, cur_lane)
    if len(cur_lane.predecessor) > 0:
        return lanelet_network.find_lanelet_by_id(cur_lane.predecessor[0])
    else:
        return None


def get_predecessor_dglane(lanelet_network: LaneletNetwork, cur_lane: Lanelet | DgLanelet) -> Optional[DgLanelet]:
    pre_lanelet = get_predecessor_lane(lanelet_network, cur_lane)
    if pre_lanelet is None:
        return None
    else:
        return DgLanelet.from_commonroad_lanelet(pre_lanelet)
