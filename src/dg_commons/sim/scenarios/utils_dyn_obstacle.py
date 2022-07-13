from typing import List

import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.trajectory import State

from dg_commons import SE2Transform
from dg_commons.maps.lanes import DgLanelet, LaneCtrPoint

__all__ = ["infer_lane_from_dyn_obs", "is_dyn_obstacle_static"]


def infer_lane_from_dyn_obs(dyn_obs: DynamicObstacle, network: LaneletNetwork) -> DgLanelet:
    """Tries to find a lane corresponding to the trajectory, if no lane is found it creates one from the trajectory"""
    init_position = dyn_obs.initial_state.position
    end_position = dyn_obs.prediction.trajectory.state_list[-1].position

    init_ids = network.find_lanelet_by_position([init_position])[0]
    final_ids = network.find_lanelet_by_position([end_position])[0]
    if not init_ids or not final_ids:
        return _dglane_from_trajectory(dyn_obs.prediction.trajectory.state_list)

    for init_id in init_ids:
        lanelet = network.find_lanelet_by_id(init_id)
        merged_lanelets, _ = lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet=lanelet, network=network)
        for merged_lanelet in merged_lanelets:
            if any(str(id) in str(merged_lanelet.lanelet_id) for id in final_ids):
                return DgLanelet.from_commonroad_lanelet(merged_lanelet)
    # if we reach this point we did not find any lanelet,
    # this might occur because of change of lanes or similar by the dyn obstacle,
    # we create a 'fake' lane from its trajectory
    return _dglane_from_trajectory(dyn_obs.prediction.trajectory.state_list)


def _dglane_from_trajectory(states: List[State], width: float = 3) -> DgLanelet:
    control_points: List[LaneCtrPoint] = []
    for state in states:
        q = SE2Transform(p=state.position, theta=state.orientation)
        control_points.append(LaneCtrPoint(q=q, r=width / 2))
    return DgLanelet(control_points=control_points)


def is_dyn_obstacle_static(dyn_obs: DynamicObstacle, tol: float = 0.2) -> bool:
    """
    Checks if a dynamic obstacle is actually static (e.g., a parked car)
    :param dyn_obs: the dynamic obstacle
    :param tol: distance tolerance to consider an obstacle static
    :return:
    """
    init_position = dyn_obs.initial_state.position
    end_position = dyn_obs.prediction.trajectory.state_list[-1].position
    return np.linalg.norm(init_position - end_position) < tol
