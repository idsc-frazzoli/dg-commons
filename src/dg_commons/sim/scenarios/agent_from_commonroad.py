from typing import List, Tuple

from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad.scenario.trajectory import State
from geometry import T2value
from zuper_commons.types import ZException

from dg_commons import SE2Transform, Color
from dg_commons.maps.lanes import DgLanelet, LaneCtrPoint
from dg_commons.sim.agents.agent import Agent
from dg_commons.sim.agents.lane_follower import LFAgent
from dg_commons.sim.models import Pacejka
from dg_commons.sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn
from dg_commons.sim.models.vehicle_structures import VehicleGeometry, CAR
from dg_commons.sim.models.vehicle_utils import VehicleParameters


class NotSupportedConversion(ZException):
    pass


def model_agent_from_dynamic_obstacle(
    dyn_obs: DynamicObstacle, lanelet_network: LaneletNetwork, color: Color = "royalblue"
) -> (VehicleModelDyn, Agent):
    """
    This function aims to create a non-playing character (fixed sequence of commands) in our simulation environment from
    a dynamic obstacle of commonroad (fixed sequence of states).
    # fixme currently only cars are supported
    # fixme this function needs to be improved...
    :param color:
    :param dyn_obs:
    :param lanelet_network:
    :return:
    """
    if not dyn_obs.obstacle_type == ObstacleType.CAR:
        raise NotSupportedConversion(commonroad=dyn_obs.obstacle_type)

    axle_length_ratio = 0.8  # the distance between wheels is less than the car body
    axle_width_ratio = 0.95  # the distance between wheels is less than the car body

    l = dyn_obs.obstacle_shape.length * axle_length_ratio
    dtheta = dyn_obs.prediction.trajectory.state_list[0].orientation - dyn_obs.initial_state.orientation
    delta = dtheta / l
    x0 = VehicleStateDyn(
        x=dyn_obs.initial_state.position[0],
        y=dyn_obs.initial_state.position[1],
        theta=dyn_obs.initial_state.orientation,
        vx=dyn_obs.initial_state.velocity,
        delta=delta,
    )
    mass, rot_inertia = _estimate_mass_inertia(length=dyn_obs.obstacle_shape.length, width=dyn_obs.obstacle_shape.width)
    w_half = dyn_obs.obstacle_shape.width / 2 * axle_width_ratio
    vg = VehicleGeometry(
        vehicle_type=CAR,
        w_half=w_half,
        m=mass,
        Iz=rot_inertia,
        lf=l / 2.0,
        lr=l / 2.0,
        e=0.6,
        c_drag=0.3756,
        c_rr_f=0.003,
        c_rr_r=0.003,
        a_drag=2,
        color=color,
    )
    vp = VehicleParameters.default_car()
    model = VehicleModelDyn(
        x0=x0, vg=vg, vp=vp, pacejka_front=Pacejka.default_car_front(), pacejka_rear=Pacejka.default_car_rear()
    )

    # Agent
    dglane = infer_lane_from_dyn_obs(dyn_obs=dyn_obs, network=lanelet_network)
    agent = LFAgent(dglane)
    return model, agent


def _estimate_mass_inertia(length: float, width: float) -> Tuple[float, float]:
    """#todo justify and fix this empirical formulas"""
    alpha = 50
    beta = 1.6
    area = length * width
    mass = alpha * area ** beta
    inertia = mass * (length + width) / 6
    return mass, inertia


def dglane_from_position(
    p: T2value, network: LaneletNetwork, init_lane_selection: int = 0, succ_lane_selection: int = 0
) -> DgLanelet:
    """Gets the first merged lane from the current position"""
    # todo add possibility to select the number of the lane successor (0 by default)
    lane_id = network.find_lanelet_by_position(
        [
            p,
        ]
    )
    assert len(lane_id[0]) > 0, p
    lane = network.find_lanelet_by_id(lane_id[0][init_lane_selection])
    merged_lane = Lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet=lane, network=network)[0][
        succ_lane_selection
    ]
    return DgLanelet.from_commonroad_lanelet(merged_lane)


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
