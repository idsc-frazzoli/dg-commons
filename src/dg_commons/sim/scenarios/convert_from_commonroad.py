from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle

from dg_commons import Color
from dg_commons.controllers.speed import SpeedController
from dg_commons.controllers.steer import SteerController
from dg_commons.sim import SimModel
from dg_commons.sim.agents.agent import Agent
from dg_commons.sim.agents.lane_follower import LFAgent
from dg_commons.sim.models.vehicle_dynamic import VehicleModelDyn
from dg_commons.sim.scenarios.model_from_dyn_obstacle import infer_model_from_cr_dyn_obstacle
from dg_commons.sim.scenarios.utils_dyn_obstacle import infer_lane_from_dyn_obs


def model_agent_from_dynamic_obstacle(
    dyn_obs: DynamicObstacle, lanelet_network: LaneletNetwork, color: Color = "royalblue"
) -> (VehicleModelDyn, Agent):
    """
    This function aims to create a non-playing character (fixed sequence of commands) in our simulation environment from
    a CommonRoad dynamic obstacle (fixed sequence of states).
    # fixme currently only cars are supported
    # fixme this function needs to be improved...
    :param color:
    :param dyn_obs:
    :param lanelet_network:
    :return:
    """

    model: SimModel = infer_model_from_cr_dyn_obstacle(dyn_obs, color)
    # Agent
    dglane = infer_lane_from_dyn_obs(dyn_obs=dyn_obs, network=lanelet_network)
    agent = LFAgent(dglane, model_params=model.model_params, model_geo=model.model_geometry)
    return model, agent
