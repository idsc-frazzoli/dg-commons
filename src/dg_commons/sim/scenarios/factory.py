from typing import Optional, Dict

from commonroad.prediction.prediction import TrajectoryPrediction
from geometry import SE2_from_xytheta

from dg_commons.sim.models.obstacles import StaticObstacle

from dg_commons import PlayerName, apply_SE2_to_shapely_geo
from dg_commons.sim import logger, SimLog, SimParameters
from dg_commons.sim.scenarios import load_commonroad_scenario, NotSupportedConversion
from dg_commons.sim.scenarios.convert_from_commonroad import model_agent_from_dynamic_obstacle
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.scenarios.utils_dyn_obstacle import is_dyn_obstacle_static
from dg_commons.sim.simulator import SimContext

__all__ = ["get_scenario_commonroad_replica"]


def get_scenario_commonroad_replica(
    scenario_name: str,
    scenarios_dir: Optional[str] = None,
    sim_param: Optional[SimParameters] = None,
    ego_player: Optional[PlayerName] = None,
    seed: int = 0,
) -> SimContext:
    """
    This function load a CommonRoad scenario and tries to convert the dynamic obstacles into the Model/Agent paradigm
    used by the driving-game simulator.
    :param scenario_name:
    :param scenarios_dir:
    :param sim_param:
    :param ego_player:
    :return:
    """
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name, scenarios_dir)
    players, models = {}, {}
    static_obstacles: Dict[int, StaticObstacle] = {}

    for i, dyn_obs in enumerate(scenario.dynamic_obstacles):
        # todo try to see if it can be considered a static obstacle (e.g. parked cars)
        assert isinstance(dyn_obs.prediction, TrajectoryPrediction), "Only trajectory predictions are supported"
        if is_dyn_obstacle_static(dyn_obs):
            static_obstacles.update(
                {
                    dyn_obs.obstacle_id: StaticObstacle(
                        obstacle_type=dyn_obs.obstacle_type,
                        shape=apply_SE2_to_shapely_geo(
                            shapely_geometry=dyn_obs.obstacle_shape.shapely_object,
                            se2_value=SE2_from_xytheta(
                                [*dyn_obs.initial_state.position, dyn_obs.initial_state.orientation]
                            ),
                        ),
                    )
                }
            )
        else:
            try:
                p_name = PlayerName(f"P{i}")
                if p_name == ego_player:
                    p_name = PlayerName("Ego")
                    model, agent = model_agent_from_dynamic_obstacle(
                        dyn_obs, scenario.lanelet_network, color="firebrick"
                    )
                else:
                    model, agent = model_agent_from_dynamic_obstacle(dyn_obs, scenario.lanelet_network)

                players.update({p_name: agent})
                models.update({p_name: model})
            except NotSupportedConversion as e:
                logger.warn("Unable to convert CommonRoad dynamic obstacle due to " + e.args[0] + " skipping...")
    logger.info(f"Managed to load {len(players)}")
    for sobs in scenario.static_obstacles:
        static_obstacles.update(
            {
                sobs.obstacle_id: StaticObstacle(
                    obstacle_type=sobs.obstacle_type, shape=sobs.obstacle_shape.shapely_object
                )
            }
        )

    sim_param = SimParameters() if sim_param is None else sim_param
    return SimContext(
        dg_scenario=DgScenario(scenario, static_obstacles=static_obstacles),
        models=models,
        players=players,
        log=SimLog(),
        param=sim_param,
        seed=seed,
    )
