from typing import Optional

from dg_commons import PlayerName
from dg_commons.sim import logger, SimLog, SimParameters
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.agent_from_commonroad import model_agent_from_dynamic_obstacle, NotSupportedConversion
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import SimContext


def get_scenario_commonroad_replica(
    scenario_name: str,
    scenarios_dir: Optional[str] = None,
    sim_param: Optional[SimParameters] = None,
    ego_player: Optional[PlayerName] = None,
) -> SimContext:
    """
    This functions load a commonroad scenario and tries to convert the dynamic obstacles into the Model/Agent paradigm
    used by the driving-game simulator.
    :param scenario_name:
    :param scenarios_dir:
    :param sim_param:
    :param ego_player:
    :return:
    """
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name, scenarios_dir)
    players, models = {}, {}
    for i, dyn_obs in enumerate(scenario.dynamic_obstacles):
        try:
            playername = PlayerName(f"P{i}")
            if playername == ego_player:
                playername = PlayerName("Ego")
                model, agent = model_agent_from_dynamic_obstacle(dyn_obs, scenario.lanelet_network, color="firebrick")
            else:
                model, agent = model_agent_from_dynamic_obstacle(dyn_obs, scenario.lanelet_network)

            players.update({playername: agent})
            models.update({playername: model})
        except NotSupportedConversion as e:
            logger.warn("Unable to convert commonroad dynamic obstacle due to " + e.args[0] + " skipping...")
    logger.info(f"Managed to load {len(players)}")
    sim_param = SimParameters() if sim_param is None else sim_param
    return SimContext(
        dg_scenario=DgScenario(scenario),
        models=models,
        players=players,
        log=SimLog(),
        param=sim_param,
    )
