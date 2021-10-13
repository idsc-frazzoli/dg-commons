# import functions to read xml file and visualize commonroad objects
from math import pi

import matplotlib.pyplot as plt
import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer

# from commonroad_route_planner.route_planner import RoutePlanner
from sim.scenarios import load_commonroad_scenario
from sim.scenarios.agent_from_commonroad import model_agent_from_dynamic_obstacle


def test_commonroad_scenario_viz():
    # generate path of the file to be opened
    scenario_name = "USA_Lanker-1_1_T-1.xml"#"USA_Peach-1_1_T-1"

    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)
    scenario.translate_rotate(translation=np.array([0, 0]), angle=-pi / 2)
    rnd = MPRenderer(figsize=(20, 10))
    for dyn_obs in scenario.dynamic_obstacles:
        dyn_obs.draw(rnd)
    scenario.lanelet_network.draw(
        rnd,
        #    draw_params={"traffic_light": {"draw_traffic_lights": False}}
    )
    rnd.render()
    # plt.grid(True, "both", zorder=1000)
    plt.savefig(f"{scenario_name}.png", dpi=300)
    # write_default_params("../../sim_tests/scenarios_tests/default_params.json")


def test_npAgent_from_dynamic_obstacle():
    scenario = "USA_Lanker-1_1_T-1.xml"
    scenario, planning_problem_set = load_commonroad_scenario(scenario)
    dyn_obs = scenario.dynamic_obstacles[2]

    model, agent = model_agent_from_dynamic_obstacle(dyn_obs, lanelet_network=scenario.lanelet_network)

    print(agent, model)
