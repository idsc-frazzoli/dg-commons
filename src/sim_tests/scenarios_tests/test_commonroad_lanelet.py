import matplotlib.pyplot as plt
import numpy as np
from commonroad.visualization.mp_renderer import MPRenderer

from sim.scenarios.factory import get_scenario_commonroad_replica


def test_commonroad_lanelet():
    sim_context = get_scenario_commonroad_replica(
        scenario_name="USA_Lanker-1_1_T-1.xml")
    test = sim_context.models["P0"].get_state()
    lanelet_net = sim_context.scenario.lanelet_network
    laneletid = lanelet_net.find_lanelet_by_position([np.array([test.x, test.y])])[0][0]
    lanelet = lanelet_net.find_lanelet_by_id(laneletid)
    print(laneletid)
    print(lanelet)
    centerpoints = np.array(lanelet.center_vertices)
    rnd = MPRenderer(figsize=(20, 10))
    lanelet_net.draw(rnd)
    x, y = centerpoints.T
    rnd.render()
    rnd.ax.scatter(x, y, c="b", zorder=1000)
    plt.savefig("out/debug.png")

    lanelets_succ, _ = lanelet.all_lanelets_by_merging_successors_from_lanelet(lanelet, lanelet_net)
    rnd = MPRenderer(figsize=(20, 10))
    lanelet_net.draw(rnd)
    x, y = np.array(lanelets_succ[0].center_vertices).T
    rnd.render()
    rnd.ax.scatter(x, y, c="b", zorder=1000)
    plt.savefig("out/debug2.png")
