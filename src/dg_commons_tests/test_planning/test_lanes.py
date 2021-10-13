from typing import Hashable

import numpy as np
from matplotlib import pyplot as plt
from numpy import linspace

from dg_commons import SE2_apply_T2
from dg_commons.maps.lanes import DgLanelet
from sim.scenarios import load_commonroad_scenario


def test_lane_is_hashable():
    scenario, _ = load_commonroad_scenario("USA_Lanker-1_1_T-1")
    lanelet_net = scenario.lanelet_network
    for lanelet in lanelet_net.lanelets:
        lane = DgLanelet.from_commonroad_lanelet(lanelet)
        assert isinstance(lane, Hashable)


def test_lane_vis():
    scenario, _ = load_commonroad_scenario("USA_Lanker-1_1_T-1")
    lanelet_net = scenario.lanelet_network
    #    laneletid = lanelet_net.find_lanelet_by_position([np.array([test.x, test.y])])[0][0]

    for lanelet in lanelet_net.lanelets:
        lane = DgLanelet.from_commonroad_lanelet(lanelet)

        betas = linspace(-1, 5, 50).tolist()
        plt.figure()
        for beta in betas:
            q = lane.center_point(beta)
            radius = lane.radius(beta)
            delta_left = np.array([0, radius])
            delta_right = np.array([0, -radius])
            left = SE2_apply_T2(q, delta_left)
            right = SE2_apply_T2(q, delta_right)
            plt.plot(*left, "o")
            plt.plot(*right, "x")
            plt.gca().set_aspect("equal")
        plt.savefig(f"out/debug{lanelet.lanelet_id}.png")
        plt.close()
