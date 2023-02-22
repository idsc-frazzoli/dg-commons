import os.path
from math import isclose
from typing import Hashable

import numpy as np
from geometry import translation_angle_from_SE2
from matplotlib import pyplot as plt
from numpy import linspace

from dg_commons import SE2_apply_T2
from dg_commons.maps.lanes import DgLanelet
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons_tests import OUT_TESTS_DIR


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
        file_name = os.path.join(OUT_TESTS_DIR, f"lane_vis/{lanelet.lanelet_id}.png")
        # create directory if not exists
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        plt.savefig(file_name)
        plt.close()


def test_find_along_lane_closest_point():
    scenario, _ = load_commonroad_scenario("USA_Lanker-1_1_T-1")
    lanelet_net = scenario.lanelet_network
    #    laneletid = lanelet_net.find_lanelet_by_position([np.array([test.x, test.y])])[0][0]
    for lanelet in lanelet_net.lanelets:
        lane = DgLanelet.from_commonroad_lanelet(lanelet)

        betas = linspace(0, 5, 10).tolist()
        plt.figure()
        for beta in betas:
            q = lane.center_point(beta)
            t, _ = translation_angle_from_SE2(q)
            q_fast = lane.center_point_fast_SE2Transform(beta).as_SE2()
            t_fast, _ = translation_angle_from_SE2(q_fast)

            assert isclose(0, np.linalg.norm(t - t_fast), abs_tol=1)

            plt.plot(*t, "o")
            plt.plot(*t_fast, "x")
            plt.gca().set_aspect("equal")
        file_name = os.path.join(OUT_TESTS_DIR, f"center_point/{lanelet.lanelet_id}.png")
        # create directory if not exists
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        plt.savefig(file_name)
        plt.close()
