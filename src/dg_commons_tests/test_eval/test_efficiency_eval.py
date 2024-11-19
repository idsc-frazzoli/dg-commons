import pickle

from dg_commons import PlayerName
from dg_commons.eval.efficiency import time_goal_lane_reached, distance_traveled
from dg_commons_tests import REPO_DIR


def test_efficiency_eval():
    logs = REPO_DIR / "src/dg_commons_tests/test_eval/logs"
    with open(logs / "lanelet_network.pickle", "rb") as file:
        lanelet_network = pickle.load(file)

    with open(logs / "log.pickle", "rb") as file:
        log = pickle.load(file)

    with open(logs / "missions.pickle", "rb") as file:
        missions = pickle.load(file)

    ego_name = PlayerName("Ego")
    ego_goal_lane = missions[ego_name]
    ego_states = log[ego_name].states
    time_to_reach = time_goal_lane_reached(lanelet_network, ego_goal_lane, ego_states)
    has_reached_the_goal = time_to_reach is not None
    dist = distance_traveled(ego_states)

    time_to_reach_expected = 3.25
    dist_expected = 67.29
    eps = 1e-2

    assert has_reached_the_goal
    assert time_to_reach_expected - eps <= time_to_reach <= time_to_reach_expected + eps
    assert dist_expected - eps <= dist <= dist_expected + eps
