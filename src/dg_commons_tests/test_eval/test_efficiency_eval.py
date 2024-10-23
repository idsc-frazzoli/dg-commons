import pickle
from pathlib import Path
import os
import numpy as np
from dg_commons import PlayerName
from dg_commons.eval.efficiency import time_goal_lane_reached, distance_traveled


def test_efficiency_eval():
    file = open("src/dg_commons_tests/test_eval/logs/lanelet_network.pickle", 'rb')
    lanelet_network = pickle.load(file)
    file.close()
    file = open("src/dg_commons_tests/test_eval/logs/log.pickle", 'rb')
    log = pickle.load(file)
    file.close()
    file = open("src/dg_commons_tests/test_eval/logs/missions.pickle", 'rb')
    missions = pickle.load(file)
    file.close()

    ego_name = PlayerName("Ego")
    ego_goal_lane = missions[ego_name]
    ego_states = log[ego_name].states
    time_to_reach = time_goal_lane_reached(lanelet_network, ego_goal_lane, ego_states)
    has_reached_the_goal = time_to_reach is not None
    if has_reached_the_goal:
        print("Goal lane reached " + "at time " + str(time_to_reach))
    else:
        print("Goal lane not reached.")
    dist = distance_traveled(ego_states)
    print("Distance traveled: " + str(dist))


if __name__ == '__main__':
    test_efficiency_eval()
    