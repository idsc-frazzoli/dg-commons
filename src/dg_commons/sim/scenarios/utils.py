import os
import re
from typing import Tuple, Optional

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.scenario.scenario import Scenario
from geometry import T2value
from zuper_commons.types import ZException

from dg_commons.maps import DgLanelet

__all__ = ["load_commonroad_scenario", "dglane_from_position", "NotSupportedConversion"]


class NotSupportedConversion(ZException):
    pass


def load_commonroad_scenario(
    scenario_name: str, scenarios_dir: Optional[str] = None
) -> Tuple[Scenario, PlanningProblemSet]:
    """Loads a CommonRoad scenario.
    If no directory is provided it looks for a `scenarios` folder at the src level of the current project."""
    if scenarios_dir is None:
        dg_root_dir = __file__
        src_folder = "src"
        assert src_folder in dg_root_dir, dg_root_dir
        dg_root_dir = re.split(src_folder, dg_root_dir)[0]
        assert os.path.isdir(dg_root_dir)
        scenarios_dir = os.path.join(dg_root_dir, "scenarios")

    # generate path of the file to be opened
    scenario_name = scenario_name if scenario_name.endswith(".xml") else scenario_name + ".xml"
    scenario_path = None
    for root, dirs, files in os.walk(scenarios_dir, followlinks=True):
        for name in files:
            if name == scenario_name:
                scenario_path = os.path.abspath(os.path.join(root, name))
                break
    if scenario_path is None:
        raise FileNotFoundError(
            f"Unable to find commonroad scenario {scenario_name} within {scenarios_dir}.\n"
            f"Be aware that currently interactive scenarios cannot be loaded."
        )
    # read in the scenario and planning problem set
    return CommonRoadFileReader(scenario_path).open(lanelet_assignment=True)


def dglane_from_position(
    p: T2value, network: LaneletNetwork, init_lane_selection: int = 0, succ_lane_selection: int = 0
) -> DgLanelet:
    """Gets the merged lane from init lane select to the successive lane from the current position"""
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
