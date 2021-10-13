from dataclasses import dataclass
from typing import Optional

from shapely.geometry import Polygon

from dg_commons import SE2Transform
from dg_commons.maps import DgLanelet

__all__ = ["PlanningGoal"]


@dataclass
class PlanningGoal:
    ref_lane: Optional[DgLanelet] = None
    goal_region: Optional[Polygon] = None
    goal_pose: Optional[SE2Transform] = None

    def __post_init__(self):
        if self.ref_lane:
            assert isinstance(self.ref_lane, DgLanelet)
        if self.goal_region:
            assert isinstance(self.goal_region, Polygon)
        if self.goal_pose:
            assert isinstance(self.goal_region, SE2Transform)
        # todo some consistency checks? on not having conflicting goals
