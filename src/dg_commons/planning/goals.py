from abc import abstractmethod, ABC
from dataclasses import dataclass

import numpy as np
from geometry import translation_from_SE2
from shapely.geometry import Polygon, Point

from dg_commons import SE2Transform, X
from dg_commons.maps import DgLanelet
from dg_commons.sim.models import extract_pose_from_state

__all__ = ["PlanningGoal", "RefLaneGoal", "PolygonGoal", "PoseGoal"]


@dataclass
class PlanningGoal(ABC):
    @abstractmethod
    def is_fulfilled(self, state: X) -> bool:
        pass


@dataclass
class RefLaneGoal(PlanningGoal):
    ref_lane: DgLanelet
    goal_progress: float

    def is_fulfilled(self, state: X) -> bool:
        pose = extract_pose_from_state(state)
        return self.ref_lane.lane_pose_from_SE2_generic(pose).along_lane >= self.goal_progress


@dataclass
class PolygonGoal(PlanningGoal):
    goal: Polygon

    def is_fulfilled(self, state: X) -> bool:
        pose = extract_pose_from_state(state)
        xy = translation_from_SE2(pose)
        return self.goal.contains(Point(xy))


@dataclass
class PoseGoal(PlanningGoal):
    goal_pose: SE2Transform

    def is_fulfilled(self, state: X, tol: float = 1e-7) -> bool:
        pose = extract_pose_from_state(state)
        goal_pose = self.goal_pose.as_SE2()
        return np.linalg.norm(pose - goal_pose) <= tol
