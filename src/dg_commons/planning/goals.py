from abc import abstractmethod, ABC
from dataclasses import dataclass

import numpy as np
from geometry import translation_from_SE2
from shapely.geometry import Polygon, Point
from shapely.geometry.base import BaseGeometry

from dg_commons import SE2Transform, X
from dg_commons.maps import DgLanelet
from dg_commons.sim.models import extract_pose_from_state

__all__ = ["PlanningGoal", "RefLaneGoal", "PolygonGoal", "PoseGoal"]


@dataclass(frozen=True)
class PlanningGoal(ABC):
    @abstractmethod
    def is_fulfilled(self, state: X) -> bool:
        pass

    @abstractmethod
    def get_plottable_geometry(self) -> BaseGeometry:
        # convert to use commonroad IDrawable
        pass


@dataclass(frozen=True)
class RefLaneGoal(PlanningGoal):
    ref_lane: DgLanelet
    goal_progress: float

    def is_fulfilled(self, state: X) -> bool:
        pose = extract_pose_from_state(state)
        return self.ref_lane.lane_pose_from_SE2_generic(pose).along_lane >= self.goal_progress

    def get_plottable_geometry(self) -> BaseGeometry:
        raise NotImplementedError


@dataclass(frozen=True)
class PolygonGoal(PlanningGoal):
    goal: Polygon

    def is_fulfilled(self, state: X) -> bool:
        pose = extract_pose_from_state(state)
        xy = translation_from_SE2(pose)
        return self.goal.contains(Point(xy))

    def get_plottable_geometry(self) -> Polygon:
        return self.goal


@dataclass(frozen=True)
class PoseGoal(PlanningGoal):
    goal_pose: SE2Transform

    def is_fulfilled(self, state: X, tol: float = 1e-7) -> bool:
        pose = extract_pose_from_state(state)
        goal_pose = self.goal_pose.as_SE2()
        return np.linalg.norm(pose - goal_pose) <= tol

    def get_plottable_geometry(self) -> BaseGeometry:
        # todo return a Polygon triangle
        raise NotImplementedError
