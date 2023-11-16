from abc import abstractmethod, ABC
from dataclasses import dataclass
from functools import cached_property
from typing import TypeVar, Optional

import numpy as np
from geometry import translation_from_SE2
from shapely.geometry import Polygon, Point, LineString
from shapely.geometry.base import BaseGeometry

from dg_commons import SE2Transform, X, SE2_apply_T2, apply_SE2_to_shapely_geo
from dg_commons.maps import DgLanelet
from dg_commons.sim import SimTime
from dg_commons.sim.models import extract_pose_from_state

__all__ = ["PlanningGoal", "TPlanningGoal", "RefLaneGoal", "PolygonGoal", "PoseGoal"]


@dataclass(frozen=True)
class PlanningGoal(ABC):
    @abstractmethod
    def is_fulfilled(self, state: X, at: SimTime = 0) -> bool:
        pass

    @abstractmethod
    def get_plottable_geometry(self) -> BaseGeometry:
        # convert to use commonroad IDrawable
        pass


# from 3.11 can switch to Self
TPlanningGoal = TypeVar("TPlanningGoal", bound=PlanningGoal)


@dataclass(frozen=True)
class RefLaneGoal(PlanningGoal):
    ref_lane: DgLanelet
    goal_progress: float
    """Parametrized in along_lane [meters], need to convert from beta if using control points parametrization"""

    def is_fulfilled(self, state: X, at: SimTime = 0) -> bool:
        pose = extract_pose_from_state(state)
        xy = translation_from_SE2(pose)
        if self.goal_polygon.contains(Point(xy)):
            beyond_goal = self.ref_lane.lane_pose_from_SE2_generic(pose).along_lane >= self.goal_progress
            return beyond_goal
        else:
            return False

    def get_plottable_geometry(self) -> BaseGeometry:
        return self.goal_polygon

    @cached_property
    def goal_polygon(self) -> Polygon:
        poly = _polygon_at_along_lane(self.ref_lane, self.goal_progress)
        return poly


def _polygon_at_along_lane(
    lanelet: DgLanelet,
    at_beta: Optional[float] = None,
    inflation_radius: float = 1,
) -> Polygon:
    maxbeta = at_beta if at_beta is not None else len(lanelet.control_points)
    q = lanelet.center_point(maxbeta)
    r = lanelet.radius(maxbeta)
    delta_left = np.array([0, r])
    delta_right = np.array([0, -r])
    point_left = SE2_apply_T2(q, delta_left)
    point_right = SE2_apply_T2(q, delta_right)
    coords = [point_left.tolist(), point_right.tolist()]
    end_goal_poly = LineString(coords).buffer(inflation_radius)
    return end_goal_poly


@dataclass(frozen=True)
class PolygonGoal(PlanningGoal):
    goal: Polygon

    @classmethod
    def from_DgLanelet(cls, lanelet: DgLanelet, inflation_radius: float = 1) -> "PolygonGoal":
        end_goal_segment = _polygon_at_along_lane(lanelet, inflation_radius=inflation_radius)
        return cls(end_goal_segment)

    def is_fulfilled(self, state: X, at: SimTime = 0) -> bool:
        pose = extract_pose_from_state(state)
        xy = translation_from_SE2(pose)
        return self.goal.contains(Point(xy))

    def get_plottable_geometry(self) -> Polygon:
        return self.goal


@dataclass(frozen=True)
class PoseGoal(PlanningGoal):
    goal_pose: SE2Transform

    def is_fulfilled(self, state: X, at: SimTime = 0, tol: float = 1e-7) -> bool:
        pose = extract_pose_from_state(state)
        goal_pose = self.goal_pose.as_SE2()
        return np.linalg.norm(pose - goal_pose) <= tol

    def get_plottable_geometry(self) -> BaseGeometry:
        raise self.goal_pose

    @cached_property
    def goal_pose(self) -> Polygon:
        goal_shape = Polygon([(-0.2, 0.5), (0, 0), (-0.2, -0.5), (0.8, 0)])
        goal = apply_SE2_to_shapely_geo(goal_shape, self.goal_pose.as_SE2())
        return goal
