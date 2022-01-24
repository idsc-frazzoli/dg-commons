import math
from dg_commons_dev.planning.rrt_utils.dubins_path_planning import mod2pi
import numpy as np
from dg_commons_dev.controllers.controller_types import *
from dg_commons.maps.lanes import DgLanelet
from dg_commons import X
from dg_commons_dev.controllers.steering_controllers import *
from dg_commons_dev.behavior.replan import Replan, ReplanDescription
from typing import List
from geometry import translation_angle_from_SE2
from dg_commons.sim.scenarios.structures import StaticObstacle
import shapely.geometry as sg
from dg_commons_dev.planning.rrt_utils.utils import Node
from dg_commons_dev.planning.rrt_utils.sampling import PolygonBoundaries
from dg_commons.maps.lanes import LaneCtrPoint
from dg_commons import SE2Transform
from shapely.geometry import Point, LinearRing, Polygon
from dg_commons_dev.planning.rrt_utils.goal_region import GoalRegion
from shapely import ops
from dg_commons_dev.planning.planner_base import Planner
from dg_commons_dev.planning.rrt_dubin import RRTDubins, RRTDubinParams


@dataclass
class ReplannerParams(BaseParams):
    planner: type(Planner) = RRTDubins
    """ Planner for replanning """
    planner_params: BaseParams = RRTDubinParams()
    """ Planner parameters """
    get_back_in: Tuple[float, float] = (3, 30)
    """ 
    Get back in lane between two known obstacles if their distance along the long term path is greater than
     velocity * get_back_in[0] + get_back_in[1] 
    """
    entry_distance: Tuple[float, float] = (0.01, 15)
    """ 
    Distance between last obstacle before long term plan reconnection and reconnection point is
     velocity * entry_distance[0] + entry_distance[1] 
    """
    min_dist_from_obs: float = 2
    """ Minimal distance from obstacle to consider replanning """

    def __post_init__(self):
        assert isinstance(self.planner_params, self.planner.REF_PARAMS)
        assert 0 <= self.entry_distance[0] and 0 <= self.entry_distance[1]
        assert 0 <= self.get_back_in[0] and 0 <= self.get_back_in[1]
        assert self.entry_distance[0] < self.get_back_in[0] and self.entry_distance[1] < self.get_back_in[1]


class Replanner:
    """
    Class for managing replanning situations
    """
    REF_PARAMS: dataclass = ReplannerParams

    def __init__(self, params: ReplannerParams = ReplannerParams()):
        self.planner = params.planner(params.planner_params)
        self.params = params

    def replan(self, my_obs: X, t: float, situation: Replan, current_ref: Reference,
               lane_boundaries: List[StaticObstacle], min_distance: float) -> Optional[DgLanelet]:
        """
        Generate a new path based on current situation and on current long-term plan
        @param my_obs: Observation of my state
        @param situation: Description of replan situation
        @param current_ref: Current long term plan
        @param lane_boundaries: list of lane boundaries
        @param t: Current time instant
        @param min_distance: Minimal distance to keep from obstacles
        @return: New path or None if not found
        """
        lane: DgLanelet = current_ref.path
        replan_info: ReplanDescription = situation.infos()
        entry_points, exit_points = replan_info.entry_along_lane, replan_info.exit_along_lane

        if abs(current_ref.along_lane - entry_points[0]) < self.params.min_dist_from_obs:
            return None

        if my_obs.vx < 0:
            get_back_in_lane_dist: float = self.params.get_back_in[1]
        else:
            get_back_in_lane_dist: float = my_obs.vx * self.params.get_back_in[0] + self.params.get_back_in[1]

        n_points = len(entry_points)
        entry_along_lane = None
        for i in range(n_points - 1):
            delta_free = entry_points[i + 1] - exit_points[i]
            if get_back_in_lane_dist < delta_free:
                entry_along_lane = exit_points[i]

        if entry_along_lane is None:
            if exit_points[-1] is not None:
                entry_along_lane = exit_points[-1]
            else:
                return None

        entry_along_lane += self.entry_distance(my_obs)
        beta = lane.beta_from_along_lane(entry_along_lane)
        pos, ang = translation_angle_from_SE2(lane.center_point(beta))
        idx = min(math.ceil(beta) + 1, len(lane.control_points))

        rest = lane.control_points[idx:]
        start = Node(my_obs.x, my_obs.y, my_obs.theta)
        goal = GoalRegion(Node(pos[0], pos[1], ang), ang, 2, 0.5, 0.2)
        obs_list = [obs.shape for obs in situation.infos().obstacles]

        current_along_lane = current_ref.along_lane
        current_beta = math.floor(lane.beta_from_along_lane(current_along_lane))

        target_beta = math.ceil(beta)
        positions = []
        limits_max, limits_min = [], []
        for b in range(max(current_beta - 1, 0), min(target_beta + 1, len(lane.control_points))):
            control_point = lane.control_points[b]
            pos, ang = control_point.q.p, control_point.q.theta
            limits_max.append(mod2pi(ang + math.pi / 4)), limits_min.append(mod2pi(ang - math.pi / 4))
            positions.append((pos[0], pos[1]))
        poly: sg.Polygon = sg.LineString(positions).buffer(8)

        max_interval = 0
        angle_limits = (-math.pi, math.pi)
        for min_val in limits_min:
            for max_val in limits_max:
                res = max_val - min_val
                if res < 0:
                    res += 2 * math.pi
                if res > max_interval:
                    max_interval = res
                    min_val = min_val if min_val <= math.pi else min_val - 2 * math.pi
                    max_val = max_val if max_val <= math.pi else max_val - 2 * math.pi
                    angle_limits = (min_val, max_val)

        car_point: Point = Point((my_obs.x, my_obs.y))
        for lane_bound in lane_boundaries:
            lane_bound_shape = lane_bound.shape
            if isinstance(lane_bound_shape, LinearRing):
                temp_poly = sg.Polygon(lane_bound_shape)
                lane_bound_shape = temp_poly.boundary

            poly_splits: List[sg.Polygon] = ops.split(poly, lane_bound_shape).geoms
            for poly_split in poly_splits:
                if poly_split.contains(car_point):
                    poly = poly_split
                    break
        rand_area = PolygonBoundaries(poly)

        path = self.planner.planning(start=start, goal=goal, obstacle_list=obs_list,
                                     sampling_bounds=rand_area, limit_angles=angle_limits, min_distance=min_distance)
        if path is None:
            return None
        self.planner.plot_results(False)

        se2s = [SE2Transform(np.array([step.x, step.y]), step.yaw) for step in path]
        r = self.planner.get_width()
        lane_points = [LaneCtrPoint(val, r) for val in se2s] + rest
        return DgLanelet(lane_points)

    def entry_distance(self, my_obs):
        if my_obs.vx < 0:
            return self.params.entry_distance[1]
        else:
            return my_obs.vx * self.params.entry_distance[0] + self.params.entry_distance[1]

    def replan_to(self, my_obs: X, t: float, obs_list: List[StaticObstacle], lane_boundaries: List[StaticObstacle],
                  current_ref: Tuple[float, float, float], min_distance: float,
                  tol: Tuple[float, float, float] = (0, 0, 0)) -> Tuple[Optional[DgLanelet], float]:
        """
        Generate a new path based on current situation and on current long-term plan
        @param my_obs: Observation of my state
        @param obs_list: List of obstacles
        @param current_ref: Current long term plan
        @param lane_boundaries: list of lane boundaries
        @param t: Current time instant
        @param min_distance: Minimal distance to keep from obstacles
        @param tol: Tolerance in path finding
        @return: New path or None if not found
        """
        obs_list = [obs.shape for obs in obs_list]
        start = Node(my_obs.x, my_obs.y, my_obs.theta)
        goal = GoalRegion(Node(current_ref[0], current_ref[1], current_ref[2]), current_ref[2], *tol)

        dx, dy = current_ref[0] - my_obs.x, current_ref[1] - my_obs.y
        line: sg.LineString = sg.LineString([(my_obs.x, my_obs.y), (current_ref[0], current_ref[1])])
        ang: float = math.atan2(dy, dx)
        dist: float = np.linalg.norm(np.array([dx, dy]))
        if dist < 10e-6:
            return None, dist

        poly = line.buffer(dist)
        car_point: Point = Point((my_obs.x, my_obs.y))
        for lane_bound in lane_boundaries:
            lane_bound_shape = lane_bound.shape
            if isinstance(lane_bound_shape, LinearRing):
                temp_poly = sg.Polygon(lane_bound_shape)
                lane_bound_shape = temp_poly.boundary

            poly_splits: List[sg.Polygon] = ops.split(poly, lane_bound_shape).geoms
            for poly_split in poly_splits:
                if poly_split.contains(car_point):
                    poly = poly_split
                    break

        rand_area = PolygonBoundaries(poly)
        angle_limits = (ang - math.pi / 4, ang + math.pi / 4)

        path = self.planner.planning(start=start, goal=goal, obstacle_list=obs_list,
                                     sampling_bounds=rand_area, limit_angles=angle_limits, min_distance=min_distance)
        if path is None:
            return None, dist
        self.planner.plot_results(False)

        se2s = [SE2Transform(np.array([step.x, step.y]), step.yaw) for step in path]
        r = self.planner.get_width()
        lane_points = [LaneCtrPoint(val, r) for val in se2s]
        return DgLanelet(lane_points), dist
