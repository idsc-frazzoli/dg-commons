from dataclasses import dataclass
from math import sin, atan
from typing import Optional, Tuple

import numpy as np
import scipy.optimize
from geometry import SE2value, translation_angle_from_SE2, angle_from_SE2

from dg_commons.geo import norm_between_SE2value, SE2_apply_T2
from dg_commons.maps.lanes import DgLanelet

__all__ = ["PurePursuit", "PurePursuitParam"]


@dataclass
class PurePursuitParam:
    look_ahead_minmax: Tuple[float, float] = (3, 30)
    """min and max lookahead"""
    k_lookahead: float = 1.1
    """Scaling constant for speed dependent params"""
    min_distance: float = 2
    """Min initial progress to look for the next goal point"""
    max_extra_distance: float = 20
    """Max extra distance to look for the closest point on the ref path"""
    length: float = 3.5
    """Length of the vehicle"""


class PurePursuit:
    """
    https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/amod
    /AMOD_2020/20201019-05%20-%20ETHZ%20-%20Control%20in%20Duckietown%20(PID).pdf
    Note there is an error in computation of alpha (order needs to be inverted)
    """

    def __init__(self, params: PurePursuitParam = PurePursuitParam()):
        """
        initialise pure_pursuit control loop
        :param
        """
        self.path: Optional[DgLanelet] = None
        self.pose: Optional[SE2value] = None
        self.along_path: Optional[float] = None
        self.speed: float = 0
        self.param: PurePursuitParam = params
        # logger.debug("Pure pursuit params: \n", self.param)

    def update_path(self, path: DgLanelet):
        assert isinstance(path, DgLanelet)
        self.path = path

    def update_pose(self, pose: SE2value, along_path: float):
        assert isinstance(pose, SE2value)
        assert isinstance(along_path, float)
        self.pose = pose
        self.along_path = along_path

    def update_speed(self, speed: float):
        self.speed = speed

    def find_goal_point(self) -> Tuple[float, SE2value]:
        """
        Find goal point along the path
        :return: along_path, SE2value
        """
        lookahead = self._get_lookahead()

        def goal_point_error(along_path: float) -> float:
            """
            :param along_path:
            :return: Error between desired distance from pose to point along path
            """
            beta = self.path.beta_from_along_lane(along_path)
            cp = self.path.center_point(beta)
            dist = norm_between_SE2value(self.pose, cp)
            return np.linalg.norm(dist - lookahead)

        min_along_path = self.along_path + self.param.min_distance

        bounds = [min_along_path, min_along_path + lookahead]
        res = scipy.optimize.minimize_scalar(fun=goal_point_error, bounds=bounds, method="Bounded")
        goal_point = self.path.center_point(self.path.beta_from_along_lane(res.x))
        return res.x, goal_point

    def get_desired_steering(self) -> float:
        """
        :return: float the desired wheel angle
        """
        if any([_ is None for _ in [self.pose, self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")
        theta = angle_from_SE2(self.pose)
        rear_axle = SE2_apply_T2(self.pose, np.array([-self.param.length / 2, 0]))
        _, goal_point = self.find_goal_point()
        p_goal, theta_goal = translation_angle_from_SE2(goal_point)
        alpha = np.arctan2(p_goal[1] - rear_axle[1], p_goal[0] - rear_axle[0]) - theta
        radius = self._get_lookahead() / (2 * sin(alpha))
        return atan(self.param.length / radius)

    def _get_lookahead(self) -> float:
        return float(
            np.clip(
                self.param.k_lookahead * self.speed, self.param.look_ahead_minmax[0], self.param.look_ahead_minmax[1]
            )
        )
