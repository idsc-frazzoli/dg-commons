from dataclasses import dataclass
from math import sin, atan
from typing import Tuple, Union, List
import numpy as np
import scipy.optimize
from geometry import SE2value, translation_angle_from_SE2, SE2_from_translation_angle, angle_from_SE2
from duckietown_world.utils import SE2_apply_R2
from dg_commons.geo import norm_between_SE2value
from dg_commons import X
from dg_commons_dev.controllers.controller_types import LateralController
from dg_commons_dev.utils import BaseParams


__all__ = ["PurePursuit", "PurePursuitParam"]


@dataclass
class PurePursuitParam(BaseParams):
    look_ahead_minmax: Union[List[Tuple[float, float]], Tuple[float, float]] = (3, 30)
    """min and max lookahead"""
    k_lookahead: Union[List[float], float] = 1.8
    """Scaling constant for speed dependent params"""
    min_distance: Union[List[float], float] = 0.1
    """Min initial progress to look for the next goal point"""
    max_extra_distance: Union[List[float], float] = 5
    """Max extra distance to look for the closest point on the ref path"""
    length: Union[List[float], float] = 3.5
    """Length of the vehicle"""
    t_step: Union[List[float], float] = 0.1


class PurePursuit(LateralController):
    """ Pure Pursuit lateral controller """

    USE_STEERING_VELOCITY: bool = False

    def __init__(self, params: PurePursuitParam = PurePursuitParam()):
        self.params: PurePursuitParam = params
        self.speed: float = 0

        self.pose: SE2value
        self.along_path: float
        self.current_beta: float

        super().__init__()

    def _update_obs(self, new_obs: X):
        self.pose = SE2_from_translation_angle([new_obs.x, new_obs.y], new_obs.theta)

        control_sol_params = self.control_path.ControlSolParams(new_obs.vx, self.params.t_step)
        lanepose = self.control_path.lane_pose_from_SE2_generic(self.pose, control_sol=control_sol_params)
        self.along_path = lanepose.along_lane
        self.current_beta = self.path.beta_from_along_lane(self.along_path)

    def find_goal_point(self) -> Tuple[float, SE2value]:
        """
        Find goal point along the path
        """
        lookahead = self._get_lookahead()

        def goal_point_error(along_path: float) -> float:
            """
            returns error between desired distance from pose to point along path
            """
            beta = self.path.beta_from_along_lane(along_path)
            cp = self.path.center_point(beta)
            dist = norm_between_SE2value(self.pose, cp)
            return np.linalg.norm(dist - lookahead)

        min_along_path = self.along_path + self.params.min_distance

        bounds = [min_along_path, min_along_path + lookahead]
        res = scipy.optimize.minimize_scalar(fun=goal_point_error, bounds=bounds, method='Bounded')
        goal_point = self.path.center_point(self.path.beta_from_along_lane(res.x))
        return res.x, goal_point

    def _get_steering(self, at: float) -> float:
        if any([_ is None for _ in [self.pose, self.path]]):
            raise RuntimeError("Attempting to use PurePursuit before having set any observations or reference path")
        theta = angle_from_SE2(self.pose)
        rear_axle = SE2_apply_R2(self.pose, np.array([-self.params.length / 2, 0]))
        _, goal_point = self.find_goal_point()
        p_goal, theta_goal = translation_angle_from_SE2(goal_point)
        alpha = np.arctan2(p_goal[1] - rear_axle[1], p_goal[0] - rear_axle[0]) - theta
        radius = self._get_lookahead() / (2 * sin(alpha)) if alpha != 0 else 10e6
        return atan(self.params.length / radius)

    def _get_lookahead(self) -> float:
        """ Returns lookahead distance """
        return float(np.clip(self.params.k_lookahead * self.speed,
                             self.params.look_ahead_minmax[0],
                             self.params.look_ahead_minmax[1]))
