from dataclasses import dataclass
from math import atan
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn
import numpy as np
from geometry import SE2value, SE2_from_translation_angle, translation_angle_scale_from_E2, \
    translation_angle_from_SE2, T2value
from dg_commons import X
from duckietown_world.utils import SE2_apply_R2
import math
from duckietown_world import relative_pose
from dg_commons_dev.curve_approximation_techniques import LinearCurve, CurveApproximationTechniques
from typing import Tuple, Callable
from dg_commons_dev.controllers.controller_types import LateralController
from dg_commons_dev.utils import BaseParams


__all__ = ["Stanley", "StanleyParam"]


@dataclass
class StanleyParam(BaseParams):
    stanley_gain: float = 1
    """ Tunable gain """
    t_step: float = 0.1
    """ Time between two controller calls """

    def __post_init__(self):
        assert 0 <= self.stanley_gain
        assert 0 < self.t_step <= 30


class Stanley(LateralController):
    """ Stanley lateral controller """

    USE_STEERING_VELOCITY: bool = False
    """ 
    Whether the returned steering is the desired steering velocity or the desired steering angle 
    True: steering velocity
    False: steering angle
    """
    REF_PARAMS: Callable = StanleyParam

    def __init__(self, params: StanleyParam = StanleyParam(),
                 target_position: T2value = None):
        self.params: StanleyParam = params
        self.vehicle_geometry: VehicleGeometry = VehicleGeometry.default_car()
        self.path_approx: CurveApproximationTechniques = LinearCurve()

        self.front_pose: SE2value
        self.speed: float
        self.alpha: float
        self.current_beta: float
        self.lateral: float
        self.target_position: T2value = target_position

        super().__init__()

    def _update_obs(self, new_obs: X) -> None:
        """
        A new observation is processed and an input for the system formulated
        @param new_obs: New Observation
        """
        tr, ang = [new_obs.x, new_obs.y], new_obs.theta
        pose = SE2_from_translation_angle(tr, ang)

        front_position = SE2_apply_R2(pose, np.array([self.vehicle_geometry.lf, 0]))
        front_pose = SE2_from_translation_angle(front_position, ang)

        front_speed = False
        if front_speed:
            if X == VehicleStateDyn:
                front_speed = np.array(new_obs.vx, new_obs.vy) + new_obs.dtheta*np.array(0, self.vehicle_geometry.lf)
                self.speed = np.linalg.norm(front_speed)
            else:
                self.speed = new_obs.vx/math.cos(new_obs.delta)
        else:
            self.speed = new_obs.vx

        p, _, _ = translation_angle_scale_from_E2(front_pose)

        control_sol_params = self.control_path.ControlSolParams(new_obs.vx, self.params.t_step)
        self.current_beta, q0 = self.control_path.find_along_lane_closest_point(p, tol=1e-4,
                                                                                control_sol=control_sol_params)
        path_approx = True
        if path_approx:
            pos1, angle1, pos2, angle2, pos3, angle3 = self.next_pos(self.current_beta)
            self.path_approx.update_from_data(pos1, angle1, pos2, angle2, pos3, angle3)
            res = self.path_approx.parameters
            closest_point_func = self.path_approx.closest_point_on_path

            angle = res[2]

            self.alpha = angle - new_obs.theta
            if self.alpha > math.pi:
                self.alpha = -(2*math.pi - self.alpha)
            elif self.alpha < - math.pi:
                self.alpha = 2*math.pi + self.alpha

            closest_point = closest_point_func(front_position)
            self.lateral = - (closest_point[0] - front_position[0]) * math.sin(new_obs.theta) + \
                             (closest_point[1] - front_position[1]) * math.cos(new_obs.theta)

        else:
            rel = relative_pose(front_pose, q0)

            r, self.alpha, _ = translation_angle_scale_from_E2(rel)
            self.lateral = r[1]

    def next_pos(self, current_beta) -> Tuple[np.ndarray, float, np.ndarray, float, np.ndarray, float]:
        """
        @param current_beta: current parametrized location on path
        @return: Three positions and three angles ahead of the current position on the path
        """

        along_lane = self.path.along_lane_from_beta(current_beta)
        k = 10
        delta_step = self.speed * 0.1 * k
        along_lane1 = along_lane + delta_step / 2
        along_lane2 = along_lane1 + delta_step / 2

        beta1, beta2, beta3 = current_beta, self.path.beta_from_along_lane(along_lane1), \
                              self.path.beta_from_along_lane(along_lane2)

        q1 = self.path.center_point(beta1)
        q2 = self.path.center_point(beta2)
        q3 = self.path.center_point(beta3)

        pos1, angle1 = translation_angle_from_SE2(q1)
        pos2, angle2 = translation_angle_from_SE2(q2)
        pos3, angle3 = translation_angle_from_SE2(q3)

        self.target_position = pos3
        return pos1, angle1, pos2, angle2, pos3, angle3

    def _get_steering(self, at: float) -> float:
        """
        @param at: current time instant
        @return: steering to command
        """
        if any([_ is None for _ in [self.alpha, self.lateral, self.speed]]):
            raise RuntimeError("Attempting to use Stanley before having set any observations or reference path")

        return self.alpha + atan(self.params.stanley_gain*self.lateral/self.speed)
