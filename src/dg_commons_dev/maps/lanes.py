import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from geometry import (
    SE2value,
    translation_angle_scale_from_E2,
    translation_angle_from_SE2,
    T2value,
)
from scipy.optimize import minimize_scalar

from dg_commons import relative_pose, SE2_apply_T2
from dg_commons.maps.lanes import DgLanelet, DgLanePose


class DgLaneletControl:
    """ This class extends the search of the closest position on the lane from DgLanelet for vehicle
    lateral control purposes """

    def __init__(self, path: DgLanelet):
        self.path = path
        self.previous_along_lane = None

    @dataclass
    class ControlSolParams:
        """ Parameters required by the search function to make a guess """
        current_v: float
        dt: float
        safety_factor: float = 1.2

    def lane_pose_from_SE2_generic(self, q: SE2value, tol: float = 1e-4,
                                   control_sol: Optional[ControlSolParams] = None) -> DgLanePose:
        """ This function finds the DgLanePose based on car current pose q """
        p, _, _ = translation_angle_scale_from_E2(q)

        beta, q0 = self.find_along_lane_closest_point(p, tol=tol, control_sol=control_sol)
        along_lane = self.path.along_lane_from_beta(beta)
        rel = relative_pose(q0, q)

        r, relative_heading, _ = translation_angle_scale_from_E2(rel)
        lateral = r[1]

        return self.path.lane_pose(along_lane=along_lane, relative_heading=relative_heading, lateral=lateral)

    def find_along_lane_closest_point(self, p: T2value, tol: float = 1e-7,
                                      control_sol: Optional[ControlSolParams] = None) -> Tuple[float, SE2value]:
        """
        This function finds beta and closest pose on the lane based on car current pose p.
        If no ControlSolParams are passed, the DgLanelet version is used.
        """
        def get_delta(beta):
            q0 = self.path.center_point(beta)
            t0, _ = translation_angle_from_SE2(q0)
            d = np.linalg.norm(p - t0)

            d1 = np.array([0, -d])
            p1 = SE2_apply_T2(q0, d1)

            d2 = np.array([0, +d])
            p2 = SE2_apply_T2(q0, d2)

            D2 = np.linalg.norm(p2 - p)
            D1 = np.linalg.norm(p1 - p)
            res = np.maximum(D1, D2)
            return res

        bracket = (-1.0, len(self.path.control_points))
        if control_sol:
            beta0 = self._find_along_lane_closest_point_control(bracket, get_delta, control_sol, tol)
        else:
            res0 = minimize_scalar(get_delta, bracket=bracket, tol=tol)
            beta0 = res0.x

        q = self.path.center_point(beta0)
        self.previous_along_lane = self.path.along_lane_from_beta(beta0)
        return beta0, q

    def _find_along_lane_closest_point_control(self, bracket, func, params: ControlSolParams, tol: float = 1e-7) \
            -> float:
        factor = 1
        bracket = list(bracket)
        n_samples = int(len(self.path.control_points) * factor)
        bracket_len_b = bracket[1] - bracket[0]

        use_guess = True
        if use_guess and self.previous_along_lane:
            neg_delta, pos_delta = -10, params.current_v*params.dt*params.safety_factor
            if self.previous_along_lane + pos_delta >= 0:
                bracket[0] = max(self.path.beta_from_along_lane(self.previous_along_lane+neg_delta), bracket[0])
                bracket[1] = min(self.path.beta_from_along_lane(self.previous_along_lane+pos_delta), bracket[1])

        bracket_len = bracket[1]-bracket[0]
        n_samples = max(2, int(n_samples * bracket_len / bracket_len_b))
        samples = np.linspace(bracket[0], bracket[1], n_samples)
        beta = 0
        cost = math.inf

        for sample in samples:
            c_cost = func(sample)
            if c_cost < cost:
                beta = sample
                cost = c_cost

        interval = bracket_len / (n_samples - 1)
        bracket = (beta - interval, beta + interval)
        bracket = (bracket[0], bracket[1])

        res0 = minimize_scalar(func, bracket=bracket, tol=tol)
        beta = res0.x
        return beta
