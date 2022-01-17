import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, List
import numpy as np
from geometry import (
    SE2value,
    translation_angle_scale_from_E2,
    translation_angle_from_SE2,
    T2value,
)
from scipy.optimize import minimize_scalar
from math import isclose
from dg_commons import relative_pose, SE2_apply_T2
from dg_commons.maps.lanes import DgLanelet, DgLanePose


class DgLaneletControl:
    """
    This class extends the DgLanelet search of the closest point on a path to a position for vehicle lateral and
    longitudinal control purposes
    """

    def __init__(self, path: DgLanelet):
        self.path = path
        self.previous_along_lane = None

    @dataclass
    class ControlSolParams:
        """ Parameters required by the search function to make a guess """

        current_v: float
        """ Current vehicle velocity """
        dt: float
        """ Time interval between two subsequent searches """
        safety_factor: float = 1.2
        """ The look ahead distance gets multiplied by this value in the end. 
        This makes the search more conservative """

    def lane_pose_from_SE2_generic(self, q: SE2value, tol: float = 1e-4,
                                   control_sol: Optional[ControlSolParams] = None) -> DgLanePose:
        """
        This function finds the DgLanePose based on car current pose q
        @param q: Current car pose
        @param tol: Tolerance in the search
        @param control_sol: The parameters for formulating a guess
        @return: Current lane pos
        """

        p, _, _ = translation_angle_scale_from_E2(q)

        beta, q0 = self.find_along_lane_closest_point(p, tol=tol, control_sol=control_sol)
        along_lane = self.path.along_lane_from_beta(beta)
        rel = relative_pose(q0, q)

        r, relative_heading, _ = translation_angle_scale_from_E2(rel)
        lateral = r[1]

        return self.path.lane_pose(along_lane=along_lane, relative_heading=relative_heading, lateral=lateral)

    def get_func(self, p: T2value) -> Callable[[float], float]:
        """
        Get cost function for closest point on lane
        @param p: current position
        @return: cost function
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

        return get_delta

    def find_along_lane_closest_point(self, p: T2value, tol: float = 1e-7,
                                      control_sol: Optional[ControlSolParams] = None) -> Tuple[float, SE2value]:
        """
        This function finds beta and closest pose on the lane based on car current pose p.
        If no ControlSolParams are passed, the DgLanelet search version is used.
        @param p: Current car pose
        @param tol: Tolerance in the search
        @param control_sol: The parameters for formulating a guess
        @return: current beta and current closest pose on path
        """
        bracket = (-1.0, len(self.path.control_points))
        if control_sol:
            beta0 = self._find_along_lane_closest_point_control(p, control_sol, tol)
        else:
            res0 = minimize_scalar(self.get_func(p), bracket=bracket, tol=tol)
            beta0 = res0.x
        q = self.path.center_point(beta0)
        self.previous_along_lane = self.path.along_lane_from_beta(beta0)
        return beta0, q

    def _find_along_lane_closest_point_control(self, p: T2value, params: ControlSolParams, tol: float = 1e-7) -> float:
        factor: float = 1
        neg_delta, pos_delta = -10, params.current_v * params.dt * params.safety_factor
        interval = (neg_delta, pos_delta)
        n_samples = int(len(self.path.control_points) * factor)
        return self.find_along_lane_initial_guess(p, self.previous_along_lane, n_samples, tol, interval)

    def find_along_lane_initial_guess(self, p: T2value, initial_guess: Optional[float], n_samples: int,
                                      tol: float = 1e-7, interval: Tuple[float, float] = (-5, 5))\
            -> float:
        func = self.get_func(p)
        interval = list(interval)
        n_control_points = len(self.path.control_points)

        bracket: List[float, float] = [-1, n_control_points]
        bracket_len_before: float = bracket[1] - bracket[0]
        if initial_guess:
            if initial_guess + interval[1] >= 0:
                bracket[0] = max(self.beta_from_along_lane(initial_guess + interval[0]), bracket[0])
                bracket[1] = min(self.beta_from_along_lane(initial_guess + interval[1]), bracket[1])
        bracket_len_after: float = bracket[1] - bracket[0]
        n_samples = max(10, int(n_samples * bracket_len_after / bracket_len_before))
        samples = np.linspace(bracket[0], bracket[1], n_samples)
        beta = 0
        cost = math.inf
        for sample in samples:
            c_cost = func(sample)
            if c_cost < cost:
                beta = sample
                cost = c_cost
        new_interval = bracket_len_after / (n_samples - 1)
        bracket = [beta - new_interval, beta + new_interval]
        bracket: Tuple[float, float] = (bracket[0], bracket[1])

        res0 = minimize_scalar(func, bracket=bracket, tol=tol)
        beta = res0.x
        return beta

    def beta_from_along_lane(self, along_lane: float) -> float:
        """Returns the progress along the lane (parametrized in control points)"""
        lengths = self.path.get_lane_lengths()
        x0 = along_lane
        n = len(self.path.control_points)
        S = sum(lengths)

        if x0 < 0:
            beta = x0
            return beta
        elif x0 > S:
            beta = (n - 1.0) + (x0 - S)
            return beta
        elif isclose(x0, S, abs_tol=1e-8):
            beta = n - 1.0
            return beta
        assert 0 <= x0 < S, (x0, S)

        cumulative = np.cumsum([0] + lengths)
        val = int(np.argmax(cumulative >= x0))

        if val == 0:
            start_x = cumulative[val]
        else:
            start_x = cumulative[val-1]
        end_x = cumulative[val]
        if start_x <= x0 <= end_x:
            beta = (val - 1) + (x0 - start_x) / lengths[val - 1]
            return beta

        assert False
