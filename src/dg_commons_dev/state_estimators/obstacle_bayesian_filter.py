import itertools
import math

import numpy as np
from geometry import SE2_from_translation_angle, rot2d_from_angle
from dg_commons.sim.models.vehicle import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from typing import Tuple, List
from dataclasses import dataclass
from dg_commons_dev.utils import BaseParams
from dg_commons_dev.state_estimators.estimator_types import Estimator
from dg_commons_dev.state_estimators.utils import Poisson, PoissonParams, CircularGrid
from dg_commons_dev.state_estimators.temp_curve import PCurve
from dg_commons.geo import SE2_apply_T2
from dg_commons import X
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from scipy.ndimage import measurements
from shapely.geometry import Point, Polygon, MultiPoint, LineString, shape
from shapely.affinity import affine_transform


@dataclass
class ObstacleBayesianParam(BaseParams):
    ratio_area_area_obs: float = 0.01
    """ Expected ratio total area to area occupied by obstacles"""

    fp_distribution: PCurve = PCurve(0.05)
    """ FN distribution """
    fn_distribution: PCurve = PCurve(0.05)
    """ FP distribution """
    acc_distribution: PCurve = PCurve(0.05)
    """ Acc distribution """

    grid_radius: float = 10
    """ Radius of circular grid considered """
    n_points: int = 10
    """ Number of points on a line """
    n_lines: int = 10
    """ Number of lines """
    distance_to_first_ring: float = 2
    """ Distance from center of lidar to first detection ring """

    geometry_params: VehicleGeometry = VehicleGeometry.default_car()
    """ Vehicle Geometry """
    vehicle_params: VehicleParameters = VehicleParameters.default_car()
    """ Vehicle Parameters """
    t_step: float = 0.1
    """ Time interval between two calls """


class ObstacleBayesian(Estimator):
    """ Bayesian estimator for presence of obstacles """
    REF_PARAMS: dataclass = ObstacleBayesianParam

    def __init__(self, params: ObstacleBayesianParam, generate_lidar_plot: bool = False):
        self.params: ObstacleBayesianParam = params
        self.generate_moving_dist: bool = generate_lidar_plot

        self.current_belief = CircularGrid(self.params.grid_radius, self.params.n_points,
                                           self.params.n_lines, self.params.distance_to_first_ring)
        self.shape = self.current_belief.values.shape
        self.fn, self.fp, self.acc = np.zeros(self.shape), np.zeros(self.shape), np.zeros(self.shape)
        self.prior_p = np.zeros(self.shape)
        for idx, position in self.current_belief.gen_structure():
            self.fn[idx] = self.params.fn_distribution.evaluate_distribution(np.array(position))
            self.fp[idx] = self.params.fp_distribution.evaluate_distribution(np.array(position))
            self.acc[idx] = self.params.acc_distribution.evaluate_distribution(np.array(position))
            self.prior_p[idx] = 1 - Poisson(
                PoissonParams(lamb=self.current_belief.areas[idx] * self.params.ratio_area_area_obs)
                ).pmf(0)

        self.prior_p_ext = self.prior_p[0, -1]

        self.current_belief.values = self.prior_p
        self.pos_x = self.current_belief.pos_x
        self.pos_y = self.current_belief.pos_y

        acc_areas = 4 * np.power(self.acc, 2)
        acc_areas = np.where(acc_areas > 0, acc_areas, 10e-8)
        self.p_detection_given_origin = np.where(acc_areas > 0, np.divide(1, acc_areas), 0)

        self.data = []
        self.poses = []
        self.previous_state: X = None

    def update_prediction(self, u_k: X) -> None:
        """
        Internal current belief get projected ahead by params.dt using the kinematic bicycle model
        and the input to the system u_k.
        @param u_k: Vehicle input k
        @return: None
        """
        if self.previous_state is None:
            self.previous_state = u_k
            return
        delta_x, delta_y = u_k.x - self.previous_state.x, u_k.y - self.previous_state.y
        delta_pos = np.matmul(rot2d_from_angle(-self.previous_state.theta), np.array([delta_x, delta_y]))
        delta_theta = u_k.theta - self.previous_state.theta
        delta_se2 = SE2_from_translation_angle(delta_pos, delta_theta)
        self.previous_state = u_k
        results = self.current_belief.values_by_positions(self.current_belief.apply_se2(delta_se2),
                                                          if_nan=self.prior_p_ext)
        results = np.where(np.isnan(results), self.prior_p_ext, results)
        res = np.reshape(results, self.current_belief.values.shape)
        self.current_belief.values = res

    def update_measurement(self, detections: List[Tuple[float, float]], max_dependency_dist: float = 0) -> None:
        """
        For each grid point: update obstacles probability distribution based on
        FN/FP/Accuracy curves and the provided detections
        @param detections: x- and y-position of the detections
        @param max_dependency_dist: dependency between detections that are more distant than this parameter is discarded
        """
        current_b = self.current_belief.as_numpy()

        mat = self.p_did_not_cause_any_given_d_only(detections, max_dependency_dist)
        num = np.multiply(1-self.fn, current_b)
        den = num + np.multiply(self.fp, 1-current_b)
        p_obs_given_causing = np.divide(num, den)
        num = np.multiply(self.fn, current_b)
        den = num + np.multiply(1-self.fp, 1 - current_b)
        p_obs_given_not_causing = np.divide(num, den)

        result = np.multiply(p_obs_given_causing, 1-mat) + np.multiply(p_obs_given_not_causing, mat)
        self.current_belief.values = result
        self.current_belief.update_regular_grid()

        if self.generate_moving_dist:
            self.data.append((result, SE2_from_translation_angle(
                np.array([self.previous_state.x, self.previous_state.y]), self.previous_state.theta)))

    def p_did_not_cause_any_given_d_only(self, detections: List[Tuple[float, float]], max_dependency_dist: float = 0)\
            -> np.ndarray:
        """
        For each grid point: compute probability that it did not cause any of the detections
        @param detections: x- and y-position of the detections
        @param max_dependency_dist: dependency between detections that are more distant than this parameter is discarded
        @return: array with the previously described probabilities at each grid point
        """
        involvement = {}
        candidates = []
        independent: bool = max_dependency_dist == 0
        res = np.ones(self.shape)
        for detection in detections:
            in_accuracy: np.ndarray = np.where((self.pos_x - self.acc <= detection[0]) &
                                               (detection[0] <= self.pos_x + self.acc) &
                                               (self.pos_y - self.acc <= detection[1]) &
                                               (detection[1] <= self.pos_y + self.acc),
                                               self.p_detection_given_origin, 0)

            sum_acc_prob: float = float(np.sum(in_accuracy))
            in_accuracy = in_accuracy / sum_acc_prob if sum_acc_prob != 0 else in_accuracy

            if independent:
                res = np.multiply(res, 1 - in_accuracy)
            else:
                candidates_i = np.nonzero(in_accuracy)
                candidates.append(list(zip(list(candidates_i[0]), list(candidates_i[1]))))
                involvement[detection] = in_accuracy

        if not independent:
            def helper_fct(nth: int, idx: Tuple[int, int]):
                my_detection = detections[nth]
                candidate_of_interest = []

                idx_to_consider = []
                for i in range(nth-1):
                    other_detection = detections[i]
                    dist = np.linalg.norm(np.array([my_detection[0] - other_detection[0],
                                                    my_detection[1] - other_detection[1]]))
                    if dist <= max_dependency_dist:
                        candidate_of_interest.append(candidates[i])
                        idx_to_consider.append(i)

                pairs = itertools.product(*candidate_of_interest)

                value = 0
                for count1, pair in enumerate(pairs):
                    val = 1
                    for count2, element in enumerate(pair):
                        total_p = 1
                        helper = list(involvement.values())[idx_to_consider[count2]]
                        total_p -= helper[idx]
                        helper[idx] = 0
                        for i in range(count2):
                            pair_of_interest = pair[i]
                            total_p -= helper[pair_of_interest]
                            helper[pair_of_interest] = 0

                        val *= helper[element] / total_p if total_p != 0 else 0

                    total_p = 1
                    helper = involvement[detection]
                    for element in pair:
                        total_p -= helper[element]
                        helper[element] = 0

                    val *= 1 - helper[idx] / total_p if total_p != 0 else 1
                    value += val
                return value

            p_matrices = np.ones(self.shape + (len(detections),))
            for nth, detection in enumerate(detections):
                for idx in candidates[nth]:
                    p_matrices[idx + (nth, )] = helper_fct(nth, idx)
            mat = np.prod(p_matrices, axis=2)
        else:
            mat = res

        return mat

    def get_shapely(self, threshold: float = 0.7, resampling_resolution: float = 0.2) -> List[Polygon]:
        """
        Returns a list of polygons representing the estimated obstacles in world coordinate frame
        @param threshold: Threshold for probability. Above this value, a point is considered as belonging to an obstacle
        @param resampling_resolution: The distribution is resampled on a uniform grid with the passed resolution
        @return: List of polygons
        """
        pose = SE2_from_translation_angle(np.array([self.previous_state.x, self.previous_state.y]),
                                          self.previous_state.theta)

        radius = self.current_belief.radius
        int_radius = self.current_belief.internal
        n_shape = 2 * int(radius / resampling_resolution)
        half_n_internal = int(int_radius / resampling_resolution)
        steps = list(np.linspace(-radius, radius, n_shape))
        ys, xs = np.meshgrid(steps, steps)
        query_pos = np.vstack((xs.ravel(), ys.ravel())).T
        results = self.current_belief.values_by_positions(query_pos, if_nan=0)
        res = np.reshape(results, (n_shape, n_shape))
        res[int((n_shape / 2) - half_n_internal):int((n_shape / 2) + half_n_internal),
            int((n_shape / 2) - half_n_internal):int((n_shape / 2) + half_n_internal)] = 0
        thresholded = np.where(res > threshold, 1, 0)
        connectivity_array = np.array([[1, 1, 1],
                                       [1, 1, 1],
                                       [1, 1, 1]])
        polygons = []
        labeled_array, num_features = measurements.label(thresholded, structure=connectivity_array)
        labeled_array = np.reshape(labeled_array, n_shape*n_shape)
        for i in range(num_features):
            idx_s = np.argwhere(labeled_array == (i+1))
            points = query_pos[idx_s, :]
            points = np.reshape(points, (points.shape[0], 2))

            multipoint = MultiPoint(points)
            poly = multipoint.convex_hull
            poly = poly.buffer(resampling_resolution)
            poly = affine_transform(poly, [pose[0, 0], pose[0, 1], pose[1, 0],
                                           pose[1, 1], pose[0, 2], pose[1, 2]])
            polygons.append(poly)
        return polygons

    def simulation_ended(self):
        """
        Generate animation with moving distribution. Only if generate_moving_dist is true.
        """

        if self.generate_moving_dist:
            fps = 10
            frn = len(self.data)

            change_frames_to: int = 0
            if change_frames_to:
                factor = frn/change_frames_to
                data_new = []
                for i in range(change_frames_to):
                    idx = int(factor*i)
                    data_new.append(self.data[idx])

                frn = change_frames_to
                self.data = data_new

            x = np.zeros(self.pos_x.shape + (frn, ))
            y, z = np.zeros(x.shape), np.zeros(x.shape)
            for i, data in enumerate(self.data):
                my_pose = data[1]
                for idx, position in self.current_belief.gen_structure():
                    wld_position = SE2_apply_T2(my_pose, position)
                    x[idx + (i, )] = wld_position[0]
                    y[idx + (i, )] = wld_position[1]
                z[:, :, i] = data[0]

            def update_plot(frame_number, z_array, plot):
                plot[0].remove()
                plot[0] = ax.plot_surface(x[:, :, frame_number], y[:, :, frame_number], z_array[:, :, frame_number],
                                          cmap="magma")

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            plot = [ax.plot_surface(x[:, :, 0], y[:, :, 0], z[:, :, 0], color='0.75', rstride=1, cstride=1)]
            ax.set_zlim(0, 1.1)
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            multiplier = min(x_max - x_min, y_max - y_min)
            ax.set_box_aspect((np.ptp(x), np.ptp(y), multiplier*np.ptp(z)))
            ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(z, plot), interval=1000 / fps)
            ani.save("test.gif")
