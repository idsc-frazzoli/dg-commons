from geometry import T2value
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import yaml
import os

path = os.path.dirname(os.path.abspath(__file__))
sensing_curves_path = os.path.join(path, 'sensing_data')


def get_sensor_curves(sensor_name, min_distance=0, max_distance=40, max_n_points=200):
    path_to_file = os.path.join(sensing_curves_path, sensor_name) + ".yaml"
    with open(path_to_file, 'r') as file:
        sensing_data = yaml.full_load(file)

    fp = sensing_data["fp"]
    fn = sensing_data["fn"]
    acc = sensing_data["accuracy"]
    ds = float(sensing_data["ds"])
    radius = min(float(sensing_data["max_distance"]), max_distance)

    fp_curve = SensingCurve(sensor_name, fp, radius, min_distance, ds, max_n_points)
    fn_curve = SensingCurve(sensor_name, fn, radius, min_distance, ds, max_n_points)
    acc_curve = SensingCurve(sensor_name, acc, radius, min_distance, ds, max_n_points)

    return fp_curve, fn_curve, acc_curve


class SensingCurve:
    def __init__(self, name, values, max_distance, min_distance, step_size, max_n_points=200):
        self.n_points = min(int((max_distance - min_distance) / step_size), max_n_points)
        self.sensor_name = name
        self.values = values
        self.step_size = step_size
        self.max_distance = max_distance

    def evaluate_distribution_dist_interp(self, dist: float) -> float:
        """
        Evaluate distribution
        @param dist: Relative distance
        @return: Probability of event
        """
        if self.max_distance < dist:
            return 0.5

        step_n = int(dist / self.step_size)
        top, bottom = math.ceil(step_n), math.floor(step_n)
        return self.values[bottom] + ((step_n - bottom) / self.step_size) * (self.values[top] - self.values[bottom])

    def evaluate_distribution_dist(self, dist: float) -> float:
        """
        Evaluate distribution
        @param dist: Relative distance
        @return: Probability of event
        """
        if self.max_distance < dist:
            return 0.5
        else:
            idx = int(dist / self.step_size) - 1
            return self.values[idx]

    def evaluate_distribution(self, rel_pos: T2value) -> float:
        """
        Evaluate distribution
        @param rel_pos: Relative position
        @return: Probability of event
        """
        distance = np.linalg.norm(rel_pos)
        return self.evaluate_distribution_dist(distance)

    def sample(self, rel_pos: T2value) -> bool:
        """
        Sample from distribution
        @param rel_pos: Relative position
        @return: Whether event occurred or not
        """
        return random.uniform(0, 1) < self.evaluate_distribution(rel_pos)


class PCurve:
    """
    Test Model FP/FN/Accuracy as

    P(Event at distance d) = 2 * arctan(distance * parameter) / pi * convergence_param
    """
    sensor_name = "Test"

    def __init__(self, conv_speed: float, conv_val: float = 0.5):
        self.conv_speed = conv_speed
        self.conv_val = conv_val

    def evaluate_distribution(self, rel_pos: T2value) -> float:
        """
        Evaluate distribution
        @param rel_pos: Relative position
        @return: Probability of event
        """
        pseudo_distance = np.linalg.norm(rel_pos) * self.conv_speed
        return math.atan(pseudo_distance) / math.pi * 2 * self.conv_val

    def evaluate_distribution_dist(self, dist: float) -> float:
        """
        Evaluate distribution
        @param dist: Relative position
        @return: Probability of event
        """
        pseudo_distance = dist * self.conv_speed
        return math.atan(pseudo_distance) / math.pi * 2 * self.conv_val

    def sample(self, rel_pos: T2value) -> bool:
        """
        Sample from distribution
        @param rel_pos: Relative position
        @return: Whether event occurred or not
        """
        return random.uniform(0, 1) < self.evaluate_distribution(rel_pos)
