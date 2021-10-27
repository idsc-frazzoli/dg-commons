import numpy as np
from dg_commons_dev.analysis.metrics_def import Metric, MetricEvaluationContext, MetricEvaluationResult, EvaluatedMetric
from typing import Callable
from dg_commons.sim.models.model_utils import acceleration_constraint
from dg_commons.sim.models.vehicle_utils import steering_constraint
import logging
from typing import List, Dict, Tuple
from dg_commons import PlayerName, X, U
from dg_commons.seq.sequence import Timestamp, DgSampledSequence
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from duckietown_world import LanePose
import matplotlib.pyplot as plt


def get_evaluated_metric(
        players: List[PlayerName], f: Callable[[PlayerName], EvaluatedMetric]
) -> MetricEvaluationResult:
    mer: Dict[PlayerName, EvaluatedMetric] = {}
    for player_name in players:
        mer[player_name] = f(player_name)
    return mer


class DeviationLateral(Metric):
    brief_description: str = "Lateral Deviation"
    file_name: str = brief_description.replace(" ", "_").lower()
    description = "This metric describes the deviation from reference path. "
    scale: float = 1.0
    relative: Dict[PlayerName, List[float]] = {}

    def evaluate(self, context: MetricEvaluationContext,
                 plot: bool = False, output_dir: str = '') -> MetricEvaluationResult:

        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            _, interval = context.get_interval(player)
            player_lane_pos: DgSampledSequence[LanePose] = context.get_lane_poses(player)

            relative: List[float] = []
            for time in interval:
                lane_pose: LanePose = player_lane_pos.at(time)
                relative.append(float(lane_pose.lateral))

            self.relative[player]: List[float] = relative
            absolute = [abs(v) for v in relative]
            ret = self.get_evaluated_metric(interval=interval, val=absolute)
            return ret

        result: MetricEvaluationResult = get_evaluated_metric(context.get_players(), calculate_metric)
        if plot:
            self.plot(result, context, output_dir)
        return result

    def plot(self, result: MetricEvaluationResult, context: MetricEvaluationContext, output_dir: str):
        stamps: DgSampledSequence[Timestamp] = self.plot_increment_cumulative(result, context, output_dir)

        for player in context.get_players():
            plt.plot(stamps, self.relative[player], label=player)

        Metric.save_fig(output_dir, title=self.description, name=self.file_name)

        for i, player in enumerate(context.get_players()):
            center_points: List[np.ndarray] = [point.q.p for point in context.planned_lanes[player].control_points]
            x_center: List[float] = [q[0] for q in center_points]
            y_center: List[float] = [q[1] for q in center_points]

            x_trajectory: List[float] = [q.p[0] for q in context.get_poses(player)]
            y_trajectory: List[float] = [q.p[1] for q in context.get_poses(player)]

            points: List[np.ndarray] = context.planned_lanes[player].lane_profile()
            x_lateral: List[float] = [point[0] for point in points]
            y_lateral: List[float] = [point[1] for point in points]
            x_max, x_min = max(x_trajectory), min(x_trajectory)
            y_max, y_min = max(y_trajectory), min(y_trajectory)
            delta_x, delta_y = x_max - x_min, y_max - y_min
            if delta_y > delta_x:
                factor_y: float = 0.1
                factor_x: float = delta_y/delta_x/4
            else:
                factor_y: float = delta_x/delta_y/4
                factor_x: float = 0.1

            plt.xlim(x_min - factor_x * delta_x, x_max + factor_x * delta_x)
            plt.ylim(y_min - factor_y * delta_y, y_max + factor_y * delta_y)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.draw()

            plt.scatter(x=x_trajectory[0], y=y_trajectory[0], c='r', label='x0')
            plt.plot(x_lateral, y_lateral, 'b', label='Boundaries')
            plt.plot(x_center, y_center, '--', linewidth=0.5, color="lightgray", label='Center Line')
            plt.plot(x_trajectory, y_trajectory, 'r', linewidth=0.5, label='Trajectory')

            Metric.save_fig(output_dir, title=player, name=player, dpi=200)


class DeviationVelocity(Metric):
    brief_description: str = "Velocity Deviation"
    file_name: str = brief_description.replace(" ", "_").lower()
    description: str = "This metric describes the deviation from reference velocity. "
    scale: float = 1.0
    relative: Dict[PlayerName, List[float]] = {}

    def evaluate(self, context: MetricEvaluationContext,
                 plot: bool = False, output_dir: str = '') -> MetricEvaluationResult:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            interval, _ = context.get_interval(player)
            player_states: X = context.actual_trajectory[player]
            target_vels: DgSampledSequence[float] = context.target_velocities[player]

            relative: List[float] = []
            for time in interval:
                player_vel: float = player_states.at(time).vx
                target_vel: float = target_vels.at(time)
                relative.append(float(player_vel - target_vel))

            self.relative[player]: List[float] = relative
            absolute = [abs(v) for v in relative]
            ret: EvaluatedMetric = self.get_evaluated_metric(interval=interval, val=absolute)
            return ret

        result: MetricEvaluationResult = get_evaluated_metric(context.get_players(), calculate_metric)
        if plot:
            self.plot(result, context, output_dir)
        return result

    def plot(self, result: MetricEvaluationResult, context: MetricEvaluationContext, output_dir: str):
        stamps: DgSampledSequence[Timestamp] = self.plot_increment_cumulative(result, context, output_dir)

        for player in context.get_players():
            plt.plot(result[player].incremental.timestamps, result[player].incremental.values, label=player)

        Metric.save_fig(output_dir, name=self.brief_description, title=self.description)

        for player in context.get_players():
            deviations = [self.relative[player][i] + context.target_velocities[player].values[i]
                          for i, _ in enumerate(self.relative[player])]
            plt.plot(stamps, deviations, label=player)
            plt.plot(stamps, context.target_velocities[player].values, "--", color="lightgray")

        Metric.save_fig(output_dir, title=self.description, name=self.file_name)


class SteeringVelocity(Metric):
    brief_description: str = "Steering Velocity"
    file_name: str = brief_description.replace(" ", "_").lower()
    description: str = "This metric describes the commanded steering velocity"
    scale: float = 1.0
    relative: Dict[PlayerName, List[float]] = {}

    def evaluate(self, context: MetricEvaluationContext,
                 plot: bool = False, output_dir: str = '') -> MetricEvaluationResult:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            _, interval = context.get_interval(player)
            commands: U = context.commands[player]
            states: X = context.actual_trajectory[player]
            vehicle_params: VehicleParameters = context.vehicle_params[player]

            relative: List[float] = []
            for time in interval:
                theta: float = float(states.at(time).delta)
                steering_vel: float = float(commands.at(time).ddelta)
                logging.disable()
                relative.append(steering_constraint(theta, steering_vel, vehicle_params))
                logging.disable(logging.NOTSET)

            self.relative[player]: List[float] = relative
            absolute = [abs(v) for v in relative]
            ret: EvaluatedMetric = self.get_evaluated_metric(interval=interval, val=absolute)
            return ret

        result: MetricEvaluationResult = get_evaluated_metric(context.get_players(), calculate_metric)
        if plot:
            self.plot(result, context, output_dir)
        return result

    def plot(self, result: MetricEvaluationResult, context: MetricEvaluationContext, output_dir: str):
        stamps: DgSampledSequence[Timestamp] = self.plot_increment_cumulative(result, context, output_dir)

        for player in context.get_players():
            plt.plot(stamps, self.relative[player], label=player)

        max_value: float = context.vehicle_params[player].ddelta_max
        min_value: float = - max_value
        n: int = len(stamps)
        plt.plot(stamps, n * [max_value], '--', color='lightgray', label='limits')
        plt.plot(stamps, n * [min_value], '--', color='lightgray')

        Metric.save_fig(output_dir, name=self.file_name, title=self.description)


class Acceleration(Metric):
    brief_description: str = "Acceleration"
    file_name: str = brief_description.replace(" ", "_").lower()
    description: str = "This metric describes the commanded acceleration"
    scale: float = 1.0
    relative: Dict[PlayerName, List[float]] = {}

    def evaluate(self, context: MetricEvaluationContext,
                 plot: bool = False, output_dir: str = '') -> MetricEvaluationResult:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            _, interval = context.get_interval(player)
            commands: U = context.commands[player]
            states: X = context.actual_trajectory[player]
            vehicle_params: VehicleParameters = context.vehicle_params[player]

            relative: List[float] = []
            for time in interval:
                speed: float = states.at(time).vx
                acc: float = float(commands.at(time).acc)
                logging.disable()
                relative.append(acceleration_constraint(speed, acc, vehicle_params))
                logging.disable(logging.NOTSET)

            self.relative[player]: List[float] = relative
            absolute = [abs(v) for v in relative]
            ret: EvaluatedMetric = self.get_evaluated_metric(interval=interval, val=absolute)
            return ret

        result: MetricEvaluationResult = get_evaluated_metric(context.get_players(), calculate_metric)
        if plot:
            self.plot(result, context, output_dir)
        return result

    def plot(self, result: MetricEvaluationResult, context: MetricEvaluationContext, output_dir):
        stamps: DgSampledSequence[Timestamp] = self.plot_increment_cumulative(result, context, output_dir)

        for player in context.get_players():
            plt.plot(stamps, self.relative[player], label=player)

        acc_limits: Tuple[float, float] = context.vehicle_params[player].acc_limits
        min_value, max_value = acc_limits[0], acc_limits[1]
        n: int = len(stamps)
        plt.plot(stamps, n * [max_value], '--', color='lightgray', label='limits')
        plt.plot(stamps, n * [min_value], '--', color='lightgray')

        Metric.save_fig(output_dir, name=self.file_name, title=self.description)


class DTForCommand(Metric):
    brief_description: str = "DTForCommand"
    file_name: str = brief_description.replace(" ", "_").lower()
    description: str = "This metric describes the time required to formulate a command"
    scale: float = 1.0

    def evaluate(self, context: MetricEvaluationContext,
                 plot: bool = False, output_dir: str = '') -> MetricEvaluationResult:
        def calculate_metric(player: PlayerName) -> EvaluatedMetric:
            _, interval = context.get_interval(player)
            dt_commands: DgSampledSequence[float] = context.dt_commands[player]

            vals: List[float] = []
            for time in interval:
                dt: float = dt_commands.at(time)
                vals.append(dt)

            ret: EvaluatedMetric = self.get_evaluated_metric(interval=interval, val=vals)
            return ret

        result: MetricEvaluationResult = get_evaluated_metric(context.get_players(), calculate_metric)
        if plot:
            self.plot(result, context, output_dir)
        return result

    def plot(self, result: MetricEvaluationResult, context: MetricEvaluationContext, output_dir):
        self.plot_increment_cumulative(result, context, output_dir)


# Workaround to have a list of all metrics types available TODO: Find alternative solution
metrics_list: List[type(Metric)] = [DeviationVelocity, DeviationLateral, SteeringVelocity, Acceleration, DTForCommand]
