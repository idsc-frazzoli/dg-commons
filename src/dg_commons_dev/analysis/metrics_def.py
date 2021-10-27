from dg_commons.maps.lanes import DgLanelet
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List, Mapping, Optional, Dict, Tuple
from dg_commons import PlayerName, X, U
from dg_commons.seq.sequence import Timestamp, DgSampledSequence
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.models.vehicle import VehicleGeometry
from duckietown_world import SE2Transform, LanePose
from dg_commons import SE2Transform, relative_pose
import os
import matplotlib.pyplot as plt
from geometry import translation_angle_scale_from_E2, SE2_from_translation_angle
from dg_commons.seq.seq_op import seq_integrate
from functools import lru_cache


class EvaluatedMetric:
    total: float
    description: str
    title: str
    incremental: DgSampledSequence
    cumulative: DgSampledSequence

    def __init__(
            self,
            title: str,
            description: str,
            total: float,
            incremental: Optional[DgSampledSequence],
            cumulative: Optional[DgSampledSequence],
    ):
        self.title = title
        self.description = description
        self.total = total
        self.incremental = incremental
        self.cumulative = cumulative

    def __repr__(self):
        return f"{self.title} = {self.total:.2f}"

    def __add__(self, other: "EvaluatedMetric") -> "EvaluatedMetric":
        if other is None:
            return self
        return self.add(m1=self, m2=other)

    @staticmethod
    @lru_cache(None)
    def add(m1: "EvaluatedMetric", m2: "EvaluatedMetric") -> "EvaluatedMetric":
        if m1.title != m2.title:
            raise NotImplementedError(f"add implemented only for same metric, "
                                      f"received {m1.title, m2.title}")

        if m1.incremental is None:
            inc = None
        else:
            t_1, t_2 = m1.incremental.timestamps, m2.incremental.timestamps
            if t_1[-1] != t_2[0]:
                raise ValueError(f"Timestamps need to be consecutive - {t_1[-1], t_2[0]}")
            times_i = t_1 + t_2[1:]
            vals_i = m1.incremental.values + m2.incremental.values[1:]
            inc = DgSampledSequence(timestamps=times_i, values=vals_i)

        if m1.cumulative is None:
            cum = None
        else:
            times_c = m1.cumulative.timestamps + m2.cumulative.timestamps
            c_end = m1.cumulative.values[-1]
            vals_c = m1.cumulative.values + tuple([v + c_end for v in m2.cumulative.values])
            cum = DgSampledSequence(timestamps=times_c, values=vals_c)

        return EvaluatedMetric(title=m1.title, description=m1.description,
                               total=m1.total + m2.total, incremental=inc, cumulative=cum)

    __radd__ = __add__


def get_integrated(sequence: DgSampledSequence[float]) -> Tuple[DgSampledSequence[float], float]:
    if len(sequence) <= 1:
        cumulative = 0.0
        dtot = 0.0
    else:
        cumulative = seq_integrate(sequence)
        dtot = cumulative.values[-1]
    return cumulative, dtot


def differentiate(val: List[float], t: List[Timestamp]) -> List[float]:
    if len(val) != len(t):
        msg = "values and times have different sizes - ({},{})," " can't differentiate".format(
            len(val), len(t)
        )
        raise ValueError(msg)

    def func_diff(i: int) -> float:
        dy = val[i + 1] - val[i]
        dx = float(t[i + 1] - t[i])
        if dx < 1e-8:
            raise ValueError(f"identical timestamps for func_diff - {t[i]}")
        return dy / dx

    ret: List[float] = [0.0] + [func_diff(i) for i in range(len(t) - 1)]
    return ret


@dataclass
class MetricEvaluationContext:
    planned_lanes: Mapping[PlayerName, DgLanelet]
    """ Planned lanes to follow for each player """

    actual_trajectory: Mapping[PlayerName, DgSampledSequence[X]]
    """ Trajectory for each player """

    commands: Mapping[PlayerName, DgSampledSequence[U]]
    """ Commands for each player """

    target_velocities: Mapping[PlayerName, DgSampledSequence[float]]
    """ Planned velocities """

    dt_commands: Mapping[PlayerName, DgSampledSequence[float]]
    """ Time required for a command """

    betas: Mapping[PlayerName, DgSampledSequence[float]]
    """ Where on the lane """

    vehicle_params: Optional[Mapping[PlayerName, VehicleParameters]] = None
    """ Vehicle parameters """

    geometry_params: Optional[Mapping[PlayerName, VehicleGeometry]] = None
    """ Vehicle parameters """

    def __post_init__(self):
        self.n_players: int = len(self.planned_lanes.keys())
        """ Number of players """

        self.vehicle_params = dict(zip(self.planned_lanes.keys(), self.n_players*[VehicleParameters.default_car()])) \
            if self.vehicle_params is None else self.vehicle_params

        self.geometry_params = dict(zip(self.planned_lanes.keys(), self.n_players*[VehicleGeometry.default_car()])) \
            if self.geometry_params is None else self.geometry_params

        poses: Dict[PlayerName, List[SE2Transform]] = {}
        lane_poses: Dict[PlayerName, DgSampledSequence[LanePose]] = {}

        for player, betas in self.betas.items():
            intervals = self.betas[player].get_sampling_points()
            path = self.planned_lanes[player]
            helper1 = []
            helper2 = []
            for time in intervals:
                beta = betas.at(time)
                state = self.actual_trajectory[player].at(time)

                position, angle = [state.x, state.y], state.theta
                q = SE2_from_translation_angle(position, angle)
                q0 = path.center_point(beta)

                along_lane = path.along_lane_from_beta(beta)
                rel = relative_pose(q, q0)
                r, relative_heading, _ = translation_angle_scale_from_E2(rel)
                lateral = r[1]

                lane_pose = path.lane_pose(along_lane=along_lane, relative_heading=relative_heading, lateral=lateral)

                helper1.append(SE2Transform(position, angle))
                helper2.append(lane_pose)

            poses[player] = helper1
            lane_poses[player] = DgSampledSequence(values=helper2, timestamps=intervals)

            self._pose: Dict[PlayerName, List[SE2Transform]] = poses
            self._lane_pose: Dict[PlayerName, DgSampledSequence[LanePose]] = lane_poses

    def get_interval(self, player: PlayerName) -> Tuple[Tuple[Timestamp], Tuple[Timestamp]]:
        return self.actual_trajectory[player].get_sampling_points(), self.commands[player].get_sampling_points()

    def get_players(self) -> List[PlayerName]:
        return list(self.actual_trajectory.keys())

    def get_poses(self, player: PlayerName) -> List[SE2Transform]:
        return self._pose[player]

    def get_lane_poses(self, player: PlayerName) -> DgSampledSequence[LanePose]:
        return self._lane_pose[player]


class Metric(metaclass=ABCMeta):
    _instances = {}
    brief_description: str
    file_name: str
    description: str
    scale: float

    def __new__(cls, *args, **kwargs):
        # Allow creation of only one instance of each subclass (singleton)
        if cls._instances.get(cls, None) is None:
            cls._instances[cls] = super(Metric, cls).__new__(cls, *args, **kwargs)
        return Metric._instances[cls]

    @abstractmethod
    def evaluate(self, context: MetricEvaluationContext) -> "MetricEvaluationResult":
        """ Evaluates the metric for all players given a context. """

    def get_evaluated_metric(self, interval: Tuple[Timestamp], val: List[float]) -> EvaluatedMetric:
        incremental: DgSampledSequence[float] = DgSampledSequence[float](interval, val)
        cumulative, total = get_integrated(incremental)
        ret: EvaluatedMetric = EvaluatedMetric(title=type(self).__name__, description=self.description,
                                               total=total, incremental=incremental, cumulative=cumulative)
        return ret

    def plot_increment_cumulative(self, result, context: MetricEvaluationContext, output_dir) \
            -> DgSampledSequence[Timestamp]:
        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        for player in context.get_players():
            stamps: DgSampledSequence[Timestamp] = result[player].incremental.timestamps
            axs[0].plot(stamps, result[player].incremental.values, label=player)
            axs[0].set_title("Absolute " + self.brief_description)
            axs[1].plot(result[player].cumulative.timestamps, result[player].cumulative.values, label=player)
            axs[1].set_title("Integral " + self.brief_description)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=5)

        for ax in fig.get_axes():
            ax.label_outer()
        Metric.save_fig(output_dir, title="", name=self.file_name + "_absolute", fig=fig)
        return stamps

    @staticmethod
    def save_fig(output_dir, title, name, dpi=None, fig=plt):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, shadow=True, ncol=5)
        try:
            fig.title(title)
        except Exception:
            fig.suptitle(title)
        fig_file = os.path.join(output_dir, name)

        fig.savefig(fig_file, dpi=dpi)
        plt.clf()


MetricEvaluationResult = Mapping[PlayerName, EvaluatedMetric]
PlayerOutcome = Mapping[Metric, EvaluatedMetric]
TrajGameOutcome = Mapping[PlayerName, PlayerOutcome]
