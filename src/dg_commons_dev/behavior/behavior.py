import time
from dataclasses import dataclass
from typing import MutableMapping, Dict, Optional, List, Union, Tuple, Any
from duckietown_world import relative_pose, SE2Transform
from geometry import SE2value
from dg_commons import PlayerName
from dg_commons.utils_toolz import valmap
from dg_commons.sim.models import extract_pose_from_state
from dg_commons.sim.simulator_structures import PlayerObservations
from dg_commons_dev.behavior.behavior_types import Behavior, Situation
from dg_commons_dev.behavior.emergency import Emergency, EmergencyParams
from dg_commons_dev.behavior.yield_to import Yield, YieldParams
from dg_commons_dev.behavior.cruise import CruiseParams, Cruise
from dg_commons_dev.behavior.replan import ReplanParams, Replan
from dg_commons_dev.behavior.utils import SituationObservations
import copy
from dg_commons_dev.utils import BaseParams
from dg_commons.sim.scenarios.structures import StaticObstacle
from dg_commons.maps.lanes import DgLanelet
from dg_commons_dev.controllers.controller_types import Reference
from dg_commons_dev.behavior.utils import SituationObservations, occupancy_prediction, SituationPolygons,\
    PlayerObservations, intentions_prediction


@dataclass
class BehaviorSituation:
    """ Data structure to communicate the most prominent situation and its description """

    situation: Situation = None
    """ Current Situation """

    def is_emergency(self) -> bool:
        """
        @return: is the most prominent situation an emergency situation?
        """
        assert self._is_situation_type()
        return isinstance(self.situation, Emergency)

    def is_yield(self) -> bool:
        """
        @return: is the most prominent situation a yield situation?
        """
        assert self._is_situation_type()
        return isinstance(self.situation, Yield)

    def is_cruise(self) -> bool:
        """
        @return: is the most prominent situation a cruise situation?
        """
        assert self._is_situation_type()
        return isinstance(self.situation, Cruise)

    def is_replan(self) -> bool:
        """
        @return: is the most prominent situation a replan situation?
        """
        assert self._is_situation_type()
        return isinstance(self.situation, Replan)

    def _is_situation_type(self):
        help1: bool = self.situation is not None
        help2: bool = isinstance(self.situation, Emergency) or isinstance(self.situation, Yield) or \
            isinstance(self.situation, Cruise) or isinstance(self.situation, Replan)
        return help1 and help2


@dataclass
class SpeedBehaviorParam(BaseParams):
    safety_time_braking: float = 1.5
    """Evaluates safety distance from vehicle in front based on distance covered in this delta time"""
    emergency: type(Emergency) = Emergency
    emergency_params: EmergencyParams = EmergencyParams()
    """ Emergency Behavior """
    yield_to: type(Emergency) = Yield
    yield_params: YieldParams = YieldParams()
    """ Yield Behavior """
    cruise: type(Cruise) = Cruise
    cruise_params: CruiseParams = CruiseParams()
    """ Cruise Params """
    replan: type(Replan) = Replan
    replan_params: ReplanParams = ReplanParams()
    """ Replan Params  """

    min_look_ahead: float = 30
    """ Minimal look ahead distance """
    min_safety_dist: float = 20
    """ Minimal safety distance """
    distance_from_obstacles: float = 0.5
    """ Minimal distance to keep from obstacles """

    dt_commands: float = 0.1
    """ Period of decision making """
    def __post_init__(self):
        assert 0 <= self.safety_time_braking <= 50
        assert isinstance(self.emergency_params, self.emergency.REF_PARAMS)
        assert isinstance(self.yield_params, self.yield_to.REF_PARAMS)
        assert isinstance(self.cruise_params, self.cruise.REF_PARAMS)
        assert isinstance(self.replan_params, self.replan.REF_PARAMS)
        assert 0 < self.distance_from_obstacles
        assert 0 <= self.dt_commands <= 50


class SpeedBehavior(Behavior[MutableMapping[PlayerName, PlayerObservations], Tuple[float, Situation]]):
    """ Determines the reference speed and the situation """

    def __init__(self, params: SpeedBehaviorParam = SpeedBehaviorParam(), my_name: Optional[PlayerName] = None):
        self.params: SpeedBehaviorParam = copy.deepcopy(params)
        self.my_name: PlayerName = my_name
        self.agents: MutableMapping[PlayerName, PlayerObservations]
        self.speed_ref: float = 0

        self.yield_to = self.params.yield_to(self.params.yield_params, self.params.safety_time_braking)
        self.emergency = self.params.emergency(self.params.emergency_params, self.params.safety_time_braking, plot=True)
        self.cruise = self.params.cruise(self.params.cruise_params, self.params.safety_time_braking, plot=True)
        self.replan = self.params.replan(self.params.replan_params, self.params.safety_time_braking, plot=True)
        self.obs: SituationObservations = SituationObservations(my_name=self.my_name, dt_commands=params.dt_commands,
                                                                distances=(self.params.min_look_ahead,
                                                                           self.params.min_safety_dist))
        self.situation: BehaviorSituation = BehaviorSituation()
        self.something_found: Optional[float] = None
        """ The speed reference"""

    def update_observations(self, agents: MutableMapping[PlayerName, PlayerObservations], ref: Reference,
                            static_obstacles: List[StaticObstacle] = []):
        self.obs.agents = agents

        my_pose = extract_pose_from_state(agents[self.my_name].state)

        def rel_pose(other_obs: PlayerObservations) -> SE2Transform:
            other_pose: SE2value = extract_pose_from_state(other_obs.state)
            return SE2Transform.from_SE2(relative_pose(my_pose, other_pose))

        agents_rel_pose: Dict[PlayerName, SE2Transform] = valmap(rel_pose, agents)
        self.obs.rel_poses = agents_rel_pose
        self.obs.static_obstacles = static_obstacles
        self.obs.planned_path = (ref.path, ref.along_lane)
        self.obs.distances = self._get_look_ahead(agents[self.my_name].state.vx)

    def get_situation(self, at: float,
                      consider_replanning: bool = True) -> Tuple[float, BehaviorSituation, Any]:
        self.obs.my_name = self.my_name

        polygon, polygons = \
            intentions_prediction(self.obs.distances[0], self.obs.planned_path[0],
                                  self.obs.planned_path[1], min_distance=self.params.distance_from_obstacles)
        c_frames, c_classes = self.cruise.update_observations(self.obs, polygon, polygons)
        if consider_replanning:
            _, _ = self.replan.update_observations(self.obs, polygon, polygons)
        e_frames, e_classes = self.emergency.update_observations(self.obs, polygon, polygons)
        if self.emergency.is_true():
            self.situation.situation = self.emergency
            self.speed_ref = 0
            self.something_found = self.obs.distances[0]
        elif consider_replanning and self.replan.is_true():
            self.situation.situation = self.replan
            self.speed_ref = self.cruise.infos().speed_ref
            self.something_found = self.obs.distances[0]
        else:
            self.situation.situation = self.cruise
            self.speed_ref = self.cruise.infos().speed_ref
            if self.cruise.infos().is_following:
                self.something_found = self.obs.distances[0]

        frames = c_frames + e_frames
        classes = c_classes + e_classes
        return self.speed_ref, self.situation, zip(frames, classes)

    def reset(self):
        self.cruise = Cruise

    def simulation_ended(self):
        self.emergency.simulation_ended()
        self.cruise.simulation_ended()

    def _get_look_ahead(self, vel: float) -> Tuple[float, float]:
        """
        Compute look ahead distance
        @param vel: Current velocity
        @return: Look ahead distance
        """
        if vel < 0:
            look_ahead = self.params.min_look_ahead
            safety_dist = self.params.min_safety_dist
        else:
            look_ahead = self.params.safety_time_braking * vel + self.params.min_look_ahead
            safety_dist = self.params.safety_time_braking * vel + self.params.min_safety_dist

        if self.something_found:
            look_ahead = max(self.something_found, look_ahead)
        return look_ahead, safety_dist
