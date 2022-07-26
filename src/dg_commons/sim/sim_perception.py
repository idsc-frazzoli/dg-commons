from abc import ABC, abstractmethod
from bisect import bisect_left
from dataclasses import replace
from typing import List, Dict, Tuple

from dg_commons import PlayerName, SE2Transform, sPolygon2crPolygon, fd
from dg_commons.perception.sensor import Sensor
from dg_commons.sim import SimObservations, PlayerObservations, SimTime
from dg_commons.sim.models import extract_pose_from_state
from dg_commons.sim.scenarios import DgScenario
from commonroad_dc.pycrcc import Polygon as crPolygon


class ObsFilter(ABC):
    @abstractmethod
    def sense(self, scenario: DgScenario, full_obs: SimObservations, pov: PlayerName) -> SimObservations:
        """
        Filters the observations of a player based on their point of view.
        :param scenario: scenario at hand
        :param full_obs: full observations
        :param pov: which player's point of view
        :return:
        """
        pass


class IdObsFilter(ObsFilter):
    """Identity visibility filter"""

    def sense(self, scenario: DgScenario, full_obs: SimObservations, pov: PlayerName) -> SimObservations:
        return full_obs


class FovObsFilter(ObsFilter):
    """
    FOV visibility filter
    """

    def __init__(self, sensor: Sensor):
        self.sensor: Sensor = sensor
        self._static_obstacles: List[crPolygon] = []
        self._tmp_debug: int = 0

    def sense(self, scenario: DgScenario, full_obs: SimObservations, pov: PlayerName) -> SimObservations:
        """
        Filters the observations of a player.
        :param scenario: scenario at hand
        :param full_obs: full observations
        :param pov: which player's point of view
        :return:
        """
        # first update sensor pose
        self.sensor.pose = SE2Transform.from_SE2(extract_pose_from_state(full_obs.players[pov].state))
        # then filter
        if not self._static_obstacles:
            self._static_obstacles = [sPolygon2crPolygon(o.shape) for o in scenario.static_obstacles.values()]

        self._tmp_debug += 1
        dynamic_obstacles = [sPolygon2crPolygon(p_obs.occupancy) for p, p_obs in full_obs.players.items() if p != pov]

        all_obstacles = self._static_obstacles + dynamic_obstacles

        fov_poly = self.sensor.fov_as_polygon(all_obstacles)
        new_players: Dict[PlayerName, PlayerObservations] = {pov: full_obs.players[pov]}
        for p, p_obs in full_obs.players.items():
            if p == pov:
                continue

            if fov_poly.intersects(p_obs.occupancy):
                new_players[p] = p_obs

        return replace(full_obs, players=fd(new_players))


class DelayedObsFilter(ObsFilter):
    """Wrapper for the observations that introduces delay/latency"""

    def __init__(self, obs_filter: ObsFilter, latency: SimTime):
        assert issubclass(type(obs_filter), ObsFilter)
        self.obs_filter = obs_filter
        self.latency = latency
        self.obs_history: List[Tuple[SimTime, SimObservations]] = []

    def sense(self, scenario: DgScenario, full_obs: SimObservations, pov: PlayerName) -> SimObservations:
        obs = self.obs_filter.sense(scenario, full_obs, pov)
        shifted_t = obs.time + self.latency
        self.obs_history.append((shifted_t, replace(obs, time=shifted_t)))
        history_ts = self._get_obs_history_timestamps()
        idx = bisect_left(history_ts, obs.time)
        return self.obs_history[idx][1]

    def _get_obs_history_timestamps(self) -> List[SimTime]:
        return [_[0] for _ in self.obs_history]
