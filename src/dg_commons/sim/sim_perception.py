from abc import ABC, abstractmethod
from dataclasses import replace
from typing import List, Dict

from dg_commons import PlayerName, SE2Transform, sPolygon2crPolygon, fd
from dg_commons.perception.sensor import Sensor
from dg_commons.sim import SimObservations, PlayerObservations
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

    def sense(self, scenario: DgScenario, full_obs: SimObservations, pov: PlayerName) -> SimObservations:
        """
        Filters the observations of a player.
        :param scenario: scenario at hand
        :param full_obs: full observations
        :param pov: which player's point of view
        :return:
        """
        # first update sensor pose
        self.pose = SE2Transform.from_SE2(extract_pose_from_state(full_obs.players[pov].state))
        # then filter
        if not self._static_obstacles:
            self._static_obstacles = [sPolygon2crPolygon(o.shape) for o in scenario.static_obstacles.values()]

        all_obstacles = self._static_obstacles + [
            sPolygon2crPolygon(p_obs.occupancy) for p, p_obs in full_obs.players.items() if p != pov
        ]
        fov_poly = self.sensor.fov_as_polygon(all_obstacles)
        new_players: Dict[PlayerName, PlayerObservations] = {pov: full_obs.players[pov]}
        for p, p_obs in full_obs.players.items():
            if p == pov:
                continue
            if fov_poly.intersects(p_obs.occupancy):
                new_players[p] = p_obs
        return replace(full_obs, players=fd(new_players))


# todo in can be implemented a more efficient version of this filter
