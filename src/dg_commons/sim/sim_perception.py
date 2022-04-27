from abc import ABC, abstractmethod

from dg_commons import PlayerName
from dg_commons.perception.sensor import Sensor
from dg_commons.sim import SimObservations
from dg_commons.sim.scenarios import DgScenario


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

    def sense(self, scenario: DgScenario, full_obs: SimObservations, pov: PlayerName) -> SimObservations:
        """
        Filters the observations of a player.
        :param scenario: scenario at hand
        :param full_obs: full observations
        :param pov: which player's point of view
        :return:
        """
