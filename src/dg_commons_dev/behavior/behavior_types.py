from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from dataclasses import dataclass
from dg_commons_dev.utils import BaseParams


Obs = TypeVar('Obs')
Rel = TypeVar('Rel')
S = TypeVar("S")
SParams = TypeVar("SParams")


@dataclass
class SituationParams(BaseParams):
    """ The situation parameters parametrize if/how a situation occurs """
    pass


class Situation(ABC, Generic[Obs, SParams]):
    """ A situation is a set of circumstances in which one finds oneself """

    @abstractmethod
    def update_observations(self, new_obs: Obs):
        """
        Update the information about the circumstances, choose whether this particular situation is occurring and
        computes some key parameters
        """
        pass

    @abstractmethod
    def is_true(self) -> bool:
        """ Returns whether this particular situation is occurring """
        pass

    @abstractmethod
    def infos(self) -> SParams:
        """ Returns important parameters describing this situation """
        pass


@dataclass
class BehaviorParams(BaseParams):
    """ The behavior parameters parametrize the behavior """
    pass


class Behavior(ABC, Generic[Obs, S]):
    """ Behavior manages the process of deciding which situation is occurring """

    @abstractmethod
    def update_observations(self, new_obs: Obs):
        """ New observations come and a decision on the current situation is made """
        pass

    @abstractmethod
    def get_situation(self, at: float) -> S:
        """ The current situation is returned """
        pass
