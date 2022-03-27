from abc import ABC, abstractmethod
from typing import Callable, Optional, Any, TypeVar

from dg_commons import DgSampledSequence, U, PlayerName, X
from dg_commons.sim import SimTime
from dg_commons.sim.simulator_structures import SimObservations, InitSimObservations

__all__ = ["TAgent", "Agent", "NPAgent", "PolicyAgent"]

TAgent = TypeVar("TAgent", bound="Agent")


class Agent(ABC):
    """This provides the abstract interface of an agent"""

    @abstractmethod
    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """This method will get called once for each player at the beginning of the simulation"""
        pass

    # todo make this agent be able to take a map, static obstacles and a goal
    # a DgScenario object? or simply pass them via the init method?!

    @abstractmethod
    def get_commands(self, sim_obs: SimObservations) -> U:
        """This method gets called for each player inside the update loop of the simulator"""
        pass

    def on_get_extra(
        self,
    ) -> Optional[Any]:
        """This method gets called for each player inside the update loop of the simulator,
        it is used purely for logging. For example pass all the trajectories that have been generated at that step.
          To return something only at certain timestamps simply return None in the others."""
        pass


class NPAgent(Agent):
    """
    Non-playing character which returns commands based purely on the sim time
    """

    def __init__(self, commands_plan: DgSampledSequence[U]):
        assert isinstance(commands_plan, DgSampledSequence)
        self.commands_plan = commands_plan

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        pass

    def get_commands(self, sim_obs: SimObservations) -> U:
        t: SimTime = sim_obs.time
        return self.commands_plan.at_or_previous(t)


class PolicyAgent(Agent):
    """
    Playing character which returns commands based on its policy (function from state to commands)
    """

    def __init__(self, policy: Callable[[X], U]):
        self.policy = policy
        self.my_name: Optional[PlayerName] = None

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        self.my_name = init_sim_obs.my_name

    def get_commands(self, sim_obs: SimObservations) -> U:
        my_state: X = sim_obs.players[self.my_name]
        return self.policy(my_state)
