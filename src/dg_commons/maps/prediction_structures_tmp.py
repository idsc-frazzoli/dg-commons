from __future__ import annotations

from typing import List, Dict, Callable, Union, TypeVar, Generic, get_args
from toolz.dicttoolz import merge_with, valmap
from dg_commons import PlayerName


def multiply_or_keep(factors: List):
    if len(factors) == 2:
        return factors[0] * factors[1]
    elif len(factors) == 1:
        return factors[0]
    else:
        raise ValueError("Something is wrong in the dictionaries you are trying to multiply.")


# constraint type of V to float (and therefore also to int) and to bool
V = TypeVar("V", float, bool)


class GoalMapping(Dict[int, V], Generic[V]):
    """Values (probabilities or rewards or booleans) over possible resources (indexed by integers)."""

    def __add__(self, other: GoalMapping[V]):
        """Add two dictionaries. Can add dictionaries with different keys."""
        assert isinstance(other, GoalMapping)
        return merge_with(sum, self, other)

    def safe_add(self, other: GoalMapping[V]):
        """Only add dictionaries if they have the same keys."""
        assert isinstance(other, GoalMapping)
        assert other.keys() == self.keys(), "Safe addition can only be carried through when keys of both dicts " \
                                            "are the identical."
        return self.__add__(other)

    def __mul__(self, other: GoalMapping[V]):
        assert isinstance(other, GoalMapping)
        return merge_with(multiply_or_keep, self, other)

    def scalar_multiplication(self, other: float):
        if isinstance(other, int):
            other = float(other)
        assert isinstance(other, float)
        return valmap(lambda x: x * other, self)

    def __sub__(self, other: GoalMapping[V]):
        assert isinstance(other, GoalMapping)
        other = other.scalar_multiplication(-1.0)
        return self.__add__(other)

    def valfun(self, func: Callable):
        return GoalMapping[V](valmap(func, self))


class GoalRewards(GoalMapping[float]):
    """Methods specific for rewards"""


class GoalBools(GoalMapping[bool]):
    """Methods specific for booleans"""

    def boolean_intersect(self, other: GoalBools) -> GoalBools:
        """ Check if booleans are matching between two dicts. If they are not, other is considered the ground truth.
            No item removal is performed, instead the boolean value is changed.

            Example:
                self = {1: True, 2: True, 3: True}
                other = {1: True, 4: True}

                return {1: True,  4: True}
                """
        assert isinstance(other, GoalBools)
        for key in other:
            self[key] = other[key]

        keys_to_remove = []
        for key in self:
            if key in other.keys():
                continue
            else:
                keys_to_remove.append(key)

        for delkey in keys_to_remove:
            del self[delkey]

        return self


class GoalLikelihoods(GoalMapping[float]):
    """Methods specific for likelihoods"""

    def normalize(self) -> GoalLikelihoods:
        return GoalLikelihoods(valmap(lambda x: x / sum(self.values()), self))

    def initialize_prior(self, distribution: str) -> GoalLikelihoods:
        if distribution.lower() == "uniform":
            goals = self.keys()
            uniform = [1.0 / float(len(goals))] * len(goals)
            return GoalLikelihoods(dict(zip(goals, uniform)))
        elif distribution != "Uniform":
            raise NotImplementedError("The prior distribution you asked for is not implemented yet.")


T = TypeVar("T", GoalRewards, GoalLikelihoods, GoalBools)


class PlayerGoalMapping(Dict[PlayerName, T], Generic[T]):
    """A player keeps track of values for the others' possible goals/resources"""

    def __add__(self, other: PlayerGoalMapping):
        """Add two dictionaries. Can add dictionaries with different keys."""
        assert isinstance(other, PlayerGoalMapping)
        for player in self.keys():
            self[player] = self[player] + other[player]
        for player in other.keys():
            if player not in self.keys():
                self[player] = other[player]
        return self

    def safe_add(self, other: PlayerGoalMapping):
        """Only add dictionaries if they have the same keys."""
        assert isinstance(other, PlayerGoalMapping)
        assert other.keys() == self.keys(), "Safe addition can only be carried through when keys of both dicts " \
                                            "are the identical."
        for player in self.keys():
            self[player] = self[player].safe_add(other[player])
        return self

    def __mul__(self, other: PlayerGoalMapping):
        assert isinstance(other, PlayerGoalMapping)
        assert self.keys() == other.keys()
        for player in self.keys():
            self[player] * other[player]
        return self

    def scalar_multiplication(self, other: float):
        if isinstance(other, int):
            other = float(other)
        assert isinstance(other, float)
        for player in self.keys():
            self[player].scalar_multiplication(other=other)
        return self

    def __sub__(self, other: PlayerGoalMapping):
        assert isinstance(other, PlayerGoalMapping)
        other = other.scalar_multiplication(-1.0)
        return self.__add__(other)

    # fixme: can we merge the three from_lists functions?

    def valfun(self, func: Callable) -> PlayerGoalMapping:
        for player, goal_likelihood in self.items():
            self[player] = self[player].valfun(func=func)
        return PlayerGoalMapping[T](self)


class PlayerGoalLikelihoods(PlayerGoalMapping[GoalLikelihoods]):
    def normalize(self):
        for player in self.keys():
            self[player] = self[player].normalize()
        return PlayerGoalLikelihoods(self)

    def initialize_prior(self, distribution: str):
        for player_id, player in enumerate(self.keys()):
            self[player] = self[player].initialize_prior(distribution=distribution)
        return PlayerGoalLikelihoods(self)

    @staticmethod
    def from_lists(players: List[PlayerName], goals: List[List[int]], init_value: float = -1.0):
        """Initialize a PlayerGoalLikelihoods object from lists of players and goals."""

        assert len(players) == len(goals), "Number of players and number of goal-sets must match."
        temp = {}
        for player_id, player in enumerate(players):
            init_values = [init_value] * len(goals[player_id])
            temp[player] = GoalLikelihoods(dict(zip(goals[player_id], init_values)))

        return PlayerGoalLikelihoods(temp)


class PlayerGoalRewards(PlayerGoalMapping[GoalRewards]):
    """Comment"""

    @staticmethod
    def from_lists(players: List[PlayerName], goals: List[List[int]], init_value: float = -1.0):
        """Initialize a PlayerGoalRewards object from lists of players and goals."""

        assert len(players) == len(goals), "Number of players and number of goal-sets must match."
        temp = {}
        for player_id, player in enumerate(players):
            init_values = [init_value] * len(goals[player_id])
            temp[player] = GoalRewards(dict(zip(goals[player_id], init_values)))

        return PlayerGoalRewards(temp)


class PlayerGoalBools(PlayerGoalMapping[GoalBools]):
    @staticmethod
    def from_lists(players: List[PlayerName], goals: List[List[int]], init_value: bool = False):
        """Initialize a PlayerGoalBools object from lists of players and goals."""

        assert len(players) == len(goals), "Number of players and number of goal-sets must match."
        temp = {}
        for player_id, player in enumerate(players):
            init_values = [init_value] * len(goals[player_id])
            temp[player] = GoalBools(dict(zip(goals[player_id], init_values)))

        return PlayerGoalBools(temp)

    def boolean_intersect(self, other: PlayerGoalBools) -> PlayerGoalBools:
        """ Check if items are matching between two dicts. If they are not, other is considered the ground truth."""
        assert isinstance(other, PlayerGoalBools)
        assert self.keys() == other.keys(), "You are comparing dictionaries with different players."

        for player in self.keys():
            self[player] = self[player].boolean_intersect(other[player])
        return self


class Prediction:
    """Class to handle probabilities, costs and rewards on DynamicGraphs."""

    def __init__(self, players: List[PlayerName], goals: List[List[int]]):
        # prediction parameters
        self.params: PredictionParams = PredictionParams(players=players, goals=goals)

        # dictionary containing information about reachability of each goal by each agent
        self.reachability: PlayerGoalBools = PlayerGoalBools().from_lists(players=players, goals=goals)

        # probabilities
        # dictionary containing probability of each goal for each agent
        self.probabilities: PlayerGoalLikelihoods = PlayerGoalLikelihoods().from_lists(players=players, goals=goals)

        # rewards
        # dictionary containing optimal rewards from current position to goal
        self.suboptimal_reward: PlayerGoalRewards = PlayerGoalRewards().from_lists(players=players, goals=goals)
        # dictionary containing optimal rewards from initial position to goal
        self.optimal_reward: PlayerGoalRewards = PlayerGoalRewards().from_lists(players=players, goals=goals)


class PredictionParams:
    """
    Class for storing prediction parameters
    """

    def __init__(self, players: List[PlayerName], goals: List[List[int]], beta: float = 1.0,
                 distribution: str = "Uniform"):
        self.distribution = distribution
        self.beta = beta
        self.priors: PlayerGoalLikelihoods = PlayerGoalLikelihoods().from_lists(players=players, goals=goals)
        self.priors.initialize_prior(distribution=distribution)


if __name__ == "__main__":
    print("hielo")
