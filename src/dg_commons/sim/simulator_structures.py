from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Generic, Any, Dict, Mapping, Tuple, Optional

from geometry import SE2value, T2value
from shapely.geometry import Polygon
from zuper_commons.types import ZValueError

from dg_commons import DgSampledSequence, PlayerName, X, U
from dg_commons.sim.goals import PlanningGoal
from dg_commons.seq.sequence import DgSampledSequenceBuilder, Timestamp, UndefinedAtTime
from dg_commons.sim import SimTime, ImpactLocation, logger
from dg_commons.sim.models.model_structures import ModelGeometry, ModelType, ModelParameters

__all__ = [
    "SimObservations",
    "InitSimObservations",
    "SimParameters",
    "SimModel",
    "SimLog",
    "PlayerLog",
    "LogEntry",
    "PlayerLogger",
    "PlayerObservations",
]

from dg_commons.sim.scenarios import DgScenario


@dataclass(frozen=True)
class SimParameters:
    dt: SimTime = SimTime("0.05")
    """Simulation step [s]"""
    dt_commands: SimTime = SimTime("0.1")
    """How often shall we ask the agents for new commands [s]"""
    max_sim_time: SimTime = SimTime(6)
    """Max Simulation time overall [s]"""
    sim_time_after_collision: SimTime = SimTime(0)
    """The simulation time for which to continue after the first collision is detected [s]"""


@dataclass(frozen=True)
class PlayerObservations:
    state: X
    occupancy: Optional[Polygon]


@dataclass(frozen=True)
class SimObservations:
    """The observations from the simulator passed to each agent"""

    players: Mapping[PlayerName, PlayerObservations]
    time: SimTime


@dataclass(frozen=True)
class InitSimObservations:
    """The observations passed to the simulator on episode init"""

    my_name: PlayerName
    seed: int
    dg_scenario: Optional[DgScenario] = None
    goal: Optional[PlanningGoal] = None


@dataclass(frozen=True)
class LogEntry:
    """A log entry for a player"""

    state: X
    """Player's state"""
    commands: U
    """Commands of the player"""
    extra: Any
    """Extra object for rendering afterwards"""
    info: float
    """For now it is only the computation time"""


@dataclass
class PlayerLog:
    """A log for a player"""

    states: DgSampledSequence[X]
    commands: DgSampledSequence[U]
    extra: DgSampledSequence[Any]
    info: DgSampledSequence[float]

    def at_interp(self, t: Timestamp) -> LogEntry:
        """State gets interpolated, commands and extra not."""
        try:
            extra = self.extra.at_or_previous(t)
        except (UndefinedAtTime, ZValueError):
            extra = None

        return LogEntry(
            state=self.states.at_interp(t),
            commands=self.commands.at_or_previous(t),
            extra=extra,
            info=self.info.at_or_previous(t),
        )


@dataclass
class PlayerLogger(Generic[X, U]):
    """The logger of a player that builds the log"""

    states: DgSampledSequenceBuilder[X] = field(default_factory=DgSampledSequenceBuilder[X])
    commands: DgSampledSequenceBuilder[U] = field(default_factory=DgSampledSequenceBuilder[U])
    extra: DgSampledSequenceBuilder[Any] = field(default_factory=DgSampledSequenceBuilder[Any])
    info: DgSampledSequenceBuilder[float] = field(default_factory=DgSampledSequenceBuilder[float])

    def as_sequence(
        self,
    ) -> PlayerLog:
        return PlayerLog(
            states=self.states.as_sequence(),
            commands=self.commands.as_sequence(),
            extra=self.extra.as_sequence(),
            info=self.info.as_sequence(),
        )


class SimLog(Dict[PlayerName, PlayerLog]):
    """The logger for a simulation. For each player it records sampled sequences of states, commands and extra
    arguments than an agent might want to log."""

    def __setitem__(self, key, value):
        if not isinstance(value, PlayerLog):
            raise ZValueError("Invalid value for PlayerLog", value=value, required_type=PlayerLog)
        super(SimLog, self).__setitem__(key, value)

    def at_interp(self, t: Timestamp) -> Mapping[PlayerName, LogEntry]:
        interpolated_entry: Dict[PlayerName, LogEntry] = {}
        for player in self:
            interpolated_entry[player] = self[player].at_interp(t)
        return interpolated_entry

    def get_init_time(self) -> SimTime:
        return min([self[p].states.get_start() for p in self])

    def get_last_time(self) -> SimTime:
        return max([self[p].states.get_end() for p in self])


class SimModel(ABC, Generic[X, U]):
    _state: X
    """State of the model"""
    # XT: Type[X] = object
    """Type of the state"""
    has_collided: bool = False
    """Whether or not the object has already collided"""

    @abstractmethod
    def update(self, commands: U, dt: SimTime):
        """The model gets updated via this function"""
        pass

    @abstractmethod
    def get_footprint(self) -> Polygon:
        """This returns the footprint of the model that is used for collision checking"""
        pass

    @abstractmethod
    def get_pose(self) -> SE2value:
        """Return pose of the model"""
        pass

    @abstractmethod
    def get_velocity(self, in_model_frame: bool) -> (T2value, float):
        """Get velocity of the model
        :param in_model_frame: whether in body frame, or global frame"""
        pass

    @abstractmethod
    def set_velocity(self, vel: T2value, omega: float, in_model_frame: bool):
        """Set velocity of the model
        :param vel:
        :param omega:
        :param in_model_frame: If the passed value are already in body frame (True) or global (False)
        """
        pass

    @abstractmethod
    def get_mesh(self) -> Mapping[ImpactLocation, Polygon]:
        pass

    @property
    @abstractmethod
    def model_geometry(self) -> ModelGeometry:
        pass

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        pass

    @property
    @abstractmethod
    def model_params(self) -> ModelParameters:
        pass

    def get_state(self) -> X:
        return deepcopy(self._state)

    def set_state(self, new_state: X):
        logger.warn("Setting a new state to simulation model, this is a dangerous operation")
        self._state = new_state

    @abstractmethod
    def get_extra_collision_friction_acc(
        self,
    ) -> Tuple[float, float, float]:
        pass
