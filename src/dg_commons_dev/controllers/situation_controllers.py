from dg_commons_dev.controllers.controller_types import Reference
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from typing import Optional, Callable
from dg_commons_dev.controllers.interface import Controller
from dg_commons_dev.behavior.emergency import EmergencyDescription
from dg_commons_dev.behavior.yield_to import YieldDescription
from dataclasses import dataclass
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons_dev.utils import BaseParams


@dataclass
class OptionalCommands:
    """
    This dataclass is used to obtain the following situation:
        if optional_command_i is None:
            command_i = command_i that would have been used in a cruise situation
        else:
            command_i = optional_command_i

    This leaves the situation controller the freedom of affecting only some of the commands.
    """
    acc: Optional[float] = None
    ddelta: Optional[float] = None

    def is_normal_required(self) -> bool:
        """
        Check whether cruise commands need to be computed
        @return: True if they need to be computed, False otherwise
        """
        return self.acc is None or self.ddelta is None

    def get_commands(self, normal_commands: VehicleCommands) -> VehicleCommands:
        """
        Replaces normal commands with optional commands if optional command is not None.
        @param normal_commands: Commands that would have been used in a cruise situation
        @return: Commands for the vehicle
        """
        if self.acc is not None:
            normal_commands.acc = self.acc
        if self.ddelta is not None:
            normal_commands.ddelta = self.ddelta
        return normal_commands

    def to_vehicle_commands(self) -> VehicleCommands:
        """
        Convert OptionalCommand to VehicleCommand if this is possible
        @return: VehicleCommand with same attributes as self
        """
        assert self.acc is not None and self.ddelta is not None
        return VehicleCommands(acc=self.acc, ddelta=self.ddelta)


@dataclass
class EmergencyControllerParams(BaseParams):
    """ Emergency controller parameters """

    vehicle_params: VehicleParameters = VehicleParameters.default_car()
    """ Vehicle parameters """


class EmergencyController(Controller[Reference, EmergencyDescription, OptionalCommands]):
    """ Emergency controller implements how to react to a certain emergency situation in terms of acc and ddelta """
    REF_PARAMS: Callable = EmergencyControllerParams

    def __init__(self, params=EmergencyControllerParams(), ref: Reference = None):
        self.params = params
        self.ref: Reference = ref

    def update_ref(self, new_ref: Reference):
        """
        The reference is updated
        @param new_ref: New controller reference
        """
        self.ref = new_ref

    def control(self, new_obs: EmergencyDescription, t: float) -> OptionalCommands:
        """
        A new observation is processed and an input for the system formulated and returned
        @param new_obs: New Observation
        @param t: Current time instant
        @return: optional commands
        """
        assert self.ref is not None

        commands = OptionalCommands(acc=self.params.vehicle_params.acc_limits[0])
        return commands


@dataclass
class YieldControllerParams(BaseParams):
    """ Yield controller parameters """

    vehicle_params: VehicleParameters = VehicleParameters.default_car()
    """ Vehicle parameters """


class YieldController(Controller[Reference, YieldDescription, OptionalCommands]):
    """ Yield controller implements how to react to a certain yield situation in terms of acc and ddelta """
    REF_PARAMS: Callable = YieldControllerParams

    def __init__(self, params=YieldControllerParams()):
        self.params = params
        self.ref: Optional[Reference] = None

    def update_ref(self, new_ref: Reference):
        """
        The reference is updated
        @param new_ref: New controller reference
        """
        self.ref = new_ref

    def control(self, new_obs: YieldDescription, t: float) -> OptionalCommands:
        """
        A new observation is processed and an input for the system formulated and returned
        @param new_obs: New Observation
        @param t: Current time instant
        @return: optional commands
        """
        assert self.ref is not None

        commands = OptionalCommands(acc=self.params.vehicle_params.acc_limits[0])
        return commands
