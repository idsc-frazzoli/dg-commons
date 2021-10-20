from abc import abstractmethod, ABC
from typing import Callable, Set, Optional, Union

from dg_commons import Timestamp
from dg_commons.planning.trajectory import Trajectory, TrajectoryGraph
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands
from dg_commons.sim.models.vehicle_utils import VehicleParameters


class TrajGenerator(ABC):
    def __init__(
        self,
        vehicle_dynamics: Callable[[VehicleState, VehicleCommands, Timestamp], VehicleState],
        vehicle_param: VehicleParameters,
    ):
        self.vehicle_dynamics = vehicle_dynamics
        self.vehicle_param = vehicle_param

    @abstractmethod
    def generate(self, x0: Optional[VehicleState]) -> Union[Set[Trajectory], TrajectoryGraph]:
        """Passing the current initial state is optional"""
        pass
