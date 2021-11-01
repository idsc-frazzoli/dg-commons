from typing import Tuple
from abc import abstractmethod
from dg_commons_dev.controllers.mpc.mpc_base_classes.lateral_mpc_base import vehicle_params, LatMPCKinBase, LatMPCKinBaseParam
from dg_commons_dev.controllers.utils.cost_functions import *
from dg_commons_dev.controllers.controller_types import LatAndLonController
from dg_commons_dev.utils import BaseParams


@dataclass
class FullMPCKinBaseParam(LatMPCKinBaseParam, BaseParams):
    cost: CostFunctions = QuadraticCost
    """ Cost function """
    cost_params: CostParameters = QuadraticParams(
        q=SemiDef(matrix=np.eye(3)),
        r=SemiDef(matrix=np.eye(2))
    )
    """ Cost function parameters """
    acc_bounds: Tuple[float, float] = vehicle_params.acc_limits
    """ Accelertion bounds """
    v_bounds: Tuple[float, float] = vehicle_params.vx_limits
    """ Velocity Bounds """


class FullMPCKinBase(LatMPCKinBase, LatAndLonController):
    """ MPC for vehicle lateral control abstract class """

    @abstractmethod
    def __init__(self, params, model_type: str):
        super().__init__(params, model_type)
        self.a = self.model.set_variable(var_type='_u', var_name='a')

    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        error = [target_x - self.state_x, target_y - self.state_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]
        lterm, _ = self.cost.cost_function(error, inp)
        return lterm

    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        error = [target_x - self.state_x, target_y - self.state_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]
        _, mterm = self.cost.cost_function(error, inp)
        return mterm

    def set_bounds(self):
        super().set_bounds()
        self.mpc.bounds['lower', '_x', 'v'] = self.params.v_bounds[0]
        self.mpc.bounds['upper', '_x', 'v'] = self.params.v_bounds[1]
        self.mpc.bounds['lower', '_u', 'a'] = self.params.acc_bounds[0]
        self.mpc.bounds['upper', '_u', 'a'] = self.params.acc_bounds[1]

    def _update_reference_speed(self, speed_ref: float):
        self.speed_ref = speed_ref

    def _get_acceleration(self, at: float) -> float:
        if any([_ is None for _ in [self.path]]):
            raise RuntimeError("Attempting to use MPC before having set any observations or reference path")
        try:
            return self.u[2][0]
        except IndexError:
            return self.u[1][0]
