from typing import Union, List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import do_mpc
from dataclasses import dataclass
from dg_commons_dev.controllers.utils.cost_functions import CostFunctions, QuadraticCost, CostParameters, \
    MapCostParam, QuadraticParams
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons_dev.utils import BaseParams
from dg_commons_dev.controllers.interface import Controller
from casadi import *


@dataclass
class MPCKinBAseParam(BaseParams):
    n_horizon: Union[List[int], int] = 15
    """ Horizon Length """
    t_step: Union[List[float], float] = 0.1
    """ Sample Time """
    cost: Union[List[CostFunctions], CostFunctions] = QuadraticCost
    """ Cost function """
    cost_params: Union[List[CostParameters], CostParameters] = QuadraticParams()
    """ Cost function parameters """
    delta_input_weight: Union[List[float], float] = 1e-2
    """ Weighting factor in cost function for varying input """
    rear_axle: Union[List[bool], bool] = False
    """ Whether to control rear axle position instead of cog """
    vehicle_geometry: Union[List[VehicleGeometry], VehicleGeometry] = VehicleGeometry.default_car()

    def __post_init__(self):
        if isinstance(self.cost, list):
            assert len(self.cost) == len(self.cost_params)
            for i, technique in enumerate(self.cost):
                assert MapCostParam[technique] == type(self.cost_params[i])

        else:
            assert MapCostParam[self.cost] == type(self.cost_params)

        super().__post_init__()


class MPCKinBase(Controller, ABC):
    """ MPC for vehicle control abstract class """

    @abstractmethod
    def __init__(self, params, model_type: str):
        self.params = params
        self.setup_mpc: Dict[str, Any] = {
            'n_horizon': self.params.n_horizon,
            't_step': self.params.t_step,
            'store_full_solution': True,
        }

        self.model: do_mpc.model = do_mpc.model.Model(model_type)
        self.state_x: SX = self.model.set_variable(var_type='_x', var_name='state_x', shape=(1, 1))
        self.state_y: SX = self.model.set_variable(var_type='_x', var_name='state_y', shape=(1, 1))
        self.theta: SX = self.model.set_variable(var_type='_x', var_name='theta', shape=(1, 1))
        self.v: SX = self.model.set_variable(var_type='_x', var_name='v', shape=(1, 1))
        self.delta: SX = self.model.set_variable(var_type='_x', var_name='delta', shape=(1, 1))
        self.target_speed: SX = self.model.set_variable(var_type='_tvp', var_name='target_speed', shape=(1, 1))
        self.speed_ref: float = 0

        self.mpc: do_mpc.controller.MPC = do_mpc.controller.SX_nan()
        self.tvp_temp: SX = do_mpc.controller.SX_nan()

    def __post_init__(self):
        """
        Ensures that the mpc is set up in the __init__ method
        """
        assert self.mpc is not do_mpc.controller.SX_nan()

    def set_up_mpc(self)->None:
        """
        This method sets up the mpc and needs to be called in the inheriting __init__ method after the model setup
        """

        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_param(**self.setup_mpc)
        suppress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        self.mpc.set_param(nlpsol_opts=suppress_ipopt)
        target_x, target_y, target_angle = self.compute_targets()

        lterm = self.lterm(target_x, target_y, self.target_speed)
        mterm = self.mterm(target_x, target_y, self.target_speed)

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.set_rterm(
            v_delta=self.params.delta_input_weight
        )

        self.set_bounds()
        self.set_scaling()

        self.tvp_temp = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.func)

        self.mpc.setup()

    def func(self, t_now) -> SX:
        """
        Sets up the time varying parameter speed ref. Might be overwritten
        in case of additional time varying parameters
        @param t_now: current time instant
        @return: time varying parameters
        """
        self.tvp_temp['_tvp', :] = np.array([self.speed_ref])
        return self.tvp_temp

    @abstractmethod
    def lterm(self, target_x, target_y, speed_ref, target_angle=None) -> SX:
        """ The lterm needs to be implemented. This is the stage cost """
        pass

    @abstractmethod
    def mterm(self, target_x, target_y, speed_ref, target_angle=None) -> SX:
        """ The mterm needs to be implemented. This is the terminal cost """
        pass

    @abstractmethod
    def set_bounds(self) -> None:
        """ Method to set both the state and the input bounds. Can be empty. """
        pass

    @abstractmethod
    def set_scaling(self) -> None:
        """ States and inputs might be rescaled. Can be empty. """
        pass

    @abstractmethod
    def compute_targets(self) -> Tuple[SX, ...]:
        """ This method returns the target state. """
        pass
