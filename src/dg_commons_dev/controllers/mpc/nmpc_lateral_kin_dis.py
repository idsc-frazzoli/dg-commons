from dg_commons_dev.controllers.utils.discretization_techniques import discretizations
from dg_commons_dev.controllers.mpc.mpc_base_classes.lateral_mpc_base import LatMPCKinBaseParam, LatMPCKinBase
from typing import List, Union
from dataclasses import dataclass
from casadi import *


__all__ = ["NMPCLatKinDis", "NMPCLatKinDisParam"]


@dataclass
class NMPCLatKinDisParam(LatMPCKinBaseParam):
    path_approx_technique: Union[List[str], str] = 'linear'
    """ Path approximation technique """
    dis_technique: Union[List[str], str] = 'Kinematic Euler'
    """ Discretization technique """
    dis_t: Union[List[float], float] = 0.1
    """ Discretization Time Step """


class NMPCLatKinDis(LatMPCKinBase):
    """ Nonlinear MPC for lateral control of vehicle. Kinematic model with prior discretization """

    USE_STEERING_VELOCITY: bool = True

    def __init__(self, params: NMPCLatKinDisParam = NMPCLatKinDisParam()):
        model_type: str = 'discrete'  # either 'discrete' or 'continuous'
        super().__init__(params, model_type)

        assert self.params.dis_technique in discretizations.keys()
        assert self.params.t_step % self.params.dis_t < 10e-10

        f: List[SX] = [self.state_x, self.state_y, self.theta, self.v, self.delta, 0]
        f[-1]: SX = 0 if self.params.analytical else self.s

        for _ in range(int(self.params.t_step/self.params.dis_t)):
            dis_input: List[SX] = [f[0], f[1], f[2], f[3], f[4], f[5], self.v_delta, 0, 0,
                         self.params.vehicle_geometry, self.params.dis_t, self.params.rear_axle]
            dis_input[7]: SX = 0 if self.params.analytical else self.v_s
            f: List[SX] = discretizations[self.params.dis_technique](*dis_input)

        # Set right right hand side of differential equation for x, y, theta, v, delta and s
        self.model.set_rhs('state_x', f[0])
        self.model.set_rhs('state_y', f[1])
        self.model.set_rhs('theta', f[2])
        self.model.set_rhs('v', f[3])
        self.model.set_rhs('delta', f[4])
        if not self.params.analytical:
            self.model.set_rhs('s', f[5])

        self.model.setup()
        self.set_up_mpc()

    def compute_targets(self):
        if self.params.analytical:
            self.path_approx.update_from_parameters(self.path_params)
            return *self.path_approx.closest_point_on_path([self.state_x, self.state_y]), None
        else:
            self.path_approx.update_from_parameters(self.path_params)
            return self.s, self.path_approx.function(self.s), None

    def set_scaling(self):
        self.mpc.scaling['_x', 'state_x'] = 1
        self.mpc.scaling['_x', 'state_y'] = 1
        self.mpc.scaling['_x', 'theta'] = 1
        self.mpc.scaling['_x', 'v'] = 1
        self.mpc.scaling['_x', 'delta'] = 1
        self.mpc.scaling['_u', 'v_delta'] = 1
