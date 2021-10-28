from dg_commons_dev.controllers.mpc.full_mpc_base import FullMPCKinBaseParam, FullMPCKinBase
from dg_commons_dev.controllers.discretization_techniques import discretizations
from typing import List, Union
from dataclasses import dataclass
from casadi import *


__all__ = ["NMPCFullKinDis", "NMPCFullKinDisParam"]


@dataclass
class NMPCFullKinDisParam(FullMPCKinBaseParam):
    dis_technique: Union[List[str], str] = 'Kinematic Euler'
    """ Discretization technique """
    dis_t: Union[List[float], float] = 0.01
    """ Discretization Time Step """


class NMPCFullKinDis(FullMPCKinBase):
    """ Nonlinear MPC for full control of vehicle. Kinematic model with prior discretization """

    USE_STEERING_VELOCITY: bool = True

    def __init__(self, params: NMPCFullKinDisParam = NMPCFullKinDisParam()):
        model_type: str = 'discrete'  # either 'discrete' or 'continuous'
        super().__init__(params, model_type)

        assert self.params.dis_technique in discretizations.keys()
        assert self.params.t_step % self.params.dis_t < 10e-10

        f: List[SX] = [self.state_x, self.state_y, self.theta, self.v, self.delta, 0]
        f[-1] = 0 if self.params.analytical else self.s

        for _ in range(int(self.params.t_step / self.params.dis_t)):
            dis_input: List[SX] = [f[0], f[1], f[2], f[3], f[4], f[5], self.v_delta, 0, self.a,
                         self.params.vehicle_geometry, self.params.dis_t, self.params.rear_axle]
            dis_input[7]: SX = 0 if self.params.analytical else self.v_s
            f: List[SX] = discretizations[self.params.dis_technique](*dis_input)

        self.model.set_rhs('state_x', f[0])
        self.model.set_rhs('state_y', f[1])
        self.model.set_rhs('theta', f[2])
        self.model.set_rhs('v', f[3])
        self.model.set_rhs('delta', f[4])
        if not self.params.analytical:
            self.model.set_rhs('s', f[5])
        """ Set right right hand side of difference equation for x, y, theta, v, delta and s """

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
        self.mpc.scaling['_x', 'state_x']: float = 1
        self.mpc.scaling['_x', 'state_y']: float = 1
        self.mpc.scaling['_x', 'theta']: float = 1
        self.mpc.scaling['_x', 'v']: float = 1
        self.mpc.scaling['_x', 'delta']: float = 1
        self.mpc.scaling['_u', 'v_delta']: float = 1
        self.mpc.scaling['_u', 'a']: float = 1
