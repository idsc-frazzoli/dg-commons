from dg_commons_dev.controllers.mpc.full_mpc_base import FullMPCKinBase, FullMPCKinBaseParam
from casadi import *
from dataclasses import dataclass


__all__ = ["NMPCFullKinCont", "NMPCFullKinContParam"]


@dataclass
class NMPCFullKinContParam(FullMPCKinBaseParam):
    pass


class NMPCFullKinCont(FullMPCKinBase):
    """ Nonlinear MPC for full control of vehicle. Kinematic model without prior discretization """

    USE_STEERING_VELOCITY: bool = True

    def __init__(self, params: NMPCFullKinContParam = NMPCFullKinContParam()):
        model_type: str = 'continuous'  # either 'discrete' or 'continuous'
        super().__init__(params, model_type)

        # Set right right hand side of differential equation for x, y, theta, v, delta and s
        dtheta: SX = self.v * tan(self.delta) / self.params.vehicle_geometry.length
        if self.params.rear_axle:
            self.model.set_rhs('state_x', cos(self.theta) * self.v)
            self.model.set_rhs('state_y', sin(self.theta) * self.v)
        else:
            vy: SX = dtheta * self.params.vehicle_geometry.lr
            self.model.set_rhs('state_x', self.v * cos(self.theta) - vy * sin(self.theta))
            self.model.set_rhs('state_y', self.v * sin(self.theta) + vy * cos(self.theta))

        self.model.set_rhs('theta', dtheta)
        self.model.set_rhs('v', self.a)
        self.model.set_rhs('delta', self.v_delta)
        if not self.params.analytical:
            self.model.set_rhs('s', self.v_s)

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
