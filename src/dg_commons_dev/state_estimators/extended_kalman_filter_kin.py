import numpy as np
from scipy.integrate import solve_ivp
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands, VehicleGeometry
from dg_commons.sim.models.vehicle_utils import steering_constraint, VehicleParameters
from dg_commons.sim.models.model_utils import apply_full_acceleration_limits
from typing import Optional
from dg_commons_dev.utils import SemiDef
from dg_commons_dev.state_estimators.estimator_types import Estimator
import math
from dg_commons import X
from dataclasses import dataclass
from dg_commons_dev.utils import BaseParams
from dg_commons_dev.state_estimators.dropping_trechniques import DroppingTechniques, LGB


@dataclass
class ExtendedKalmanKinParam(BaseParams):
    n_states: int = VehicleState.get_n_states()
    """ Number of states """
    n_commands: int = VehicleCommands.get_n_commands()
    """ Number of commands """
    model_covariance: SemiDef = SemiDef(n_states * [0])
    """ Modeling covariance matrix """
    meas_covariance: SemiDef = SemiDef(n_states * [0])
    """ Measurement covariance matrix """
    initial_variance: SemiDef = meas_covariance
    """ Initial covariance matrix """
    geometry_params: VehicleGeometry = VehicleGeometry.default_car()
    """ Vehicle Geometry """
    vehicle_params: VehicleParameters = VehicleParameters.default_car()
    """ Vehicle Parameters """
    t_step: float = 0.1
    """ Time interval between two calls """
    dropping_techniques: DroppingTechniques = LGB()
    """ Dropping technique """

    def __post_init__(self):
        assert len(self.model_covariance.eig) == self.n_states
        assert len(self.meas_covariance.eig) == self.n_states
        assert len(self.initial_variance.eig) == self.n_states
        assert 0 <= self.t_step <= 30


class ExtendedKalmanKin(Estimator):
    """ Extended Kalman Filter with kinematic bicycle model and identity measurement model """
    REF_PARAMS: dataclass = ExtendedKalmanKinParam

    def __init__(self, x0=None, params=ExtendedKalmanKinParam()):
        self.params: ExtendedKalmanKinParam = params

        self.model_covariance: np.ndarray = params.model_covariance.matrix
        self.meas_covariance: np.ndarray = params.meas_covariance.matrix
        self.p: np.ndarray = params.initial_variance.matrix

        self.state: X = x0
        self.dt: float = self.params.t_step
        self.current_model_noise_realization: X = None

    def update_prediction(self, u_k: Optional[VehicleCommands]) -> None:
        """
        Internal vehicle state and covariance matrix get projected ahead by params.dt using the kinematic bicycle model
        and the input to the system u_k.
        @param u_k: Vehicle input k
        @return: None
        """
        if self.state is None or u_k is None:
            return

        self.state, self.p = self._solve_dequation(u_k)

    def update_measurement(self, measurement_k: X) -> None:
        """
        Internal vehicle state and covariance matrix get updated based on identity measurement model and on measurement
        k.
        @param measurement_k: kth system measurement
        @return: None
        """
        n_states = self.params.n_states
        if self.state is None:
            self.state = measurement_k
            return

        if measurement_k is not None:
            h = self._h(self.state)
            state = self.state.as_ndarray().reshape((n_states, 1))
            meas = measurement_k.as_ndarray().reshape((n_states, 1))

            try:
                self.p = self.p.astype(float)
                helper = np.linalg.inv(np.matmul(np.matmul(h, self.p), h.T) + self.meas_covariance)
                k = np.matmul(np.matmul(self.p, h.T), helper)
                state = state + np.matmul(k, (meas - state))

                self.state = VehicleState.from_array(np.matrix.flatten(state))
                self.p = np.matmul(np.eye(n_states)-np.matmul(k, h), self.p)
            except np.linalg.LinAlgError:
                pass

    def _solve_dequation(self, u_k: VehicleCommands):
        n_states: int = self.params.n_states
        n_commands: int = self.params.n_commands

        def vec_to_mat(v):
            return v.reshape(n_states, n_states)

        def mat_to_vec(mat):
            return np.matrix.flatten(mat)

        def _stateactions_from_array(state_input: np.ndarray) -> [VehicleState, VehicleCommands]:
            state = VehicleState.from_array(state_input[0:n_states])
            actions = VehicleCommands(acc=state_input[VehicleCommands.idx["acc"] + n_states],
                                      ddelta=state_input[VehicleCommands.idx["ddelta"] + n_states])
            return state, actions

        def _dynamics(t, y):
            part1, part2 = y[:(n_states + n_commands)], y[(n_states + n_commands):]
            state0, actions = _stateactions_from_array(state_input=part1)
            dx = self._dynamics(x0=state0, u=actions)
            du = np.zeros([len(VehicleCommands.idx)])

            f = self._f(state0)
            p = vec_to_mat(part2)
            dp = np.matmul(f, p) + np.matmul(p, f.T) + self.model_covariance

            return np.concatenate([dx.as_ndarray(), du, mat_to_vec(dp)])

        state_zero = np.concatenate([self.state.as_ndarray(), u_k.as_ndarray(), mat_to_vec(self.p)])
        result = solve_ivp(fun=_dynamics, t_span=(0.0, float(self.dt)), y0=state_zero)
        if not result.success:
            raise RuntimeError("Failed to integrate ivp!")

        result_values = result.y[:, -1]
        part1, part2 = result_values[:(n_states + n_commands)], result_values[(n_states + n_commands):]
        new_state, _ = _stateactions_from_array(state_input=part1)
        new_p = vec_to_mat(part2)

        return new_state, new_p

    def _f(self, state):
        l = self.params.geometry_params.length
        lr = self.params.geometry_params.lr
        s_t = math.sin(state.theta)
        c_t = math.cos(state.theta)
        t_d = math.tan(state.delta)
        c_d = math.cos(state.delta)
        v = state.vx
        return np.array([[0, 0, -v*s_t-c_t*t_d*lr*v/l, c_t-t_d*s_t*lr/l, -v*s_t*lr/(l*c_d**2)],
                         [0, 0,  v*c_t-s_t*t_d*lr*v/l, s_t+t_d*c_t*lr/l, v*c_t*lr/(l*c_d**2)],
                         [0, 0, 0, t_d/l, v/(l*c_d**2)],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]])

    def _h(self, state):
        return np.eye(5)

    def _dynamics(self, x0: VehicleState, u: VehicleCommands) -> VehicleState:
        """ Kinematic bicycle model, returns state derivative for given control inputs """
        l = self.params.geometry_params.length
        lr = self.params.geometry_params.lr

        vx = x0.vx
        dtheta = vx * math.tan(x0.delta) / l
        vy = dtheta * lr
        costh = math.cos(x0.theta)
        sinth = math.sin(x0.theta)
        xdot = vx * costh - vy * sinth
        ydot = vx * sinth + vy * costh

        ddelta = steering_constraint(x0.delta, u.ddelta, self.params.vehicle_params)
        acc = apply_full_acceleration_limits(x0.vx, u.acc, self.params.vehicle_params)
        return VehicleState(x=xdot, y=ydot, theta=dtheta, vx=acc, delta=ddelta)
