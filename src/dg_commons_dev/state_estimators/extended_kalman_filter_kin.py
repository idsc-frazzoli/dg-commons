import numpy as np
from scipy.integrate import solve_ivp
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands, VehicleGeometry
from dg_commons.sim.models.vehicle_utils import steering_constraint, VehicleParameters
from dg_commons.sim.models.model_utils import acceleration_constraint
from typing import Optional, Union
from dg_commons_dev.utils import SemiDef
from dg_commons_dev.state_estimators.dropping_trechniques import DroppingTechniques, \
    LGB, LGBParam, DroppingMaps
from typing import List
from dg_commons_dev.state_estimators.estimator_types import Estimator
import math
from dg_commons import X
from dataclasses import dataclass
from dg_commons_dev.utils import BaseParams


@dataclass
class ExtendedKalmanKinParam(BaseParams):
    n_states: Union[List[int], int] = VehicleState.get_n_states()
    """ Number of states """
    n_commands: Union[List[int], int] = VehicleCommands.get_n_commands()
    """ Number of commands """
    actual_model_var: Union[List[SemiDef], SemiDef] = SemiDef(n_states * [0])
    """ Actual Modeling covariance matrix """
    actual_meas_var: Union[List[SemiDef], SemiDef] = SemiDef(n_states * [0])
    """ Actual Measurement covariance matrix """
    belief_model_var: Union[List[SemiDef], SemiDef] = actual_model_var
    """ Belief modeling covariance matrix """
    belief_meas_var: Union[List[SemiDef], SemiDef] = actual_meas_var
    """ Belief measurement covariance matrix """
    initial_variance: Union[List[SemiDef], SemiDef] = actual_model_var
    """ Initial covariance matrix """
    dropping_technique: Union[List[type(DroppingTechniques)], type(DroppingTechniques)] = LGB
    """ Dropping Technique """
    dropping_params: Union[List[BaseParams], BaseParams] = LGBParam()
    """ Dropping parameters """
    geometry_params: Union[List[VehicleGeometry], VehicleGeometry] = VehicleGeometry.default_car()
    """ Vehicle Geometry """
    vehicle_params: Union[List[VehicleParameters], VehicleParameters] = VehicleParameters.default_car()
    """ Vehicle Parameters """
    t_step: Union[List[float], float] = 0.1
    """ Time interval between two calls """

    def __post_init__(self):
        if isinstance(self.dropping_params, list):
            assert len(self.dropping_technique) == len(self.dropping_params)
            for i, technique in enumerate(self.dropping_technique):
                assert DroppingMaps[technique] == type(self.dropping_params[i])

        else:
            assert DroppingMaps[self.dropping_technique] == type(self.dropping_params)

        super().__post_init__()


class ExtendedKalmanKin(Estimator):
    """ Extended Kalman Filter with kinematic bicycle model and identity measurement model """

    def __init__(self, x0=None, params=ExtendedKalmanKinParam(),
                 model_noise_realization=None):
        self.params: ExtendedKalmanKinParam = params

        self.actual_model_noise: np.ndarray = params.actual_model_var.matrix
        self.actual_meas_noise: np.ndarray = params.actual_meas_var.matrix
        self.belief_model_noise: np.ndarray = params.belief_model_var.matrix
        self.belief_meas_noise: np.ndarray = params.belief_meas_var.matrix
        self.p: np.ndarray = params.initial_variance.matrix

        self.dropping: DroppingTechniques = params.dropping_technique(params.dropping_params)

        self.state: X = x0
        self.dt: float = self.params.t_step
        self.current_model_noise_realization: X = model_noise_realization

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

        if not self.dropping.drop():
            h = self._h(self.state)
            state = self.state.as_ndarray().reshape((n_states, 1))
            # Perturb measurement and reshape
            measurement_k = measurement_k + ExtendedKalmanKin._realization(self.actual_meas_noise)
            meas = measurement_k.as_ndarray().reshape((n_states, 1))

            try:
                self.p = self.p.astype(float)
                helper = np.linalg.inv(np.matmul(np.matmul(h, self.p), h.T) + self.belief_meas_noise)
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
            dp = np.matmul(f, p) + np.matmul(p, f.T) + self.belief_model_noise

            return np.concatenate([dx.as_ndarray(), du, mat_to_vec(dp)])

        self.current_model_noise_realization = self._realization(self.actual_model_noise)
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
        acc = acceleration_constraint(x0.vx, u.acc, self.params.vehicle_params)
        return VehicleState(x=xdot, y=ydot, theta=dtheta, vx=acc, delta=ddelta) + self.current_model_noise_realization

    @staticmethod
    def _realization(var: np.ndarray):
        dim = int(var.shape[0])
        return VehicleState.from_array(np.random.multivariate_normal(np.zeros(dim), var))
