from dg_commons.sim.models.vehicle import VehicleModel, TVehicleState
import numpy as np
from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands, VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons_dev.utils import SemiDef
from dg_commons_dev.state_estimators.dropping_trechniques import DroppingTechniques


class NoisyVehicleModel(VehicleModel):
    def __init__(self, x0: TVehicleState, vg: VehicleGeometry, vp: VehicleParameters,
                 model_covariance: SemiDef = None, meas_covariance: SemiDef = None,
                 dropping_technique: DroppingTechniques = None):
        super().__init__(x0, vg, vp)
        n_states = x0.get_n_states()
        self.model_covariance: SemiDef = SemiDef(n_states * [0]) if model_covariance is None else model_covariance
        self.meas_covariance: SemiDef = SemiDef(n_states * [0]) if meas_covariance is None else meas_covariance
        self.dropping_technique: DroppingTechniques = dropping_technique
        self.model_noise = self._realization(self.model_covariance.matrix)

    def dynamics(self, x0: VehicleState, u: VehicleCommands) -> VehicleState:
        """ Noisy kinematic bicycle model, returns state derivative for given control inputs """
        exact_dynamics = super().dynamics(x0, u)
        return exact_dynamics + self.model_noise

    def state_measurement(self):
        self.model_noise = self._realization(self.model_covariance.matrix)
        if self.dropping_technique is None or (not self.dropping_technique.drop()):
            meas_noise_realization = self._realization(self.meas_covariance.matrix)
            state = self.get_state()
            return state + meas_noise_realization
        else:
            return None

    @staticmethod
    def _realization(var: np.ndarray):
        dim = int(var.shape[0])
        return VehicleState.from_array(np.random.multivariate_normal(np.zeros(dim), var))
