import random
from dataclasses import dataclass
from typing import Union, List, Dict, Tuple
import scipy
import numpy as np
from dg_commons_dev.state_estimators.utils import PDistribution, PDistributionParams, ExponentialParams, Exponential
from dg_commons_dev.state_estimators.estimator_types import DroppingTechniques
from dg_commons_dev.utils import BaseParams


@dataclass
class LGBParam(BaseParams):
    """ Linear Gaussian Bernoulli Parameters """

    failure_p: float = 0
    """ Failure Probability """

    def __post_init__(self):
        if isinstance(self.failure_p, list):
            all(1 >= i >= 0 for i in self.failure_p)
        else:
            assert 1 >= self.failure_p >= 0
        super().__post_init__()


class LGB(DroppingTechniques):
    """ Linear Gaussian Bernoulli """

    def __init__(self, params=LGBParam()):
        self.params = params
        self.steps = 0
        self.counter = 0

    def drop(self) -> bool:
        """
        Random sampling from Bernoulli distribution with parameter params.failure_p.
        P(failure) = params.failure_p
        @return: True if the measurement is lost, False otherwise
        """
        val = random.uniform(0, 1)

        return_val = False
        if val <= self.params.failure_p:
            return_val = True

        current_state = 0 if return_val else 1
        self.counter += current_state
        self.steps += 1

        return return_val

    def mean(self) -> float:
        """
        Statistical mean
        @return: The statistical mean
        """
        return self.counter/self.steps


@dataclass
class LGMParam(LGBParam):
    """ Linear Gaussian Markov Parameters """

    recovery_p: float = 0
    """ Recovery Probability """

    def __post_init__(self):
        super().__post_init__()
        f_list = isinstance(self.failure_p, list)
        r_list = isinstance(self.recovery_p, list)

        temp_f = self.failure_p if f_list else [self.failure_p]
        temp_r = self.recovery_p if r_list else [self.recovery_p]
        self.expected_value = []
        for i in temp_f:
            for j in temp_r:
                self.expected_value.append(self.process_instance(i, j))
        super().__post_init__()

    @staticmethod
    def process_instance(f, r) -> float:
        """
        The expected value of the Markov distribution with parameters params.failure_p, params.recovery_p.
        P(failure | was working at previous time step) = params.failure_p
        P(recovery | was failed at previous time step) = params.recovery_p
        @param f: failure probability
        @param r: recovery probability
        @return: expected value
        """
        assert 1 >= r >= 0

        mat = np.array([[1-f, f], [r, 1-r]])
        eigval, eigvecl, eigvecr = scipy.linalg.eig(mat, left=True)

        steady_mat = np.matmul(np.matmul(eigvecl, np.diag([0 if eig < 1 else 1 for eig in eigval])),
                               scipy.linalg.inv(eigvecl))
        steady_values = np.matmul(steady_mat, np.array([[1], [0]]))
        expected_value = steady_values[0]*1 + 0*steady_values[1]
        return expected_value


class LGM(DroppingTechniques):
    """ Linear Gaussian Markov """

    def __init__(self, params=LGMParam()):
        self.params = params
        self.current_state = 1
        self.counter = 0
        self.steps = 0

    def drop(self) -> bool:
        """
        Random sampling from Markov distribution with parameters params.failure_p, params.recovery_p:
        P(failure | was working at previous time step) = params.failure_p
        P(recovery | was failed at previous time step) = params.recovery_p
        @return: True if the measurement is lost, False otherwise
        """
        val = random.uniform(0, 1)

        return_val = False
        if self.current_state == 1:
            if val <= self.params.failure_p:
                self.current_state = 0
                return_val = True
        else:
            if val <= self.params.recovery_p:
                self.current_state = 1
            else:
                return_val = True

        self.counter += self.current_state
        self.steps += 1

        return return_val

    def mean(self) -> float :
        """
        Statistical Mean
        @return: The statistical mean
        """
        return self.counter/self.steps


@dataclass
class LGSMParam(BaseParams):
    """ Linear Gaussian Semi - Markov parameters """

    failure_distribution: type(PDistribution) = Exponential
    """ Failure probability distribution """
    failure_params: PDistributionParams = ExponentialParams()
    """ Failure probability distribution parameters """

    recovery_distribution: type(PDistribution) = Exponential
    """ Recovery probability distribution """
    recovery_params: PDistributionParams = ExponentialParams()
    """ Recovery probability distribution parameters """

    dt: float = 0.1
    """ Time interval between two subsequent calls """


class LGSM(DroppingTechniques):
    """ Linear Gaussian Semi - Markov """

    def __init__(self, params=LGSMParam()):
        self.dt = params.dt
        self.failure_distribution = params.failure_distribution(params.failure_params)
        self.recovery_distribution = params.recovery_distribution(params.recovery_params)

        self.failure_deltas = []
        self.recovery_deltas = []
        self.current_state = 1
        self.delta_t = 0
        self.val = random.uniform(0, 1)
        self.counter = 0
        self.steps = 0

    def drop(self) -> bool:
        """
        Random sampling from Semi-Markov probability distribution with failure distribution params.failure_distribution
        and recovery distribution params.recovery_distribution.
        There are two states:

        params.failure_distribution describe the time probability distribution for a failure to occur
        params.recovery_distribution describe the time probability distribution for a recovery to occur

        if state is <working>: the time of the next failure is randomly sampled from failure_distribution
        if state is <failed>: the time of the next recovery is randomly sampled from recovery_distribution
        @return: True if the measurement is lost, False otherwise
        """
        return_val = False
        if self.current_state == 1:
            if self.change_from_1():
                self.failure_deltas.append(self.delta_t-self.dt/2)
                self.delta_t = 0
                self.current_state = 0
                self.val = random.uniform(0, 1)
                return_val = True
        elif self.current_state == 0:
            if self.change_from_0():
                self.recovery_deltas.append(self.delta_t-self.dt/2)
                self.delta_t = 0
                self.current_state = 1
                self.val = random.uniform(0, 1)
            else:
                return_val = True
        self.delta_t += self.dt

        self.counter += self.current_state
        self.steps += 1

        return return_val

    def mean(self) -> Tuple[float, float, float]:
        """
        Statistical means
        @return: The statistical mean state (0: failed, 1: working), the statistical mean time to failure and the
            statistical mean time to recovery.
        """
        return self.counter/self.steps, \
               sum(self.failure_deltas) / len(self.failure_deltas) if len(self.failure_deltas) != 0 else None, \
               sum(self.recovery_deltas) / len(self.recovery_deltas) if len(self.recovery_deltas) != 0 else None

    def change_from_1(self) -> bool:
        """
        @return: Whether a change from working to failed occurred at this time step
        """
        val = self.failure_distribution.cdf(self.delta_t)
        return self.val <= val

    def change_from_0(self) -> bool:
        """
        @return: Whether a change from failed to working occurred at this time step
        """
        val = self.recovery_distribution.cdf(self.delta_t)
        return self.val <= val


DroppingMaps: Dict[type(DroppingTechniques), type(BaseParams)] = {
    LGB: LGBParam,
    LGM: LGMParam,
    LGSM: LGSMParam
}
