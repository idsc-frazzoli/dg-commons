import random
from dataclasses import dataclass
from typing import Union, List, Dict
import scipy
import numpy as np
from dg_commons_dev.state_estimators.utils import PDistribution, PDistributionParams, ExponentialParams, Exponential
from dg_commons_dev.state_estimators.estimator_types import DroppingTechniques
from dg_commons_dev.utils import BaseParams


@dataclass
class LGBParam(BaseParams):

    failure_p: Union[List[float], float] = 0
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
        val = random.uniform(0, 1)

        return_val = False
        if val <= self.params.failure_p:
            return_val = True

        current_state = 0 if return_val else 1
        self.counter += current_state
        self.steps += 1

        return return_val

    def mean(self):
        return self.counter/self.steps


@dataclass
class LGMParam(LGBParam):

    recovery_p: Union[float, List[float]] = 0
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
    def process_instance(f, r):
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

    def mean(self):
        return self.counter/self.steps


@dataclass
class LGSMParam(BaseParams):
    failure_distribution: Union[List[type(PDistribution)], type(PDistribution)] = Exponential
    failure_params: Union[List[PDistributionParams], PDistributionParams] = ExponentialParams()

    recovery_distribution: Union[List[type(PDistribution)], type(PDistribution)] = Exponential
    recovery_params: Union[List[PDistributionParams], PDistributionParams] = ExponentialParams()

    dt: Union[List[float], float] = 0.1


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

    def mean(self):
        return self.counter/self.steps, \
               sum(self.failure_deltas) / len(self.failure_deltas) if len(self.failure_deltas) != 0 else None, \
               sum(self.recovery_deltas) / len(self.recovery_deltas) if len(self.recovery_deltas) != 0 else None

    def change_from_1(self) -> bool:
        val = self.failure_distribution.cdf(self.delta_t)
        return self.val <= val

    def change_from_0(self) -> bool:
        val = self.recovery_distribution.cdf(self.delta_t)
        return self.val <= val


DroppingMaps: Dict[type(DroppingTechniques), type(BaseParams)] = {
    LGB: LGBParam,
    LGM: LGMParam,
    LGSM: LGSMParam
}
