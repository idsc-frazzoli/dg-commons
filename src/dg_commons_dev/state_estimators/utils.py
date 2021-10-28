from dataclasses import dataclass
from typing import Union, List
import math
from dg_commons_dev.utils import BaseParams


@dataclass
class ExponentialParams(BaseParams):
    """ Exponential Distribution Parameters """

    lamb: Union[List[float], float] = 1

    def __post_init__(self):
        if isinstance(self.lamb, list):
            for i in self.lamb:
                assert i > 0
        else:
            assert self.lamb > 0
        super().__post_init__()


class Exponential:
    """ Exponential Distribution """

    def __init__(self, params: ExponentialParams):
        self.params = params

    def cdf(self, t):
        assert t >= 0

        return 1 - math.exp(-self.params.lamb * t)

    def pdf(self, t):
        assert t >= 0

        return self.params.lamb * math.exp(-self.params.lamb * t)


PDistribution = Union[Exponential]
PDistributionParams = Union[ExponentialParams]
