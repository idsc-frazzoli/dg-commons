from dataclasses import dataclass
from typing import Union, List
import math
from dg_commons_dev.utils import BaseParams


@dataclass
class ExponentialParams(BaseParams):
    """
    Exponential Distribution Parameters
    """

    lamb: float = 1
    """ lambda > 0 """

    def __post_init__(self):
        assert self.lamb > 0


class Exponential:
    """
    Exponential Distribution Parameters

    f(t) = lambda * exp(- lambda * t) if t >= 0
    f(t) = 0 otherwise
    """
    REF_PARAMS: dataclass = ExponentialParams

    def __init__(self, params: ExponentialParams):
        self.params = params

    def cdf(self, t):
        """
        Cumulative distribution function
        @param t: val
        @return: evaluation at t
        """
        assert t >= 0

        return 1 - math.exp(-self.params.lamb * t)

    def pdf(self, t):
        """
        Probability density function
        @param t: val
        @return: evaluation at t
        """
        assert t >= 0

        return self.params.lamb * math.exp(-self.params.lamb * t)


PDistribution = Union[Exponential]
PDistributionParams = Union[ExponentialParams]
