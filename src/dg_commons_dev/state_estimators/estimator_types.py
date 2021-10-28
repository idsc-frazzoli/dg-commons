from abc import ABC, abstractmethod
from dg_commons import U, X
from dataclasses import dataclass
from dg_commons_dev.utils import BaseParams


@dataclass
class EstimatorParams(BaseParams):
    """ The estimator parameters parametrize the estimation process """
    pass


class Estimator(ABC):
    """ An estimator estimates a state based on a prediction model and on measurements """

    @abstractmethod
    def update_prediction(self, uk: U):
        """ The estimate gets updated based on model predictions and on the input to the system """
        pass

    @abstractmethod
    def update_measurement(self, mk: X):
        """ The estimate gets updated based on a measurement model and a measurement """
        pass


@dataclass
class DroppingTechniquesParams(BaseParams):
    """ The dropping techniques parameters parametrize the dropping process """
    pass


class DroppingTechniques(ABC):
    """ A dropping technique emulates the loss of measurements """

    @abstractmethod
    def drop(self) -> bool:
        """ Returns true if a measurement is lost """
        pass
