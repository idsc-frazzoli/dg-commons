from abc import ABC, abstractmethod
from dg_commons import U, X


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


class DroppingTechniques(ABC):
    """ A dropping technique emulates the loss of measurements """

    @abstractmethod
    def drop(self) -> bool:
        """ Returns true if a measurement is lost """
        pass
