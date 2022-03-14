from typing import Generic, TypeVar
from abc import ABC, abstractmethod

Ref = TypeVar('Ref')
Obs = TypeVar('Obs')
U = TypeVar('U')


class Controller(ABC, Generic[Ref, Obs, U]):
    """ Interface for controllers """

    @abstractmethod
    def update_ref(self, new_ref: Ref):
        """ The reference is updated """
        pass

    @abstractmethod
    def control(self, new_obs: Obs, t: float) -> U:
        """ A new observation is processed and an input for the system formulated and returned """
        pass
