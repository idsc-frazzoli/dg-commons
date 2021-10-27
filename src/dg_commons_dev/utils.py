from dataclasses import dataclass, fields
from typing import Optional, List, Callable, TypeVar, Mapping, Dict, Any, Generic
import numpy as np
import scipy.linalg
from abc import ABC
import itertools
import copy


@dataclass
class SemiDef:
    """
    Positive Semi-definite Matrices

    The order induced by eq, lt le is as follows:
    Semidef1 <= Semidef2 if sort(eig(Semidef1))_i <= sort(eig(Semidef2))_i for all i meaning that the ellipse induced
    by the quadratic form of Semidef1 is included or equal to the ellipse induced by the quadratic form of Semidef2.
    """

    eig: Optional[List[float]] = None

    matrix: Optional[np.ndarray] = None

    def __post_init__(self):
        defined_eig: bool = self.eig is not None
        defined_mat: bool = self.matrix is not None

        assert defined_mat or defined_eig

        if defined_eig and type(self.eig) is not list:
            self.eig: List[float] = self.eig.tolist()

        if defined_eig and defined_mat:
            mat_eig: List[float] = SemiDef.eig_from_matrix(self.matrix)
            mat_eig.sort()
            self.eig.sort()

            assert mat_eig == self.eig
        elif defined_eig:
            self.matrix: np.ndarray = SemiDef.matrix_from_eigenvalues(self.eig)
        else:
            self.eig: List[float] = SemiDef.eig_from_matrix(self.matrix)

        assert all(i >= 0 for i in self.eig)

        assert np.allclose(self.matrix, self.matrix.T)

    @staticmethod
    def matrix_from_eigenvalues(eigenvalues: List[float]) -> np.ndarray:
        n: int = len(eigenvalues)
        s: np.ndarray = np.diag(eigenvalues)
        q, _ = scipy.linalg.qr(np.random.rand(n, n))
        semidef: np.ndarray = q.T @ s @ q
        return semidef

    @staticmethod
    def eig_from_matrix(semidef: np.ndarray) -> List[float]:
        eig = np.linalg.eigvalsh(semidef).tolist()
        eig: List[float] = [round(i, 6) for i in eig]
        return eig

    def __eq__(self, other):
        return self.matrix == other.matrix

    def __lt__(self, other):
        self.eig.sort()
        other.eig.sort()

        return all(self.eig[i] < other.eig[i] for i, _ in enumerate(self.eig))

    def __le__(self, other):
        self.eig.sort()
        other.eig.sort()

        return all(self.eig[i] <= other.eig[i] for i, _ in enumerate(self.eig))


R = TypeVar("R")


def func(values: R) -> bool:
    return True


@dataclass
class BaseParams(ABC, Generic[R]):
    """
    Base dataclass every parameter dataclasses are inheriting from.
    Creates the baseline for iterating over different set of parameters.
    """

    condition: Callable[[R], bool] = func
    """ 
    This function is a filter for instances of the inheriting dataclasses. 
    Takes an instance as input argument and returns true if the instance is approved.
    """

    def __post_init__(self):
        lists: List[Any] = []
        single_list: List[bool] = []
        for field in fields(self):
            values: Any = getattr(self, field.name)
            res: Any = self.process_mutually_exclusive_values(values)
            lists.append(res)
            single_list.append(len(res) == 1)

        self.xplets: List[Any] = list(itertools.product(*lists))
        self.is_single: bool = all(single_list)
        if self.is_single:
            self.n_total: int = 1
        else:
            helper: List[Any] = copy.deepcopy(self.xplets)
            for xplet in self.xplets:
                try:
                    new_instance: Optional[R] = self.__new__(type(self))
                    new_instance.__init__(*xplet)
                except Exception:
                    helper.remove(xplet)

            self.xplets: List[Any] = helper
            self.n_total: int = len(self.xplets)

    def process_mutually_exclusive_values(self, values: Any) -> Any:
        return_values = []

        def process_single_value(inst):
            if self.is_nested(inst):
                for val in inst.gen():
                    return_values.append(val)
            else:
                return_values.append(inst)

        if isinstance(values, list):
            for value in values:
                process_single_value(value)
        else:
            process_single_value(values)
        return return_values

    @staticmethod
    def is_nested(value: Any) -> bool:
        try:
            value.get_count()
            return True
        except AttributeError:
            return False

    def get_count(self) -> int:
        return self.n_total

    def gen(self) -> R:
        for xplet in self.xplets:
            new_instance = self.__new__(type(self))
            new_instance.__init__(*xplet)
            if self.condition(new_instance):
                yield new_instance
