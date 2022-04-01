import time
from dataclasses import dataclass, fields, asdict
from typing import List
import numpy as np
import scipy.linalg
from abc import ABC
from typing import Any


def sig_fig_round(number, n_digits=5, rounding=12):
    number = round(number, rounding)
    return_val = float(f"{number:.{n_digits}g}")
    return return_val


v_sig_fig_round = np.vectorize(sig_fig_round)


@dataclass
class SemiDef:
    """
    Positive Semi-definite Matrices with Ellipse Inclusion Order

    The order induced by eq, lt le is as follows:
    Semidef1 <= Semidef2 if sort(eig(Semidef1))_i <= sort(eig(Semidef2))_i for all i meaning that the ellipse induced
    by the quadratic form of Semidef1 is included or equal to the ellipse induced by the quadratic form of Semidef2.
    """

    eig: List[float] = None
    """ List of eigenvalues """

    matrix: np.ndarray = None
    """ Matrix as n x n numpy array """

    def __post_init__(self) -> None:
        """
        Checks that either the eigenvalues or the matrix itself is defined.
        If both of them are passed, it is checked that they match.
        """
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
        """
        Creates a random matrix with the eigenvalues passed as arguments.
        @param eigenvalues: The desired eigenvalues.
        @return: The computed matrix.
        """
        n: int = len(eigenvalues)
        s: np.ndarray = np.diag(eigenvalues)
        q, _ = scipy.linalg.qr(np.random.rand(n, n))
        semidef: np.ndarray = q.T @ s @ q
        return semidef

    @staticmethod
    def eig_from_matrix(semidef: np.ndarray) -> List[float]:
        """
        Computes the eigenvalues
        @param semidef: nxn matrix.
        @return: its eigenvalues
        """
        eig = np.linalg.eigvalsh(semidef).tolist()
        eig: List[float] = [round(i, 6) for i in eig]
        return eig

    def __eq__(self, other):
        return np.allclose(self.matrix, other.matrix)

    def to_str(self):
        return str(np.round(np.array(self.eig), 4))

    def __lt__(self, other):
        mat = other.matrix - self.matrix
        return np.all(np.linalg.eigvals(mat) > 0)

    def __le__(self, other):
        mat = other.matrix - self.matrix
        return np.all(np.linalg.eigvals(mat) >= 0)


@dataclass
class BaseParams(ABC):
    """ A BaseParams is a dataclass structure defining the external parameters of a functional class """

    def __str__(self):
        """
        Return string explicitly set for visualization
        @return: Return string
        """
        string = "############## Base Parameters {} ##############\n".format(self.__class__.__name__)
        string += "--------------------------------------------------\n"
        for field in fields(self):
            value = getattr(self, field.name)
            string += field.name + " of type {}: ".format(field.type) + str(value) + "\n"
            string += "--------------------------------------------------\n"
        return string

    def is_close(self, other: "BaseParams"):
        return_val: bool = isinstance(self, type(other))
        if return_val:
            for field in fields(self):
                return_val = return_val and self.compare(getattr(self, field.name), getattr(other, field.name))
        return return_val

    def to_str(self):
        string = self.__class__.__name__
        for field in fields(self):
            value = getattr(self, field.name)

            if hasattr(value, 'to_str'):
                value = value.to_str()
            else:
                value = str(self.rounding(value))
            string += field.name + value
        return string

    @staticmethod
    def rounding(val: Any):
        if isinstance(val, tuple):
            return (sig_fig_round(num) for num in val)
        elif isinstance(val, list):
            return [sig_fig_round(num) for num in val]
        elif isinstance(val, bool):
            if val:
                return 1
            else:
                return 0
        elif isinstance(val, int) or isinstance(val, float):
            return sig_fig_round(val)
        elif isinstance(val, np.ndarray):
            return v_sig_fig_round(val)
        else:
            return val

    @staticmethod
    def compare(val1: Any, val2: Any, tol=10e-10):
        if not isinstance(val1, type(val2)):
            return False

        is_number: bool = isinstance(val1, int) or isinstance(val1, float)
        is_numpy: bool = isinstance(val1, np.ndarray)
        if is_number:
            return abs(val1 - val2) < tol
        elif is_numpy:
            return np.allclose(val1, val2)
        else:
            return val1 == val2
