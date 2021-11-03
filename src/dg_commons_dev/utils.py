from dataclasses import dataclass, fields
from typing import Optional, List, Callable, TypeVar, Mapping, Dict, Any, Generic
import numpy as np
import scipy.linalg
from abc import ABC, abstractmethod
import itertools
import copy


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
        return self.matrix == other.matrix

    def __lt__(self, other):
        self.eig.sort()
        other.eig.sort()

        return all(self.eig[i] < other.eig[i] for i, _ in enumerate(self.eig))

    def __le__(self, other):
        self.eig.sort()
        other.eig.sort()

        return all(self.eig[i] <= other.eig[i] for i, _ in enumerate(self.eig))


@dataclass
class BaseParams(ABC):
    """ A BaseParams is a dataclass structure defining the external parameters of a functional class """

    @property
    @abstractmethod
    def reference_class(self) -> Callable[["BaseParams"], Any]:
        """
        Every base parameter class defines the parameters of a reference class,
        which needs to be set explicitly.
        """
        pass
