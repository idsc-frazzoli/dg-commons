import numpy as np
from casadi import *
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

""" closest_point_on_path methods are fully compatible with casadi """


def get_model(A, y, lamb=0):
    n_col = A.shape[1]
    return np.linalg.lstsq(A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y))


class CurveApproximationTechnique(ABC):
    """
    The classes inheriting from this class provide tools for approximating a curve based on three points on it and the
    angles between the curve and the x-axis at the previously mentioned points.
    The method closest_point_on_path needs furthermore to be numpy-free in order to be used with casadi library.
    """

    parameters: List[float]
    """ Parameters of the model in use """
    n_params: int
    """ Min # of parameters required """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """ Returns the min number of parameters required """
        pass

    @abstractmethod
    def function(self, x_val: float) -> Optional[Callable[[float], float]]:
        """
        y = f(x)
        Given the x value, this function returns the y value.
        """
        pass

    @abstractmethod
    def closest_point_on_path(self, pos: List[float]) -> List[float]:
        """ Function returning the closest point on the approximated curve to pos """
        pass

    @abstractmethod
    def update_from_data(self, positions, angles) -> None:
        """ Function updating the parameters based on the provided data """
        pass

    @abstractmethod
    def update_from_parameters(self, params) -> None:
        """ Function updating the parameters """
        pass


class LinearCurve(CurveApproximationTechnique):
    """ Class for linear approximations """

    @property
    def n_params(self) -> int:
        return 3

    def vertical_line_param(self, pos1, pos2) -> None:
        s: float = sign(pos2[1] - pos1[1])
        angle: float = pi / 2 * s
        angle: float = 2 * pi + angle if angle < 0 else angle  # returns a value in [0, 2pi]
        m: float = s * 10e3
        b: float = pos2[1] - m * pos2[0]

        self.update_from_parameters(np.array([[m], [b], [angle]]))

    def update_from_data(self, positions, angles) -> None:
        pos1 = positions[0]
        pos2 = positions[int(len(positions)/2)]
        if abs(pos1[0] - pos2[0]) == 0:
            return self.vertical_line_param(pos1, pos2)

        m: float = (pos1[1] - pos2[1]) / (pos1[0] - pos2[0])
        b: float = pos1[1] - m * pos1[0]
        angle: float = atan2(pos2[1] - pos1[1], pos2[0] - pos1[0])  # returns a value in [-pi, pi]
        angle: float = 2 * pi + angle if angle < 0 else angle  # returns a value in [0, 2pi]
        self.update_from_parameters(np.array([[m], [b], [angle]]))

    def function(self, x_val: float) -> float:
        m, b = self.parameters[0], self.parameters[1]
        y_val: float = m * x_val + b
        return y_val

    def closest_point_on_path(self, pos: List[float]) -> List[float]:
        m, b = self.parameters[0], self.parameters[1]
        x_val: float = (pos[0] + m * (pos[1] - b)) / (1 + m ** 2)
        y_val: float = (m ** 2 * pos[1] + m * pos[0] + b) / (1 + m ** 2)
        return [x_val, y_val]

    def update_from_parameters(self, params) -> None:
        m, b, angle = params[0, 0], params[1, 0], params[2, 0]

        self.parameters: List[float] = [m, b, angle]


class QuadraticCurve(CurveApproximationTechnique):
    """ Class for quadratic approximations """
    linear = False
    linear_model = LinearCurve()

    @property
    def n_params(self) -> int:
        return 3

    def update_from_data(self, positions, angles) -> None:
        n_pos = len(positions)

        mat = np.zeros((n_pos, 3))
        b = np.zeros((n_pos, 1))
        for i in range(n_pos):
            pos = positions[i]

            mat[i, 0] = pos[0] ** 2
            mat[i, 1] = pos[0]
            mat[i, 2] = 1

            b[i, 0] = pos[1]

        x_diff = abs(positions[-1][0] - positions[0][0])
        y_diff = abs(positions[-1][1] - positions[0][1])
        if x_diff/y_diff < 4*10e-2:
            linear = LinearCurve()
            linear.update_from_data(positions, angles)
            linear_params = linear.parameters
            args = np.array([[0], [0], [linear_params[0]], [linear_params[1]]])
        else:
            res: np.ndarray = get_model(mat, b, 0.001)[0]

            a, b, c = res[0][0], res[1][0], res[2][0]
            args = np.array([[a], [b], [c]])

        self.update_from_parameters(args)

    def function(self, x_val: float) -> float:
        a, b, c = self.parameters[0], self.parameters[1], self.parameters[2]
        y_val: float = a * x_val ** 2 + b * x_val + c
        return y_val

    def closest_point_on_path(self, pos: List[float]) -> List[float]:
        a, b, c = self.parameters[0], self.parameters[1], self.parameters[2]
        func = self.function

        a1 = 2 * a ** 2
        a2 = (3 * a * b)
        a3 = (1 - 2 * a * pos[1] + b ** 2 + 2 * a * c)
        a4 = (c * b - pos[1] * b - pos[0])
        sols: List[float] = solve_quadratic(a1, a2, a3, a4)
        dists_list: List[float] = [power(x_c - pos[0], 2) + power(func(x_c) - pos[1], 2) for x_c in sols]
        dists: SX = SX(4, 1)
        dists[0, 0]: SX = dists_list[0]
        dists[1, 0]: SX = dists_list[1]
        dists[2, 0]: SX = dists_list[2]
        dists[3, 0]: SX = dists_list[3]

        min_dist: SX = mmin(dists)
        x_sol: SX = casadi.inf
        for sol in sols:
            current_dist: SX = power(sol - pos[0], 2) + power(func(sol) - pos[1], 2)
            x_sol: SX = if_else(current_dist == min_dist, sol, x_sol)

        return [x_sol, func(x_sol)]

    def update_from_parameters(self, args) -> None:
        a, b, c = args[0, 0], args[1, 0], args[2, 0]
        self.parameters: List[float] = [a, b, c]


class CubicCurve(CurveApproximationTechnique):
    """ Class for cubic approximations """

    @property
    def n_params(self) -> int:
        return 4

    def closest_point_on_path(self, pos: List[float]) -> List[float]:
        raise NotImplementedError("Analytical solution for closest point to cubic fct not implemented")

    def function(self, x_val: float) -> float:
        a, b, c, d = self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]
        y_val: float = a * x_val ** 3 + b * x_val ** 2 + c * x_val + d
        return y_val

    def update_from_data(self, positions, angles) -> None:
        # if abs(pos1[0] - pos2[0]) == 0 and abs(pos2[0] - pos3[0]): # TODO: fix
        #     return vertical_line_param(pos1, pos2)
        n_pos = len(positions)

        mat = np.zeros((n_pos, 4))
        b = np.zeros((n_pos, 1))
        for i in range(n_pos):
            pos = positions[i]

            mat[i, 0] = pos[0] ** 3
            mat[i, 1] = pos[0] ** 2
            mat[i, 2] = pos[0]
            mat[i, 3] = 1

            b[i, 0] = pos[1]

        x_diff = abs(positions[-1][0] - positions[0][0])
        y_diff = abs(positions[-1][1] - positions[0][1])
        if x_diff/y_diff < 4*10e-2:
            linear = LinearCurve()
            linear.update_from_data(positions, angles)
            linear_params = linear.parameters
            args = np.array([[0], [0], [linear_params[0]], [linear_params[1]]])
        else:
            res: np.ndarray = get_model(mat, b, 0.0001)[0]

            a, b, c, d = res[0][0], res[1][0], res[2][0], res[3][0]
            args = np.array([[a], [b], [c], [d]])

        self.update_from_parameters(args)

    def update_from_parameters(self, args) -> None:
        a, b, c, d = args[0, 0], args[1, 0], args[2, 0], args[3, 0]
        self.parameters = [a, b, c, d]


CurveApproximationTechniques = Union[CurveApproximationTechnique, LinearCurve, QuadraticCurve, CubicCurve]


def cuberoot(x: SX) -> SX:
    """
    casadi-compatible cube-root
    @param x: value
    @return: cube-root of x
    """
    s: SX = sign(x)
    return s * (s * x) ** (1/3)


def solve_quadratic(a: float, b: float, c: float, d: float) -> List[float]:
    """
    Casadi-compatible solve ax^3 + bx^2 + cx + d = 0
    @param a: param1
    @param b: param2
    @param c: param3
    @param d: param4
    @return: solutions of interest for point closest to a quadratic curve
    """
    p: float = (3 * a * c - b ** 2) / (3 * (a ** 2))
    q: float = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)
    summand: float = -b / (3 * a)
    sol: List[SX] = []

    val1: SX = -q / 2 - sqrt(q ** 2 / 4 + p ** 3 / 27)
    val2: SX = -q / 2 + sqrt(q ** 2 / 4 + p ** 3 / 27)

    sol.append(cuberoot(val1) + cuberoot(val2) + summand)

    for i in range(3):
        try:
            val: float = 1 / 3 * acos((3 * q) / (2 * p) * sqrt(-3 / p)) - 2 * pi * i / 3
            sol.append(2 * sqrt(-p / 3) * cos(val) + summand)
        except Exception:
            sol.append(casadi.inf)
    return sol


def mat_mul(x: List[List[float]], y: List[List[float]]) -> List[List[float]]:
    """
    Computes casadi compatible matrix multiplication
    @param x: n x m matrix
    @param y: m x k matrix
    @return n x k matrix
    """
    result: List[List[float]] = [([0]*len(y[0]))*len(x)]

    # iterate through rows of X
    for i in range(len(x)):
        # iterate through columns of Y
        for j in range(len(y[0])):
            # iterate through rows of Y
            for k in range(len(y)):
                result[i][j] += x[i][k] * y[k][j]

    return result
