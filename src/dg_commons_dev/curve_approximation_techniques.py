from casadi import *
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union


class CurveApproximationTechnique(ABC):
    """
    The classes inheriting from this class provide tools for approximating a curve based on three points on it and the
    angles between the curve and the x-axis at the previously mentioned points.
    The method closest_point_on_path needs furthermore to be numpy-free in order to be used with casadi library.
    """

    parameters: Optional[List[float]]
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
    def update_from_data(self, pos1, angle1, pos2, angle2, pos3, angle3):
        """ Function updating the parameters based on the provided data """
        pass

    @abstractmethod
    def update_from_parameters(self, params):
        """ Function updating the parameters """
        pass


class LinearCurve(CurveApproximationTechnique):
    """ Class for linear approximations """

    @property
    def n_params(self) -> int:
        return 3

    def vertical_line_param(self, pos1, pos2):
        s: float = sign(pos2[1] - pos1[1])
        angle: float = pi / 2 * s
        angle: float = 2 * pi + angle if angle < 0 else angle  # returns a value in [0, 2pi]
        m: float = s * 10e3
        b: float = pos2[1] - m * pos2[0]

        self.update_from_parameters(np.array([[m], [b], [angle]]))

    def update_from_data(self, pos1, angle1, pos2, angle2, pos3, angle3):
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

    def update_from_parameters(self, params):
        m, b, angle = params[0, 0], params[1, 0], params[2, 0]

        self.parameters: List[float] = [m, b, angle]


class QuadraticCurve(CurveApproximationTechnique):
    """ Class for quadratic approximations """

    @property
    def n_params(self) -> int:
        return 3

    def update_from_data(self, pos1, angle1, pos2, angle2, pos3, angle3):
        # if abs(pos1[0] - pos2[0]) == 0 and abs(pos2[0] - pos3[0]): # TODO: fix
        #     return vertical_line_param(pos1, pos2)

        pref = 'var3'
        if pref == 'var1':
            mat: np.ndarray = \
                np.array([[pos1[0] ** 2, pos1[0], 1], [pos2[0] ** 2, pos2[0], 1], [pos3[0] ** 2, pos3[0], 1]])
            b: np.ndarray = np.array([[pos1[1]], [pos2[1]], [pos3[1]]])
        elif pref == 'var2':
            mat: np.ndarray = np.array([[pos1[0] ** 2, pos1[0], 1], [pos3[0] ** 2, pos3[0], 1], [2 * pos1[0], 1, 0]])
            b: np.ndarray = np.array([[pos1[1]], [pos2[1]], [tan(angle1)]])
        elif pref == 'var3':
            mat: np.ndarray = \
                np.array([[pos1[0] ** 2, pos1[0], 1], [pos2[0] ** 2, pos2[0], 1], [pos3[0] ** 2, pos3[0], 1],
                         [2 * pos1[0], 1, 0], [2 * pos2[0], 1, 0], [2 * pos3[0], 1, 0]])
            b: np.ndarray = np.array([[pos1[1]], [pos2[1]], [pos3[1]], [tan(angle1)], [tan(angle2)], [tan(angle3)]])

        res: np.ndarray = np.linalg.lstsq(mat, b)[0]
        a, b, c = res[0][0], res[1][0], res[2][0]
        # if abs(2 * a * pos2[0]) / abs(2 * a * pos2[0] + b) < 5 * 10e-2:
        #     return linear_param(pos1, angle1, pos2, angle2, pos3, angle3) # TODO: fix

        self.update_from_parameters(a, b, c)

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

    def update_from_parameters(self, *args):
        a, b, c = args[0], args[1], args[2]
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

    def update_from_data(self, pos1, angle1, pos2, angle2, pos3, angle3):
        # if abs(pos1[0] - pos2[0]) == 0 and abs(pos2[0] - pos3[0]): # TODO: fix
        #     return vertical_line_param(pos1, pos2)

        mat: np.ndarray = np.array([[pos1[0] ** 3, pos1[0] ** 2, pos1[0], 1], [pos3[0] ** 3, pos3[0] ** 2, pos3[0], 1],
                                    [3 * pos1[0] ** 2, 2 * pos1[0], 1, 0], [3 * pos3[0] ** 2, 2 * pos3[0], 1, 0]])
        b: np.ndarray = np.array([[pos1[1]], [pos3[1]], [tan(angle1)], [tan(angle3)]])
        res: np.ndarray = np.linalg.lstsq(mat, b)[0]

        a, b, c, d = res[0][0], res[1][0], res[2][0], res[3][0]

        # if abs(3 * a * pos2[0] ** 2) / abs(3 * a * pos2[0] ** 2 + 2 * b * pos2[0] + c) < 5 * 10e-2: # TODO: fix
        #     return quadratic_param(pos1, angle1, pos2, angle2, pos3, angle3)

        self.update_from_parameters(a, b, c, d)

    def update_from_parameters(self, *args):
        a, b, c, d = args[0], args[0], args[0], args[0]
        self.parameters = [a, b, c, d]


CurveApproximationTechniques = Union[CurveApproximationTechnique, LinearCurve, QuadraticCurve, CubicCurve]


def cuberoot(x):
    s: float = sign(x)
    return s * (s * x) ** (1/3)


def solve_quadratic(a, b, c, d):
    p: float = (3 * a * c - b ** 2) / (3 * (a ** 2))
    q: float = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)
    summand: float = -b / (3 * a)
    sol: List[SX] = []

    val1: float = -q / 2 - sqrt(q ** 2 / 4 + p ** 3 / 27)
    val2: float = -q / 2 + sqrt(q ** 2 / 4 + p ** 3 / 27)

    sol.append(cuberoot(val1) + cuberoot(val2) + summand)

    for i in range(3):
        try:
            val: float = 1 / 3 * acos((3 * q) / (2 * p) * sqrt(-3 / p)) - 2 * pi * i / 3
            sol.append(2 * sqrt(-p / 3) * cos(val) + summand)
        except Exception:
            sol.append(casadi.inf)
    return sol


def mat_mul(x, y):

    result: List[List[float]] = [([0]*len(y[0]))*len(x)]

    # iterate through rows of X
    for i in range(len(x)):
        # iterate through columns of Y
        for j in range(len(y[0])):
            # iterate through rows of Y
            for k in range(len(y)):
                result[i][j] += x[i][k] * y[k][j]
