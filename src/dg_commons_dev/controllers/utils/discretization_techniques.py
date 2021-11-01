from casadi import *
from typing import List, Callable, Any
from math import gcd
from dg_commons.sim.models.vehicle import VehicleGeometry

""" Fully casadi compatible """


# TODO: implement proper casadi-compatible vector dataclass with all operations


def combine_list(list1: List[Any], list2: List[Any], alpha1: float, alpha2: float) -> List[Any]:
    """
    Linear combination of lists
    @param list1: first list
    @param list2: second list
    @param alpha1: first list multiplier
    @param alpha2: second list multiplier
    @return: linear combination
    """
    zipped_lists = zip(list1, list2)
    return [alpha1 * x + alpha2 * y for (x, y) in zipped_lists]


def kin(x: SX, y: SX, theta: SX, v: SX, delta: SX, s: SX,
        v_delta: SX, v_s: SX, acc: SX, vehicle_geometry: VehicleGeometry, rear_axle: bool) \
        -> List[SX]:
    """
    Kinematic state space model derivative
    @param x: x-position
    @param y: y-position
    @param theta: orientation
    @param v: rear velocity
    @param delta: steering angle
    @param s: parametrized position on path (only for path variable implementation)
    @param v_delta: steering velocity
    @param v_s: run across velocity (only for path variable implementation)
    @param acc: rear axle acceleration along current orientation
    @param vehicle_geometry: vehicle geometry
    @param rear_axle: rear axle dynamics (True), cog dynamics (False)
    @return: list of time-derivatives for all state variables
    """
    return_val = []
    dtheta = v * tan(delta) / vehicle_geometry.length
    if rear_axle:
        return_val.append(cos(theta) * v)
        return_val.append(sin(theta) * v)
    else:
        vy = dtheta * vehicle_geometry.lr
        return_val.append(v * cos(theta) - vy * sin(theta))
        return_val.append(v * sin(theta) + vy * cos(theta))

    return_val.append(dtheta)
    return_val.append(acc)
    return_val.append(v_delta)
    return_val.append(v_s)

    return return_val


def euler(state: List[SX], f: Callable[[List[SX]], List[SX]], ts: float) -> List[SX]:
    """
    Euler discretization
    @param state: Current state
    @param f: Function computing state-dependent time derivative of the state variables
    @param ts: Delta time
    @return: Next state
    """
    rhs = f(state)
    return combine_list(state, rhs, 1, ts)


def rk4(state: List[SX], f: Callable[[List[SX]], List[SX]], h: float) -> List[SX]:
    """
    Runge Kutta 4 discretization
    @param state: Current state
    @param f: Function computing state-dependent time derivative of the state variables
    @param h: Delta time
    @return: Next state
    """
    k1 = f(state)
    k2 = f(combine_list(state, k1, 1, h/2))
    k3 = f(combine_list(state, k2, 1, h/2))
    k4 = f(combine_list(state, k3, 1, h))
    k = combine_list(combine_list(combine_list(k1, k2, 1, 2), k3, 1, 2), k4, 1, 1)
    return combine_list(state, k, 1, h/6)


def anstrom_euler(state: List[SX], f: Callable[[List[SX]], List[SX]], ts: float) -> List[SX]:
    """
    Anstrom discretization
    @param state: Current state
    @param f: Function computing state-dependent time derivative of the state variables
    @param ts: Delta time
    @return: Next state
    """
    n: List[int] = [1, 1, 1, 1, 1, 1]
    sampling = [[(k+1)*ts/num for k in range(num)] for num in n]
    lcm = 1
    for i in n:
        lcm = lcm * i // gcd(lcm, i)

    t = ts/lcm
    current_t = 0
    for i in range(lcm):
        current_t += t
        prov_state = state
        for j in range(len(n)):
            if any([abs(current_t - val) < 10e-7 for val in sampling[j]]):
                result = euler(state, f, t*lcm/n[j])
                prov_state[j] = result[j]
        state = prov_state
    return state


def kin_rk4(x: SX, y: SX, theta: SX, v: SX, delta: SX, s: SX, v_delta: SX, v_s: SX, acc: SX,
            vehicle_geometry: VehicleGeometry, ts: float, rear_axle: bool) -> List[SX]:
    """
    Runge kutta 4 for kinematic bicycle model
    @param x: x-position
    @param y: y-position
    @param theta: orientation
    @param v: rear velocity
    @param delta: steering angle
    @param s: parametrized position on path (only for path variable implementation)
    @param v_delta: steering velocity
    @param v_s: run across velocity (only for path variable implementation)
    @param acc: rear axle acceleration along current orientation
    @param vehicle_geometry: vehicle geometry
    @param rear_axle: rear axle dynamics (True), cog dynamics (False)
    @return: next state computed with rk4
    """

    def f(param):
        return kin(*param, v_delta, v_s, acc, vehicle_geometry=vehicle_geometry, rear_axle=rear_axle)

    state = [x, y, theta, v, delta, s]
    return rk4(state, f, ts)


def kin_euler(x: SX, y: SX, theta: SX, v: SX, delta: SX, s: SX, v_delta: SX, v_s: SX, acc: SX,
              vehicle_geometry: VehicleGeometry, ts: float, rear_axle):
    """
    Euler for kinematic bicycle model
    @param x: x-position
    @param y: y-position
    @param theta: orientation
    @param v: rear velocity
    @param delta: steering angle
    @param s: parametrized position on path (only for path variable implementation)
    @param v_delta: steering velocity
    @param v_s: run across velocity (only for path variable implementation)
    @param acc: rear axle acceleration along current orientation
    @param vehicle_geometry: vehicle geometry
    @param rear_axle: rear axle dynamics (True), cog dynamics (False)
    @return: next state computed with Euler
    """
    def f(param):
        return kin(*param, v_delta, v_s, acc, vehicle_geometry=vehicle_geometry, rear_axle=rear_axle)

    state = [x, y, theta, v, delta, s]
    return euler(state, f, ts)


def kin_anstrom_euler(x: SX, y: SX, theta: SX, v: SX, delta: SX, s: SX, v_delta: SX, v_s: SX, acc: SX,
                      vehicle_geometry: VehicleGeometry, ts: float, rear_axle):
    """
    Anstrom for kinematic bicycle model
    @param x: x-position
    @param y: y-position
    @param theta: orientation
    @param v: rear velocity
    @param delta: steering angle
    @param s: parametrized position on path (only for path variable implementation)
    @param v_delta: steering velocity
    @param v_s: run across velocity (only for path variable implementation)
    @param acc: rear axle acceleration along current orientation
    @param vehicle_geometry: vehicle geometry
    @param rear_axle: rear axle dynamics (True), cog dynamics (False)
    @return: next state computed with Anstrom
    """
    def f(param):
        return kin(*param, v_delta, v_s, acc, vehicle_geometry=vehicle_geometry, rear_axle=rear_axle)

    state = [x, y, theta, v, delta, s]
    return anstrom_euler(state, f, ts)


discretizations = {'Kinematic Euler': kin_euler, 'Kinematic RK4': kin_rk4, 'Anstrom Euler': kin_anstrom_euler}
