from casadi import *
from typing import List, Callable
from math import gcd
from dg_commons.sim.models.vehicle import VehicleGeometry

""" Fully casadi compatible """


# TODO: implement proper casadi-compatible vector dataclass with all operations


def combine_list(list1, list2, alpha1, alpha2):
    """ Linear combination of lists """
    zipped_lists = zip(list1, list2)
    return [alpha1 * x + alpha2 * y for (x, y) in zipped_lists]


def kin(x: SX, y: SX, theta: SX, v: SX, delta: SX, s: SX,
        v_delta: SX, v_s: SX, acc: SX, vehicle_geometry: VehicleGeometry, rear_axle: bool):
    """ Kinematic state space model derivative """
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


def euler(state: List[SX], f: Callable[[List[SX]], List[SX]], ts: float):
    """ Euler discretization """
    rhs = f(state)
    return combine_list(state, rhs, 1, ts)


def rk4(state: List[SX], f: Callable[[List[SX]], List[SX]], h: float):
    """ Runge Kutta 4 discretization """
    k1 = f(state)
    k2 = f(combine_list(state, k1, 1, h/2))
    k3 = f(combine_list(state, k2, 1, h/2))
    k4 = f(combine_list(state, k3, 1, h))
    k = combine_list(combine_list(combine_list(k1, k2, 1, 2), k3, 1, 2), k4, 1, 1)
    return combine_list(state, k, 1, h/6)


def anstrom_euler(state: List[SX], f: Callable[[List[SX]], List[SX]], ts: float):
    """ Anstrom discretization """
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
            vehicle_geometry: VehicleGeometry, ts: float, rear_axle: bool):
    """ Runge kutta 4 for kinematic bicycle model """
    def f(param):
        return kin(*param, v_delta, v_s, acc, vehicle_geometry=vehicle_geometry, rear_axle=rear_axle)

    state = [x, y, theta, v, delta, s]
    return rk4(state, f, ts)


def kin_euler(x: SX, y: SX, theta: SX, v: SX, delta: SX, s: SX, v_delta: SX, v_s: SX, acc: SX,
              vehicle_geometry: VehicleGeometry, ts: float, rear_axle):
    """ Euler for kinematic bicycle model """
    def f(param):
        return kin(*param, v_delta, v_s, acc, vehicle_geometry=vehicle_geometry, rear_axle=rear_axle)

    state = [x, y, theta, v, delta, s]
    return euler(state, f, ts)


def kin_anstrom_euler(x: SX, y: SX, theta: SX, v: SX, delta: SX, s: SX, v_delta: SX, v_s: SX, acc: SX,
                      vehicle_geometry: VehicleGeometry, ts: float, rear_axle):
    """ Anstrom for kinematic bicycle model """
    def f(param):
        return kin(*param, v_delta, v_s, acc, vehicle_geometry=vehicle_geometry, rear_axle=rear_axle)

    state = [x, y, theta, v, delta, s]
    return anstrom_euler(state, f, ts)


discretizations = {'Kinematic Euler': kin_euler, 'Kinematic RK4': kin_rk4, 'Anstrom Euler': kin_anstrom_euler}
