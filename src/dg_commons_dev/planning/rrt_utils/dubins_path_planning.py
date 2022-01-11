"""
Dubins path planner sample code
author Atsushi Sakai(@Atsushi_twi)
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from typing import Tuple, List, Optional


def dubins_path_planning(s_x: float, s_y: float, s_yaw: float, g_x: float, g_y: float, g_yaw: float, curvature: float,
                         step_size: float = 0.1):
    """
    Dubins path planner
    @param s_x: x position of start point [m]
    @param s_y: y position of start point [m]
    @param s_yaw: yaw angle of start point [rad]
    @param g_x: x position of end point [m]
    @param g_y: y position of end point [m]
    @param g_yaw: yaw angle of end point [rad]
    @param curvature: curvature for curve [1/m]
    @param step_size: (optional) step size between two path points [m]
    @return: x, y and yaw lists of the  of a path; length of path segments
    """
    g_x -= s_x
    g_y -= s_y

    l_rot = Rot.from_euler('z', s_yaw).as_matrix()[0:2, 0:2]
    le_xy = np.stack([g_x, g_y]).T @ l_rot
    le_yaw = g_yaw - s_yaw

    lp_x, lp_y, lp_yaw, modes, lengths, best_cost = dubins_path_planning_from_origin(
        le_xy[0], le_xy[1], le_yaw, curvature, step_size)

    rot = Rot.from_euler('z', -s_yaw).as_matrix()[0:2, 0:2]
    converted_xy = np.stack([lp_x, lp_y]).T @ rot
    x_list = converted_xy[:, 0] + s_x
    y_list = converted_xy[:, 1] + s_y
    yaw_list = [pi_2_pi(i_yaw + s_yaw) for i_yaw in lp_yaw]

    return x_list, y_list, yaw_list, modes, lengths, best_cost


def mod2pi(theta: float) -> float:
    """
    Set theta between 0 and 2pi
    @param theta: angle to be converted
    @return: same angle but in the interval 0 2pi
    """
    return theta - 2.0 * math.pi * math.floor(theta / 2.0 / math.pi)


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def left_straight_left(alpha: float, beta: float, d: float) \
        -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    """
    Left Straight Left canonical path
    @param alpha: angle vehicle x-axis - end position
    @param beta: vehicle target orientation - alpha
    @param d: ratio distance to r_min
    @return: radians left, radians straight, radians left
    """
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d + sa - sb

    mode = ["L", "S", "L"]
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sa - sb))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((cb - ca), tmp0)
    t = mod2pi(-alpha + tmp1)
    p = math.sqrt(p_squared)
    q = mod2pi(beta - tmp1)

    return t, p, q, mode


def right_straight_right(alpha: float, beta: float, d: float) \
        -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    """
    Right Straight Right canonical path
    @param alpha: angle vehicle x-axis - end position
    @param beta: vehicle target orientation - alpha
    @param d: ratio distance to r_min
    @return: radians right, radians straight, radians right
    """
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d - sa + sb
    mode = ["R", "S", "R"]
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sb - sa))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((ca - cb), tmp0)
    t = mod2pi(alpha - tmp1)
    p = math.sqrt(p_squared)
    q = mod2pi(-beta + tmp1)

    return t, p, q, mode


def left_straight_right(alpha: float, beta: float, d: float) \
        -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    """
    Left Straight Right canonical path
    @param alpha: angle vehicle x-axis - end position
    @param beta: vehicle target orientation - alpha
    @param d: ratio distance to r_min
    @return: radians left, radians straight, radians right
    """
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = -2 + (d * d) + (2 * c_ab) + (2 * d * (sa + sb))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
    t = mod2pi(-alpha + tmp2)
    q = mod2pi(-mod2pi(beta) + tmp2)

    return t, p, q, mode


def right_straight_left(alpha: float, beta: float, d: float) \
        -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    """
    Right Straight Left canonical path
    @param alpha: angle vehicle x-axis - end position
    @param beta: vehicle target orientation - alpha
    @param d: ratio distance to r_min
    @return: radians right, radians straight, radians left
    """
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = (d * d) - 2 + (2 * c_ab) - (2 * d * (sa + sb))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
    t = mod2pi(alpha - tmp2)
    q = mod2pi(beta - tmp2)

    return t, p, q, mode


def right_left_right(alpha: float, beta: float, d: float) \
        -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    """
    Right Left Right canonical path
    @param alpha: angle vehicle x-axis - end position
    @param beta: vehicle target orientation - alpha
    @param d: ratio distance to r_min
    @return: radians right, radians left, radians right
    """
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["R", "L", "R"]
    tmp_rlr = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp_rlr) > 1.0:
        return None, None, None, mode

    p = mod2pi(2 * math.pi - math.acos(tmp_rlr))
    t = mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + mod2pi(p / 2.0))
    q = mod2pi(alpha - beta - t + mod2pi(p))
    return t, p, q, mode


def left_right_left(alpha: float, beta: float, d: float) \
        -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    """
    Left Right Left canonical path
    @param alpha: angle vehicle x-axis - end position
    @param beta: vehicle target orientation - alpha
    @param d: ratio distance to r_min
    @return: radians left, radians right, radians left
    """
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["L", "R", "L"]
    tmp_lrl = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (- sa + sb)) / 8.0
    if abs(tmp_lrl) > 1:
        return None, None, None, mode
    p = mod2pi(2 * math.pi - math.acos(tmp_lrl))
    t = mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
    q = mod2pi(mod2pi(beta) - alpha - t + mod2pi(p))

    return t, p, q, mode


def dubins_path_planning_from_origin(end_x: float, end_y: float, end_yaw: float, curvature: float,
                                     step_size: float):
    """
    Dubins path planner with relative target pose description
    @param end_x: x position of end point [m]
    @param end_y: y position of end point [m]
    @param end_yaw: yaw angle of end point [rad]
    @param curvature: curvature for curve [1/m]
    @param step_size: (optional) step size between two path points [m]
    @return: x, y and yaw lists of the  of a path; length of path segments
    """
    dx = end_x
    dy = end_y
    dist = math.hypot(dx, dy)
    d = dist * curvature

    theta = mod2pi(math.atan2(dy, dx))
    alpha = mod2pi(- theta)
    beta = mod2pi(end_yaw - theta)

    planning_funcs = [left_straight_left, right_straight_right,
                      left_straight_right, right_straight_left,
                      right_left_right, left_right_left]

    best_cost = float("inf")
    bt, bp, bq, best_mode = None, None, None, None

    for planner in planning_funcs:
        t, p, q, mode = planner(alpha, beta, d)
        if t is None:
            continue

        cost = (abs(t) + abs(p) + abs(q))
        if best_cost > cost:
            bt, bp, bq, best_mode = t, p, q, mode
            best_cost = cost
    lengths = [bt, bp, bq]
    x_list, y_list, yaw_list, directions = generate_local_course(sum(lengths),
                                                                 lengths,
                                                                 best_mode,
                                                                 curvature,
                                                                 step_size)

    lengths = [length / curvature for length in lengths]

    return x_list, y_list, yaw_list, best_mode, lengths, best_cost


def interpolate(ind: int, length: float, mode: str, max_curvature: float, origin_x: float, origin_y: float,
                origin_yaw: float, path_x: List[float], path_y: List[float],
                path_yaw: List[float], directions: List[float]) \
        -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Compute intermediate positions and angles for canonical paths
    @param ind: Index of the path point
    @param length: Distance traveled
    @param mode: Whether S, R, L
    @param max_curvature: Circles curvature
    @param origin_x: Starting position x
    @param origin_y: Starting position y
    @param origin_yaw: Starting yaw angle
    @param path_x: Path x-positions to be updated
    @param path_y: Path y-positions to be updated
    @param path_yaw: Path yaw-angles to be updated
    @param directions: Path point directions
    @return: updated path x-positions, updated path y-positions, updated path yaw-angles, updated path-directions
    """
    if mode == "S":
        path_x[ind] = origin_x + length / max_curvature * math.cos(origin_yaw)
        path_y[ind] = origin_y + length / max_curvature * math.sin(origin_yaw)
        path_yaw[ind] = origin_yaw
    else:  # curve
        ldx = math.sin(length) / max_curvature
        ldy = 0.0
        if mode == "L":  # left turn
            ldy = (1.0 - math.cos(length)) / max_curvature
        elif mode == "R":  # right turn
            ldy = (1.0 - math.cos(length)) / -max_curvature
        gdx = math.cos(-origin_yaw) * ldx + math.sin(-origin_yaw) * ldy
        gdy = -math.sin(-origin_yaw) * ldx + math.cos(-origin_yaw) * ldy
        path_x[ind] = origin_x + gdx
        path_y[ind] = origin_y + gdy

    if mode == "L":  # left turn
        path_yaw[ind] = origin_yaw + length
    elif mode == "R":  # right turn
        path_yaw[ind] = origin_yaw - length

    if length > 0.0:
        directions[ind] = 1
    else:
        directions[ind] = -1

    return path_x, path_y, path_yaw, directions


def generate_local_course(total_length: float, lengths: List[float], modes: List[str], max_curvature: float,
                          step_size: float):
    """
    Generate path with step_size resolution and following the instruction provided by lengths and modes.
    @param total_length: Total path degree-length
    @param lengths: Degree-length of each step
    @param modes: Three modes as "S" or "R" or "L"
    @param max_curvature: Curve maximal curvature
    @param step_size: Resolution
    @return: path x-, y- positions and yaw angles, path directions
    """

    n_point = math.trunc(total_length / step_size) + len(lengths) + 4

    p_x = [0.0 for _ in range(n_point)]
    p_y = [0.0 for _ in range(n_point)]
    p_yaw = [0.0 for _ in range(n_point)]
    directions = [0.0 for _ in range(n_point)]
    ind = 1

    if lengths[0] > 0.0:
        directions[0] = 1
    else:
        directions[0] = -1

    ll = 0.0

    for (m, length, i) in zip(modes, lengths, range(len(modes))):
        if length == 0.0:
            continue
        elif length > 0.0:
            dist = step_size
        else:
            dist = -step_size

        # set origin state
        origin_x, origin_y, origin_yaw = p_x[ind], p_y[ind], p_yaw[ind]

        ind -= 1
        if i >= 1 and (lengths[i - 1] * lengths[i]) > 0:
            pd = - dist - ll
        else:
            pd = dist - ll

        while abs(pd) <= abs(length):
            ind += 1
            p_x, p_y, p_yaw, directions = interpolate(ind, pd, m,
                                                      max_curvature,
                                                      origin_x,
                                                      origin_y,
                                                      origin_yaw,
                                                      p_x, p_y,
                                                      p_yaw,
                                                      directions)
            pd += dist

        ll = length - pd - dist  # calc remain length

        ind += 1
        p_x, p_y, p_yaw, directions = interpolate(ind, length, m,
                                                  max_curvature,
                                                  origin_x, origin_y,
                                                  origin_yaw,
                                                  p_x, p_y, p_yaw,
                                                  directions)

    if len(p_x) <= 1:
        return [], [], [], []

    # remove unused data
    while len(p_x) >= 1 and p_x[-1] == 0.0:
        p_x.pop()
        p_y.pop()
        p_yaw.pop()
        directions.pop()

    return p_x, p_y, p_yaw, directions


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):

    if not isinstance(x, float) and not isinstance(x, int):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw), fc=fc,
                  ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)
