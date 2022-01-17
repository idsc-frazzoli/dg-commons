import copy
import time
from dataclasses import dataclass, field
from dg_commons import PlayerName, SE2Transform
from dg_commons.sim import PlayerObservations
from typing import MutableMapping, Dict, Optional, Tuple, Callable, List
from shapely.geometry import Polygon, LineString
from dg_commons.geo import SE2_apply_T2, T2value, SE2value
from dg_commons import X, U
from scipy.integrate import solve_ivp
import math
from dg_commons.sim.models.vehicle import VehicleParameters, VehicleGeometry, VehicleModel, VehicleCommands
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import patches
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from shapely import geometry
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.maps.lanes import DgLanelet
from geometry import translation_angle_from_SE2


@dataclass
class SituationObservations:
    """ This dataclass formalize the observations each agent is getting to formulate a decision on the situation """

    my_name: PlayerName
    """ My PlayerName """

    dt_commands: float
    """ Time interval between two subsequent calls """

    distances: Tuple[float, float]
    """ Look ahead distance, safety distance"""

    planned_path: Tuple[DgLanelet, float] = (None, None)
    """ Long term path and current position on it """

    agents: MutableMapping[PlayerName, PlayerObservations] = field(default_factory=dict)
    """ PlayerObservations for each player """

    rel_poses: Dict[PlayerName, SE2Transform] = field(default_factory=dict)
    """ Relative poses between me and the other players """

    static_obstacles: List[StaticObstacle] = field(default_factory=list)
    """ Static obstacles """


def relative_velocity(my_vel: float, other_vel: float, transform: SE2value) -> float:
    """
    @param my_vel: My absolute velocity
    @param other_vel: Other Absolute velocity
    @param transform: Other pose wrt to mine
    @return: Absolute relative velocity
    """
    other_vel_wrt_other = np.array([float(other_vel), 0.0])
    other_vel_wrt_myself = SE2_apply_T2(transform, other_vel_wrt_other)
    return my_vel - other_vel_wrt_myself[0]


def l_w_from_rectangle(occupacy: Polygon) -> Tuple[float, float]:
    """
    @param occupacy: Polygon describing a rectangle
    @return: Length and width of the rectangle
    """
    x, y = occupacy.exterior.xy[0], occupacy.exterior.xy[1]
    length = np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[3], y[3]]))
    width = np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[1], y[1]]))
    return length, width


def intentions_prediction(look_ahead_distance: float, path: DgLanelet, along_lane: float):
    """
    Generate a polygon covering the lanes involved in the short term plan and a list of polygons subdividing
    these lanes.
    @param look_ahead_distance: How far in current plan to look ahead
    @param path: Current plan
    @param along_lane: Current position on path
    @return: Short term polygon and subdivision list
    """
    l_polygon, r_polygon = [], []

    def polygon_data(position: np.ndarray, orientation: float, width: float):
        normal = np.array([-math.sin(orientation), math.cos(orientation)]) * width / 2
        l_polygon.append(tuple(position + normal))
        r_polygon.append(tuple(position - normal))

    start_beta = path.beta_from_along_lane(along_lane)
    end_beta = path.beta_from_along_lane(along_lane + look_ahead_distance)
    betas = [start_beta] + [i for i in range(math.ceil(start_beta), math.floor(end_beta) + 1)] + [end_beta]
    betas = list(dict.fromkeys(betas))

    for beta in betas:
        p = path.center_point(beta)
        r = path.radius(beta)

        pos, ang = translation_angle_from_SE2(p)
        polygon_data(pos, ang, r)

    polygons: Dict[float, Polygon] = {}
    n = len(betas)
    for i in range(n - 1):
        p1, p2 = l_polygon[i], l_polygon[i + 1]
        p3, p4 = r_polygon[i + 1], r_polygon[i]
        dist = (np.linalg.norm(np.array(p2) - np.array(p1)) + np.linalg.norm(np.array(p3) - np.array(p4))) / 2
        along_lane = path.along_lane_from_beta(betas[i])
        polygons[along_lane + 0.5*dist] = Polygon((p1, p2, p3, p4, p1))

    r_polygon.reverse()

    polygon: Polygon = Polygon(tuple(l_polygon + r_polygon + [l_polygon[0]]))
    return polygon, polygons


def states_prediction(current_state: X, time_span: float, dt: float,
                      vg=VehicleGeometry.default_car(), vp: VehicleParameters = VehicleParameters.default_car())\
        -> List[X]:
    """
    Kinematic Model x' = f(x, t), x0 = current_state is integrated until time_span and
    [x(0), x(dt), x(2*dt), ..., x(time_span)] is returned.
    @param current_state: Current state of the vehicle
    @param time_span: How far in time it should be looked
    @param dt: Time interval between two returned states
    @param vg: Vehicle geometry
    @param vp: Vehicle Parameters
    @return: List of states
    """
    model = VehicleModel(current_state, vg, vp)
    commands = VehicleCommands(acc=0.0, ddelta=0.0)
    n, rest = int(time_span/dt), time_span % dt
    dts = [dt for _ in range(n)] + [rest]

    states = [current_state]
    for dt in dts:
        model.update(commands, dt)
        state = model.get_state()
        states.append(state)

    return states


SAFETY_FACTOR = 0.5
""" 
This is an enlargement factor. When the vehicle area is projected ahead of time to evaluate the collision risk, 
not just its area lenght * width is considered but (length + 2 * SAFETY_FACTOR) * (width + SAFETY_FACTOR)."""


def occupancy_prediction(current_state: X, time_span: float,
                         vg=VehicleGeometry.default_car(), vp: VehicleParameters = VehicleParameters.default_car()) \
        -> Polygon:
    """
    Computes the polygon describing the integral area occupied by a car starting at current_state and proceeding with
    constant steering angle and velocity up until time_span.
    The underlying vehicle model is the kinematic bicycle model

    @param current_state: Current vehicle state
    @param time_span: How far in time it should be looked
    @param vg: Vehicle Geometry
    @param vp: Vehicle Parameters
    @return: Integral Area
    """
    dt = 0.3
    l_polygon, r_polygon = [], []

    width = vg.width + SAFETY_FACTOR
    lf = vg.lf + vg.bumpers_length[0] + SAFETY_FACTOR
    lr = vg.lr + vg.bumpers_length[1] + SAFETY_FACTOR

    def rear_state(state):
        position = np.array([state.x, state.y])
        vec = np.array([math.cos(state.theta), math.sin(state.theta)]) * lr
        r_position = position - vec
        return np.array([r_position[0], r_position[1], state.theta])

    def polygon_data(res: np.ndarray):
        theta, pos = res[2], res[:2]
        vec = np.array([-math.sin(theta), math.cos(theta)])*width/2
        l_polygon.append(tuple(pos + vec))
        r_polygon.insert(0, tuple(pos - vec))

    states = states_prediction(current_state, time_span, dt, vg, vp)
    for state in states:
        polygon_data(rear_state(state))

    if current_state.vx >= 0:
        final_state = np.array([state.x + math.cos(state.theta) * lf,
                                state.y + math.sin(state.theta) * lf,
                                state.theta])

        polygon_data(final_state)

    polygon = Polygon(tuple(l_polygon + r_polygon + [l_polygon[0]]))
    return polygon, states


def occupancy_prediction_var(current_state: X, time_span: float,
                         vg=VehicleGeometry.default_car(), vp: VehicleParameters = VehicleParameters.default_car()) \
        -> Polygon:
    """
    Computes the polygon describing the integral area occupied by a car starting at current_state and proceeding with
    constant steering angle and velocity up until time_span.
    The underlying vehicle model is the kinematic bicycle model

    @param current_state: Current vehicle state
    @param time_span: How far in time it should be looked
    @param vg: Vehicle Geometry
    @param vp: Vehicle Parameters
    @return: Integral Area
    """
    dt = 0.3
    l_polygon, r_polygon = [], []
    positions = []

    width = vg.width + SAFETY_FACTOR
    lf = vg.lf + vg.bumpers_length[0] + SAFETY_FACTOR
    lr = vg.lr + vg.bumpers_length[1] + SAFETY_FACTOR

    def rear_state(state):
        position = np.array([state.x, state.y])
        vec = np.array([math.cos(state.theta), math.sin(state.theta)]) * lr
        r_position = position - vec
        return np.array([r_position[0], r_position[1]])

    states = states_prediction(current_state, time_span, dt, vg, vp)
    for state in states:
        positions.append(rear_state(state))

    if current_state.vx >= 0:
        final_state = np.array([state.x + math.cos(state.theta) * lf,
                                state.y + math.sin(state.theta) * lf])

        positions.append(final_state)

    polygon = LineString(positions).buffer(width)
    return polygon, states


def entry_exit_t(intersection: Polygon, current_state, occupacy: Polygon, safety_t, vel,
                 vg=VehicleGeometry.default_car(), vp=VehicleParameters.default_car(), tol=0.01) \
        -> Tuple[float, float]:
    """
    Computes with a tolerance of tol when a certain vehicle enters and exites intersection based on predictions
    formulated with its current_state and assuming the steering angle and the velocity to be constant.
    The underlying vehicle model is the kinematic bicycle model.

    @param intersection: Query area
    @param current_state: Current vehicle state
    @param occupacy: Vehicle occupancy
    @param safety_t: Safety time braking
    @param vel: Vehicle velocity
    @param vg: Vehicle Geometry
    @param vp: Vehicle Parameters
    @param tol: Search tolerance [s]
    @return: Prediction time vehicle enters and exites the query area
    """
    stopped_inside: bool = vel <= 10e-6
    going: bool = vel > 10e-6
    assert going or stopped_inside
    if stopped_inside:
        return 0, safety_t

    length = intersection.length
    distance = max(occupacy.distance(intersection) - 1.5 * SAFETY_FACTOR, 0)

    min_t = distance/vel
    max_t = min(min_t + 2 * length / vel, safety_t)

    def find(t: float, state: X, dt: float, condition: Callable):
        t_of_interest: Optional[float] = None

        def find_temp(t_temp, state_temp, dt_temp):
            while (t_temp - max_t) < 10e-6:
                pred, states = occupancy_prediction(state_temp, dt_temp, vg=vg, vp=vp)
                inter = pred.intersection(intersection)
                if condition(inter):
                    return t_temp, state_temp
                t_temp += dt
                state_temp = states[-1]
            return None, None

        while True:
            t_of_interest, state = find_temp(t, state, dt)
            if t_of_interest is None:
                return None, None
            t = t_of_interest
            dt = dt / 2
            if dt <= tol/2:
                break

        return t_of_interest, state

    t = min_t
    initial_state = states_prediction(current_state, min_t, min_t)[-1] if min_t != 0 else current_state
    dt = (max_t - min_t)/2
    entry_t, state = find(t, initial_state, dt, lambda inter: not inter.is_empty)
    if entry_t is None:
        entry_t = safety_t - SAFETY_FACTOR / vel
        state = states_prediction(initial_state, entry_t - min_t, entry_t - min_t)[-1]

    dt = (max_t - entry_t)/2
    exit_t, _ = find(entry_t, state, dt, lambda inter: inter.is_empty)

    if exit_t is None:
        exit_t = safety_t

    return entry_t, exit_t


class SituationPolygons:
    """ This class is used to generate the data required to visualize situations """

    @dataclass
    class PolygonClass:
        """ This class describes help visualizing polygons through their nature """
        collision: bool = False
        following: bool = False
        dangerous_zone: bool = False

        def __post_init__(self):
            assert self.collision or self.dangerous_zone or self.following

        def get_color(self):
            if self.collision:
                return 'r'
            elif self.dangerous_zone:
                return 'y'
            elif self.following:
                return 'orange'

        def get_zorder(self):
            if self.collision:
                return 1
            elif self.dangerous_zone:
                return 2
            elif self.following:
                return 3

    def __init__(self, plot: bool):
        self.plot = plot
        self.current_frame: List[Polygon] = []
        self.current_class: List[SituationPolygons.PolygonClass] = []

    def plot_polygon(self, p: Polygon, polygon_class: PolygonClass) -> None:
        """
        Plot polygon p in the current frame with the characteristics provided by its class.
        @param p: Polygon to plot
        @param polygon_class: Class of p
        @return: None
        """
        if p.is_empty or not self.plot:
            return

        self.current_frame.append(p)
        self.current_class.append(polygon_class)

    def next_frame(self) -> Tuple[List[Polygon], List[PolygonClass]]:
        """
        Changes frame and returns the data of the current one.
        @return: Polygons and polygon classes of this frame
        """
        current_frame = self.current_frame
        current_classes = self.current_class
        assert len(current_classes) == len(current_frame)
        self.current_frame = []
        self.current_class = []
        return current_frame, current_classes
