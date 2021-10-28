import copy
from dataclasses import dataclass
from dg_commons import PlayerName, SE2Transform
from dg_commons.sim import PlayerObservations
from typing import MutableMapping, Dict, Optional, Tuple, Callable, List
from shapely.geometry import Polygon
from dg_commons.geo import SE2_apply_T2, T2value
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


@dataclass
class SituationObservations:
    my_name: PlayerName

    agents: Optional[MutableMapping[PlayerName, PlayerObservations]] = None

    rel_poses: Optional[Dict[PlayerName, SE2Transform]] = None

    dt_commands: Optional[float] = None


def relative_velocity(my_vel: float, other_vel: float, transform):
    print(other_vel)
    other_vel_wrt_other = [float(other_vel), 0.0]
    other_vel_wrt_myself = SE2_apply_T2(transform, other_vel_wrt_other)
    return my_vel - other_vel_wrt_myself[0]


def l_w_from_rectangle(occupacy: Polygon):
    x, y = occupacy.exterior.xy[0], occupacy.exterior.xy[1]
    length = np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[3], y[3]]))
    width = np.linalg.norm(np.array([x[0], y[0]]) - np.array([x[1], y[1]]))
    return length, width


def states_prediction(current_state: X, time_span: float, dt: float,
                      vg=VehicleGeometry.default_car(), vp: VehicleParameters = VehicleParameters.default_car()):

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


def occupancy_prediction(current_state: X, time_span: float, occupacy: Polygon,
                         vg=VehicleGeometry.default_car(), vp: VehicleParameters = VehicleParameters.default_car()):
    dt = 0.2
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

    final_state = np.array([state.x + math.cos(state.theta) * lf, \
                            state.y + math.sin(state.theta) * lf, \
                            state.theta])
    polygon_data(final_state)
    polygon = Polygon(tuple(l_polygon + r_polygon + [l_polygon[0]]))
    return polygon, states


def entry_exit_t(intersection: Polygon, current_state, occupacy: Polygon, safety_t, vel,
                 vg=VehicleGeometry.default_car(), vp=VehicleParameters.default_car(), tol=0.01):
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
                pred, states = occupancy_prediction(state_temp, dt_temp, occupacy, vg=vg, vp=vp)
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
    state = states_prediction(current_state, min_t, min_t)[-1] if min_t != 0 else current_state
    dt = (max_t - min_t)/2
    entry_t, state = find(t, state, dt, lambda inter: not inter.is_empty)
    '''entry_t = 0 if entry_t is None else entry_t
    state = current_state if state is None else state'''
    assert entry_t is not None

    dt = (max_t - entry_t)/2
    exit_t, _ = find(entry_t, state, dt, lambda inter: inter.is_empty)

    if exit_t is None:
        exit_t = safety_t

    return entry_t, exit_t


class SituationPolygons:
    @dataclass
    class PolygonClass:
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

    def plot_polygon(self, p: Polygon, polygon_class: PolygonClass):
        if p.is_empty or not self.plot:
            return

        self.current_frame.append(p)
        self.current_class.append(polygon_class)

    def next_frame(self):
        current_frame = self.current_frame
        current_classes = self.current_class
        assert len(current_classes) == len(current_frame)
        self.current_frame = []
        self.current_class = []
        return current_frame, current_classes


'''class PolygonPlotter:
    @dataclass
    class PolygonClass:
        car: bool = False
        dangerous_zone: bool = False
        conflict_area: bool = False

        def __post_init__(self):
            assert self.car or self.dangerous_zone or self.conflict_area

        def get_color(self):
            if self.car:
                return 'b'
            elif self.dangerous_zone:
                return 'lightgray'
            elif self.conflict_area:
                return 'r'

        def get_zorder(self):
            if self.car:
                return 1
            elif self.dangerous_zone:
                return 2
            elif self.conflict_area:
                return 3

    def __init__(self, plot: bool):
        self.counter = 0
        self.plot = plot
        self.frames = {"Frame": [], "Class": []}
        self.current_frame = [[], []]
        self.current_class = []
        self.min_max_x = [math.inf, -math.inf]
        self.min_max_y = [math.inf, -math.inf]
        self.max_n_items = 0
        self.previous_coll = None

    def plot_polygon(self, p: Polygon,  polygon_class: PolygonClass):
        if p.is_empty or not self.plot:
            return

        x, y = list(p.exterior.coords.xy[0][:]), list(p.exterior.coords.xy[1][:])
        min_x, max_x, min_y, max_y = min(x), max(x), min(y), max(y)
        self.min_max_x[0] = min_x if min_x < self.min_max_x[0] else self.min_max_x[0]
        self.min_max_x[1] = max_x if max_x > self.min_max_x[1] else self.min_max_x[1]
        self.min_max_y[0] = min_y if min_y < self.min_max_y[0] else self.min_max_y[0]
        self.min_max_y[1] = max_y if max_y > self.min_max_y[1] else self.min_max_y[1]
        self.current_frame[0].append(x)
        self.current_frame[1].append(y)
        self.current_class.append(polygon_class)

    def next_frame(self):
        n_items = len(self.current_frame[0])
        self.frames["Frame"].append(self.current_frame)
        self.frames["Class"].append(self.current_class)
        self.max_n_items = n_items if n_items > self.max_n_items else self.max_n_items
        self.current_frame = [[], []]

    def save_animation(self, title: str):
        if not self.plot:
            return

        n_frames, enlargement_factor = len(self.frames["Frame"]), 3
        fig = plt.figure()
        self.min_max_x = [self.min_max_x[0] - enlargement_factor, self.min_max_x[1] + enlargement_factor]
        self.min_max_y = [self.min_max_y[0] - enlargement_factor, self.min_max_y[1] + enlargement_factor]
        ax = plt.axes(xlim=tuple(self.min_max_x), ylim=tuple(self.min_max_y))
        plt.gca().set_aspect('equal', adjustable='box')

        polygons = []
        for _ in range(self.max_n_items):
            polygon = patches.Polygon(np.array([[0, 0]]), animated=True)
            polygons.append(polygon)

        def init():
            for polygon in polygons:
                polygon.set_xy(np.array([[0, 0]]))
            return polygons

        def animate(j):
            frame = self.frames["Frame"][j]
            classes = self.frames["Class"][j]
            if self.previous_coll is not None:
                self.previous_coll.remove()

            assert len(frame[0]) == len(frame[1])
            n = len(frame[0])

            for i in range(self.max_n_items):
                if i < n:
                    x = np.array([frame[0][i]]).T
                    y = np.array([frame[1][i]]).T
                    xy = np.concatenate((x, y), axis=1)
                    polygons[i].set_xy(xy)
                    polygons[i].set_zorder(classes[i].get_zorder())
                    polygons[i].set_color(classes[i].get_color())
                else:
                    polygons[i].set_xy(np.array([[0, 0]]))

            p = PatchCollection(polygons, alpha=0.4, match_original=True)
            ax.add_collection(p)
            self.previous_coll = p

            return polygons

        Writer = animation.writers['pillow']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=n_frames, interval=20, blit=True)

        dir = os.path.join("out", "situations")
        if not os.path.isdir(dir):
            os.makedirs(dir)

        name = 'emergency.gif' if title == '' else title + ".gif"
        anim.save(os.path.join(dir, name), writer=writer)'''
