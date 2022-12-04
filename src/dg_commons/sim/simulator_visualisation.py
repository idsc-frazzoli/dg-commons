from abc import ABC, abstractmethod
from dataclasses import asdict
from enum import IntEnum
from math import inf
from typing import Sequence, Tuple, Generic, Optional, List, Union

import numpy as np
from commonroad.visualization.draw_params import MPDrawParams
from commonroad.visualization.mp_renderer import MPRenderer
from decorator import contextmanager
from geometry import SE2_from_xytheta, SE2value
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Circle
from zuper_commons.types import ZValueError

from dg_commons import Color, transform_xy, apply_SE2_to_shapely_geo
from dg_commons import PlayerName, X, U
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.planning.trajectory import Trajectory
from dg_commons.sim.models.obstacles_dyn import DynObstacleState, DynObstacleModel
from dg_commons.sim.models.pedestrian import PedestrianState, PedestrianGeometry
from dg_commons.sim.models.spacecraft import SpacecraftState
from dg_commons.sim.models.spacecraft_structures import SpacecraftGeometry
from dg_commons.sim.models.vehicle import VehicleState, VehicleGeometry
from dg_commons.sim.models.vehicle_ligths import LightsColors
from dg_commons.sim.simulator import SimContext

__all__ = ["SimRenderer", "plot_vehicle", "plot_pedestrian", "plot_trajectories"]


class SimRendererABC(Generic[X, U], ABC):
    """An artist that can draw the game."""

    @abstractmethod
    def plot_arena(self, ax: Axes):
        """Context manager. Plots the arena."""
        pass

    @abstractmethod
    def plot_player(
        self, ax: Axes, player_name: PlayerName, state: X, lights_colors: LightsColors, alpha: float = 1.0, box=None
    ):
        """Draw the player at a certain state doing certain commands (if given)"""
        pass


class ZOrders(IntEnum):
    GOAL = 30
    ENV_OBSTACLE = 32
    LIGHTS = 34
    MODEL = 35
    PLAYER_NAME = 40
    TRAJECTORY = 45
    TRAJECTORY_MARKER = 46
    TIME_TEXT = 50


class SimRenderer(SimRendererABC):
    """Visualization for the trajectory games"""

    def __init__(self, sim_context: SimContext, ax: Axes = None, *args, **kwargs):
        self.sim_context = sim_context
        self.commonroad_renderer: MPRenderer = MPRenderer(ax=ax, *args, **kwargs)
        self.shapely_viz = ShapelyViz(ax=self.commonroad_renderer.ax)
        self.draw_params: MPDrawParams = MPDrawParams()

    @contextmanager
    def plot_arena(self, ax: Axes):
        if self.sim_context.dg_scenario.scenario:
            self.draw_params.lanelet_network.traffic_light.draw_traffic_lights = True
            self.sim_context.dg_scenario.lanelet_network.draw(self.commonroad_renderer, draw_params=self.draw_params)
            self.commonroad_renderer.render()
        for s_obstacle in self.sim_context.dg_scenario.static_obstacles.values():
            self.shapely_viz.add_shape(s_obstacle.shape, color=s_obstacle.geometry.color, zorder=ZOrders.ENV_OBSTACLE)
        for p, goal in self.sim_context.missions.items():
            self.shapely_viz.add_shape(goal.get_plottable_geometry(), color="orange", zorder=ZOrders.GOAL, alpha=0.5)
        yield

    def plot_player(
        self,
        ax: Axes,
        player_name: PlayerName,
        state: X,
        lights_colors: Optional[LightsColors],
        model_poly: Optional[List[Polygon]] = None,
        lights_patches: Optional[List[Circle]] = None,
        alpha: float = 0.6,
        plot_wheels: bool = False,
        plot_lights: bool = False,
    ) -> Tuple[List[Polygon], List[Circle]]:
        """Draw the player the state."""
        # todo make it nicer with a map of plotting functions based on the state type

        mg = self.sim_context.models[player_name].model_geometry
        if issubclass(type(state), VehicleState):
            return plot_vehicle(
                ax=ax,
                player_name=player_name,
                state=state,
                lights_colors=lights_colors,
                vg=mg,
                alpha=alpha,
                vehicle_poly=model_poly,
                lights_patches=lights_patches,
                plot_wheels=plot_wheels,
                plot_ligths=plot_lights,
            )
        elif issubclass(type(state), PedestrianState):
            ped_poly = plot_pedestrian(
                ax=ax,
                player_name=player_name,
                state=state,
                pg=mg,
                alpha=alpha,
                ped_poly=model_poly,
            )
            return ped_poly, []
        elif issubclass(type(state), SpacecraftState):
            scraft_poly = plot_spacecraft(
                ax=ax,
                player_name=player_name,
                state=state,
                sg=mg,
                alpha=alpha,
                scraft_poly=model_poly,
            )
            return scraft_poly, []
        elif issubclass(type(state), DynObstacleState):
            # todo merge with shapely viz
            dyn_obs_model: DynObstacleModel = self.sim_context.models[player_name]
            shape = dyn_obs_model.shape
            geo = self.sim_context.models[player_name].model_geometry
            if model_poly is None:
                model_poly = ax.fill([], [], color=geo.color, alpha=alpha, zorder=ZOrders.MODEL)
            q = SE2_from_xytheta((state.x, state.y, state.psi))
            transformed_shape = apply_SE2_to_shapely_geo(shape, q)
            model_poly[0].set_xy(np.array(transformed_shape.exterior.xy).T)
            return model_poly, []
        else:
            raise ZValueError(msg=f"Unknown state type, {type(state)}", state=state)

    def plot_trajectories(
        self,
        ax: Axes,
        player_name: PlayerName,
        trajectories: Sequence[Trajectory],
        traj_lines: Optional[List[LineCollection]] = None,
        traj_points: Optional[List[PathCollection]] = None,
        colors: Optional[List[Color]] = None,
        width: float = 1,
        alpha: float = 1,
    ) -> Tuple[List[LineCollection], List[PathCollection]]:
        mg = self.sim_context.models[player_name].model_geometry
        assert colors is None or len(colors) == len(trajectories)
        colors = mg.color if colors is None else colors
        return plot_trajectories(
            ax=ax,
            trajectories=trajectories,
            traj_lines=traj_lines,
            traj_points=traj_points,
            colors=colors,
            width=width,
            alpha=alpha,
        )


def plot_trajectories(
    ax: Axes,
    trajectories: Sequence[Trajectory],
    traj_lines: Optional[List[LineCollection]] = None,
    traj_points: Optional[List[PathCollection]] = None,
    colors: Union[List[Color], Color] = None,
    width: float = 1,
    alpha: float = 1,
) -> Tuple[List[LineCollection], List[PathCollection]]:
    segments, mcolor = [], []
    for traj in trajectories:
        sampled_traj = np.vstack([[x.x, x.y, x.vx] for x in traj.values])
        segments.append(sampled_traj[:, :2])
        mcolor.append(sampled_traj[:, 2])  # fixme marker color functionality not available yet

    if traj_lines is None:
        traj_lines = LineCollection(
            segments=[], colors=colors, linewidths=width, alpha=alpha, zorder=ZOrders.TRAJECTORY
        )
        ax.add_collection(traj_lines)
        size = np.linalg.norm(ax.bbox.size) / 1000
        traj_points = ax.scatter([], [], s=size, c="r", zorder=ZOrders.TRAJECTORY_MARKER)
        ax.add_collection(traj_points)
    traj_lines.set_segments(segments=segments)
    traj_lines.set_color(colors)
    traj_points.set_offsets(np.concatenate(segments))
    # traj_points.set_facecolor(mcolor) # todo adjust color based on velocity
    # https://stackoverflow.com/questions/23966121/updating-the-positions-and-colors-of-pyplot-scatter
    return traj_lines, traj_points


def plot_vehicle(
    ax: Axes,
    player_name: PlayerName,
    state: VehicleState,
    lights_colors: LightsColors,
    vg: VehicleGeometry,
    alpha: float,
    vehicle_poly: Optional[List[Polygon]] = None,
    lights_patches: Optional[List[Circle]] = None,
    plot_wheels: bool = False,
    plot_ligths: bool = False,
    **style_kwargs,
) -> Tuple[List[Polygon], List[Circle]]:
    """"""
    vehicle_outline: Sequence[Tuple[float, float], ...] = vg.outline
    vehicle_color: Color = vg.color
    q = SE2_from_xytheta((state.x, state.y, state.psi))
    if vehicle_poly is None:
        vehicle_box = ax.fill([], [], color=vehicle_color, alpha=alpha, zorder=ZOrders.MODEL, **style_kwargs)[0]
        vehicle_poly = [
            vehicle_box,
        ]
        x4, y4 = transform_xy(q, ((0, 0),))[0]
        ax.text(
            x4, y4, player_name, zorder=ZOrders.PLAYER_NAME, horizontalalignment="center", verticalalignment="center"
        )
        if plot_wheels:
            wheels_boxes = [
                ax.fill([], [], color="k", alpha=alpha, zorder=ZOrders.MODEL)[0] for _ in range(vg.n_wheels)
            ]
            vehicle_poly.extend(wheels_boxes)
        if plot_ligths:
            lights_patches = _plot_lights(ax=ax, q=q, lights_colors=lights_colors, vg=vg)

    outline = transform_xy(q, vehicle_outline)
    vehicle_poly[0].set_xy(outline)

    if plot_wheels:
        wheels_outlines = vg.get_rotated_wheels_outlines(state.delta)
        wheels_outlines = [q @ w_outline for w_outline in wheels_outlines]
        for w_idx, wheel in enumerate(vehicle_poly[1:]):
            xy_poly = wheels_outlines[w_idx][:2, :].T
            wheel.set_xy(xy_poly)

    if plot_ligths:
        light_dict = asdict(lights_colors)
        for i, name in enumerate(vg.lights_position):
            light_color = light_dict[name]
            position = vg.lights_position[name]
            x2, y2 = transform_xy(q, (position,))[0]
            lights_patches[i].center = x2, y2
            lights_patches[i].set_color(light_color)

    return vehicle_poly, lights_patches


def _plot_lights(ax: Axes, q: SE2value, lights_colors: LightsColors, vg: VehicleGeometry) -> List[Circle]:
    radius_light = 0.04 * vg.width
    light_dict = asdict(lights_colors)
    patches = []
    for name in vg.lights_position:
        light_color = light_dict[name]
        position = vg.lights_position[name]
        x2, y2 = transform_xy(q, (position,))[0]
        patch = Circle((x2, y2), radius=radius_light, color=light_color, zorder=ZOrders.LIGHTS)
        patches.append(patch)
        ax.add_patch(patch)
    return patches


def plot_history(ax: Axes, state: VehicleState, vg: VehicleGeometry, traces: Optional[Line2D] = None):
    if traces is None:
        (trace,) = ax.plot([], [], ",-", lw=1)
    # todo similar to https://matplotlib.org/stable/gallery/animation/double_pendulum.html#sphx-glr-gallery-animation-double-pendulum-py


def plot_pedestrian(
    ax: Axes,
    player_name: PlayerName,
    state: PedestrianState,
    pg: PedestrianGeometry,
    alpha: float,
    ped_poly: Optional[List[Polygon]],
) -> List[Polygon]:
    q = SE2_from_xytheta((state.x, state.y, state.psi))
    if ped_poly is None:
        pedestrian_box = ax.fill([], [], color=pg.color, alpha=alpha, zorder=ZOrders.MODEL)[0]
        ped_poly = [
            pedestrian_box,
        ]
        x4, y4 = transform_xy(q, ((0, 0),))[0]
        ax.text(
            x4, y4, player_name, zorder=ZOrders.PLAYER_NAME, horizontalalignment="center", verticalalignment="center"
        )
    ped_outline: Sequence[Tuple[float, float], ...] = pg.outline
    outline_xy = transform_xy(q, ped_outline)
    ped_poly[0].set_xy(outline_xy)
    return ped_poly


def plot_spacecraft(
    ax: Axes,
    player_name: PlayerName,
    state: SpacecraftState,
    sg: SpacecraftGeometry,
    alpha: float,
    scraft_poly: Optional[List[Polygon]],
) -> List[Polygon]:
    q = SE2_from_xytheta((state.x, state.y, state.psi))
    if scraft_poly is None:
        spacecraft_box = ax.fill([], [], color=sg.color, alpha=alpha, zorder=ZOrders.MODEL)[0]
        scraft_poly = [
            spacecraft_box,
        ]
        x4, y4 = transform_xy(q, ((0, 0),))[0]
        ax.text(
            x4, y4, player_name, zorder=ZOrders.PLAYER_NAME, horizontalalignment="center", verticalalignment="center"
        )
        thrusters_boxes = [
            ax.fill([], [], color="k", alpha=alpha, zorder=ZOrders.MODEL)[0] for _ in range(sg.n_thrusters)
        ]
        scraft_poly.extend(thrusters_boxes)
    # body
    ped_outline: Sequence[Tuple[float, float], ...] = sg.outline
    outline_xy = transform_xy(q, ped_outline)
    scraft_poly[0].set_xy(outline_xy)
    # thrusters
    thrusters_outline = np.array([transform_xy(q, t_outline) for t_outline in sg.thrusters_outline_in_body_frame])
    for t_idx, thruster in enumerate(scraft_poly[1:]):
        xy_poly = thrusters_outline[t_idx]
        thruster.set_xy(xy_poly)
    return scraft_poly


def approximate_bounding_box_players(obj_list: Sequence[X]) -> Union[Sequence[List], None]:
    minmax = [[inf, -inf], [inf, -inf]]
    for state in obj_list:
        x, y = state.x, state.y
        for i in range(2):
            xory = x if i == 0 else y
            if xory < minmax[i][0]:
                minmax[i][0] = xory
            if xory > minmax[i][1]:
                minmax[i][1] = xory
    if not (max(minmax) == inf and min(minmax) == -inf):
        for i in range(2):
            assert minmax[i][0] <= minmax[i][1]
            minmax[i][0] -= 10
            minmax[i][1] += 10
        return minmax
    return None


def approximate_bounding_box_single_player(state: X) -> Union[Sequence[List], None]:
    minmax = [[inf, -inf], [inf, -inf]]

    x, y = state.x, state.y
    for i in range(2):
        xory = x if i == 0 else y
        if xory < minmax[i][0]:
            minmax[i][0] = xory
        if xory > minmax[i][1]:
            minmax[i][1] = xory

    if not (max(minmax) == inf and min(minmax) == -inf):
        for i in range(2):
            assert minmax[i][0] <= minmax[i][1]
            minmax[i][0] -= 10
            minmax[i][1] += 10
        return minmax
    return None
