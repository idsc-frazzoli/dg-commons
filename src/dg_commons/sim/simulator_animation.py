import math
from itertools import chain
from typing import Mapping, List, Union, Optional, Sequence

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from toolz.sandbox import unzip
from zuper_commons.types import ZValueError

from dg_commons import PlayerName, X
from dg_commons import Timestamp
from dg_commons.sim import logger
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_ligths import (
    lightscmd2phases,
    get_phased_lights,
    red,
    red_more,
    LightsColors,
    LightsCmd,
)
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.simulator_structures import LogEntry
from dg_commons.sim.simulator_visualisation import SimRenderer, approximate_bounding_box_players, ZOrders
from dg_commons.time import time_function


@time_function
def create_animation(
    file_path: str,
    sim_context: SimContext,
    figsize: Optional[Union[list, tuple]] = None,
    dt: float = 30,
    dpi: int = 120,
    plot_limits: Optional[Union[str, Sequence[Sequence[float]]]] = "auto",
) -> None:
    """
    Creates an animation

    :param plot_limits:
    :param sim_context:
    :param file_path: filename of generated video (ends on .mp4/.gif/.avi, default mp4, when nothing is specified)
    :param figsize: size of the video
    :param dt: time step between frames in ms
    :param dpi: resolution of the video
    :return: None
    """
    logger.info("Creating animation...")
    sim_viz: SimRenderer = SimRenderer(sim_context, figsize=figsize)
    time_begin = sim_context.log.get_init_time()
    time_end = sim_context.log.get_last_time()
    if not time_begin < time_end:
        raise ValueError(f"Begin time {time_begin} cannot be greater than end time {time_end}")
    ax: Axes = sim_viz.commonroad_renderer.ax
    fig = ax.figure
    fig.set_tight_layout(True)
    ax.set_aspect("equal")
    # dictionaries with the handles of the plotting stuff
    states, actions, extra, texts = {}, {}, {}, {}
    traj_lines, traj_points = {}, {}
    history = {}
    # some parameters
    plot_wheels: bool = True
    plot_ligths: bool = True

    # self.f.set_size_inches(*fig_size)
    def _get_list() -> List:
        # fixme this is supposed to be an iterable of artists
        return (
            list(chain.from_iterable(states.values()))
            + list(chain.from_iterable(actions.values()))
            + list(extra.values())
            + list(traj_lines.values())
            + list(traj_points.values())
            + list(texts.values())
        )

    def init_plot():
        ax.clear()
        with sim_viz.plot_arena(ax=ax):
            init_log_entry: Mapping[PlayerName, LogEntry] = sim_context.log.at_interp(time_begin)
            for pname, plog in init_log_entry.items():
                lights_colors: LightsColors = get_lights_colors_from_cmds(init_log_entry[pname].commands, t=0)
                states[pname], actions[pname] = sim_viz.plot_player(
                    ax=ax,
                    state=plog.state,
                    lights_colors=lights_colors,
                    player_name=pname,
                    alpha=0.7,
                    plot_wheels=plot_wheels,
                    plot_ligths=plot_ligths,
                )
                if plog.extra:
                    try:
                        trajectories, tcolors = unzip(plog.extra)
                        traj_lines[pname], traj_points[pname] = sim_viz.plot_trajectories(
                            ax=ax, player_name=pname, trajectories=list(trajectories), colors=list(tcolors)
                        )
                    except:
                        logger.warn(f"Cannot plot extra", extra=plog.extra)
            adjust_axes_limits(
                ax=ax, plot_limits=plot_limits, players_states=[player.state for player in init_log_entry.values()]
            )
            texts["time"] = ax.text(
                0.02,
                0.96,
                "",
                transform=ax.transAxes,
                bbox=dict(facecolor="lightgreen", alpha=0.5),
                zorder=ZOrders.TIME_TEXT,
            )
        return _get_list()

    def update_plot(frame: int = 0):
        t: float = frame * dt / 1000.0
        logger.info(f"Plotting t = {t}\r")
        log_at_t: Mapping[PlayerName, LogEntry] = sim_context.log.at_interp(t)
        for pname, box_handle in states.items():
            lights_colors: LightsColors = get_lights_colors_from_cmds(log_at_t[pname].commands, t=t)
            states[pname], actions[pname] = sim_viz.plot_player(
                ax=ax,
                player_name=pname,
                state=log_at_t[pname].state,
                lights_colors=lights_colors,
                vehicle_poly=box_handle,
                lights_patches=actions[pname],
                plot_wheels=plot_wheels,
                plot_ligths=plot_ligths,
            )
            if log_at_t[pname].extra:
                try:
                    trajectories, tcolors = unzip(log_at_t[pname].extra)
                    traj_lines[pname], traj_points[pname] = sim_viz.plot_trajectories(
                        ax=ax,
                        player_name=pname,
                        trajectories=list(trajectories),
                        traj_lines=traj_lines[pname],
                        traj_points=traj_points[pname],
                        colors=list(tcolors),
                    )
                except:
                    logger.warn(f"Cannot plot extra", extra=log_at_t[pname].extra)
        adjust_axes_limits(
            ax=ax, plot_limits=plot_limits, players_states=[player.state for player in log_at_t.values()]
        )
        texts["time"].set_text(f"t = {t:.1f}s")
        texts["time"].set_transform(ax.transAxes)
        return _get_list()

    # Min frame rate is 1 fps
    dt = min(1000.0, dt)
    frame_count: int = int(float(time_end - time_begin) // (dt / 1000.0))
    plt.ioff()
    # Interval determines the duration of each frame in ms
    anim = FuncAnimation(fig=fig, func=update_plot, init_func=init_plot, frames=frame_count, blit=True, interval=dt)

    if not any([file_path.endswith(".mp4"), file_path.endswith(".gif"), file_path.endswith(".avi")]):
        file_path += ".mp4"
    fps = int(math.ceil(1000.0 / dt))
    interval_seconds = dt / 1000.0
    anim.save(
        file_path,
        dpi=dpi,
        writer="ffmpeg",
        fps=fps,
        # extra_args=["-g", "1", "-keyint_min", str(interval_seconds)]
    )
    logger.info("Animation saved...")
    ax.clear()


def adjust_axes_limits(
    ax: Axes, plot_limits: Union[str, Sequence[Sequence[float]]], players_states: Optional[Sequence[X]] = None
):
    if plot_limits is None:
        ax.autoscale()
    elif plot_limits == "auto":
        if plot_limits is None:
            raise ZValueError('Plotting with "auto" option requires players positions')
        players_limits = approximate_bounding_box_players(obj_list=players_states)
        if players_limits is not None:
            ax.axis(
                xmin=players_limits[0][0],
                xmax=players_limits[0][1],
                ymin=players_limits[1][0],
                ymax=players_limits[1][1],
            )
        else:
            ax.autoscale()
    else:
        # plotlimits are expected to be seq of seq of floats
        ax.set_xlim(plot_limits[0][0], plot_limits[0][1])
        ax.set_ylim(plot_limits[1][0], plot_limits[1][1])
    return


def get_lights_colors_from_cmds(cmds: VehicleCommands, t: Timestamp) -> LightsColors:
    """Note that braking lights are out of the agent's control"""
    try:
        lights_colors = lights_colors_from_lights_cmd(cmds.lights, cmds.acc, t)
    except AttributeError:
        # in case the model commands does not have lights command
        lights_colors = None
    return lights_colors


def lights_colors_from_lights_cmd(lights_cmd: LightsCmd, acc: float, t: Timestamp) -> LightsColors:
    phases = lightscmd2phases[lights_cmd]
    lights_colors = get_phased_lights(phases, float(t))
    if acc < 0:  # and cmds == NO_LIGHTS:
        if lights_colors.back_left == red:
            lights_colors.back_left = red_more
        if lights_colors.back_right == red:
            lights_colors.back_right = red_more
    return lights_colors
