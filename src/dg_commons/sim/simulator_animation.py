import math
from itertools import chain
from typing import Mapping, Union, Optional, Sequence, Iterable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from toolz.sandbox import unzip
from tqdm import tqdm
from zuper_commons.types import ZValueError
from dg_commons.planning.trajectory import Trajectory
from dg_commons import PlayerName, X, Timestamp
from dg_commons.sim import logger
from dg_commons.sim.models.vehicle import VehicleCommands, VehicleState
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


def create_animation(
    file_path: str,
    sim_context: SimContext,
    figsize: Optional[Union[list, tuple]] = None,
    dt: float = 30,
    dpi: int = 120,
    plot_limits: Union[str, Sequence[Sequence[float]], PlayerName] = "auto",
) -> None:
    """
    Creates an animation of the simulation.
    :param sim_context:
    :param file_path: filename of generated video (ends on .mp4/.gif/.avi, default mp4, when nothing is specified)
    :param figsize: size of the video
    :param dt: time step between frames in ms
    :param dpi: resolution of the video
    :param plot_limits: "auto"/"" ,hardcoded limits, or player name to focus on
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
    fig.tight_layout()
    ax.set_aspect("equal")
    # dictionaries with the handles of the plotting stuff
    states, actions, extra, texts, goals = {}, {}, {}, {}, {}
    states_pred = {}
    traj_lines, traj_points = {}, {}
    history = {}
    # some parameters
    plot_wheels: bool = True
    plot_ligths: bool = True

    # self.f.set_size_inches(*fig_size)
    def _get_list() -> list[Artist]:
        # fixme this is supposed to be an iterable of artists
        return (
            list(chain.from_iterable(states.values()))
            + list(chain.from_iterable(actions.values()))
            # + [artist for artist_tuple in list(chain.from_iterable(states_pred.values())) for artist in artist_tuple ]
            + [artist_tuple[0] for artist_tuple in list(chain.from_iterable(states_pred.values()))]
            + list(extra.values())
            + list(traj_lines.values())
            + list(traj_points.values())
            + list(texts.values())
        )

    def plot_pred_states(name: PlayerName, trajs: list[Trajectory], state_artists: Optional[list[Artist]]=None, alpha: float = 0.1) -> list[Artist]:
        if state_artists is None:
            state_artists = []
            for traj in trajs:
                horizon = len(traj)
                for i in range(horizon):
                    time = traj.timestamps[i]
                    state = traj.at(time)
                    state_artist, _ = sim_viz.plot_player(
                        ax=ax,
                        state=state,
                        command=VehicleCommands(acc=0, ddelta=0),
                        lights_colors = None,
                        player_name=name,
                        alpha=alpha,
                        plot_text=False,
                    )
                    state_artists.append(state_artist)
        else:
            idx = 0
            for traj in trajs:
                horizon = len(traj)
                for i in range(horizon):
                    time = traj.timestamps[i]
                    state = traj.at(time)
                    state_artist, _ = sim_viz.plot_player(
                        ax=ax,
                        state=state,
                        command=VehicleCommands(acc=0, ddelta=0),
                        lights_colors = None,
                        model_poly=state_artists[idx],
                        player_name=name,
                        alpha=alpha,
                        plot_text=False,
                    )
                    state_artists[idx] = state_artist
                    idx += 1
                    
        return state_artists
        

    def init_plot() -> Iterable[Artist]:
        ax.clear()
        with sim_viz.plot_arena(ax=ax):
            init_log_entry: Mapping[PlayerName, LogEntry] = sim_context.log.at_interp(time_begin)
            for pname, plog in init_log_entry.items():
                lights_colors: LightsColors = get_lights_colors_from_cmds(init_log_entry[pname].commands, t=0)
                states[pname], actions[pname] = sim_viz.plot_player(
                    ax=ax,
                    state=plog.state,
                    command=plog.commands,
                    lights_colors=lights_colors,
                    player_name=pname,
                    alpha=0.8,
                    plot_wheels=plot_wheels,
                    plot_lights=plot_ligths,
                )
                if plog.extra:
                    try:
                        # trajectories, tcolors = unzip(plog.extra)
                        # traj_lines[pname], traj_points[pname] = sim_viz.plot_trajectories(
                        #         ax=ax, player_name=pname, trajectories=list(trajectories), colors=list(tcolors))
                        trajectories = plog.extra
                        for name, trajs in trajectories.items():
                            colors = ["gold"]*len(trajs)
                            traj_lines[name], traj_points[name] = sim_viz.plot_trajectories(
                                ax=ax, player_name=name, trajectories=trajs, colors=colors)
                            states_pred[name] = plot_pred_states(name, trajs)
                    except Exception as e:
                        logger.debug("Cannot plot extra", extra=type(plog.extra))
                        print("init extra failed because: ", e)
            adjust_axes_limits(
                ax=ax,
                plot_limits=plot_limits,
                players_states={p: log_entry.state for p, log_entry in init_log_entry.items()},
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

    def update_plot(frame: int = 0) -> Iterable[Artist]:
        t: float = frame * dt / 1000.0
        if frame >= 15:
            pass
        log_at_t: Mapping[PlayerName, LogEntry] = sim_context.log.at_interp(t)
        for pname, box_handle in states.items():
            lights_colors: LightsColors = get_lights_colors_from_cmds(log_at_t[pname].commands, t=t)
            states[pname], actions[pname] = sim_viz.plot_player(
                ax=ax,
                player_name=pname,
                state=log_at_t[pname].state,
                command=log_at_t[pname].commands,
                lights_colors=lights_colors,
                model_poly=box_handle,
                lights_patches=actions[pname],
                plot_wheels=plot_wheels,
                plot_lights=plot_ligths,
            )
            if log_at_t[pname].extra:
                try:
                    # trajectories, tcolors = unzip(log_at_t[pname].extra)
                    # traj_lines[pname], traj_points[pname] = sim_viz.plot_trajectories(
                    #     ax=ax,
                    #     player_name=pname,
                    #     trajectories=list(trajectories),
                    #     traj_lines=traj_lines[pname],
                    #     traj_points=traj_points[pname],
                    #     colors=list(tcolors),
                    # )
                    trajectories = log_at_t[pname].extra
                    for name, trajs in trajectories.items():
                        colors = ["gold"]*len(trajs)
                        if name not in traj_lines.keys():
                            traj_lines[name], traj_points[name] = sim_viz.plot_trajectories(
                                ax=ax, player_name=name, trajectories=trajs, colors=colors)
                            states_pred[name] = plot_pred_states(name, trajs)
                        else:
                            traj_lines[name], traj_points[name] = sim_viz.plot_trajectories(
                                ax=ax, player_name=name, trajectories=trajs, colors=colors, traj_lines=traj_lines[name], traj_points=traj_points[name])
                            states_pred[name] = plot_pred_states(name, trajs, states_pred[name])
                    prev_names = list(traj_lines.keys())
                    for prev_name in prev_names:
                        if prev_name not in trajectories.keys():
                            fake_trajs = [Trajectory(timestamps=[0], values=[VehicleState(x=0, y=0, psi=0, vx=0, delta=0)])]
                            traj_lines[prev_name], traj_points[prev_name] = sim_viz.plot_trajectories(
                                ax=ax, player_name=prev_name, trajectories=fake_trajs, traj_lines=traj_lines[prev_name], traj_points=traj_points[prev_name], alpha=0)
                            states_pred[prev_name] = plot_pred_states(prev_name, trajs, states_pred[prev_name], alpha=0)
                            # prev_traj_lines.remove()
                            # prev_traj_points = traj_points.pop(prev_name)
                            # prev_traj_points.remove()
                            # prev_states_pred = states_pred.pop(prev_name)
                            # for state in prev_states_pred:
                            #     state[0].remove()
                                # pass
                except Exception as e:
                    print("update extra failed because: ", e)

            if pname in sim_context.missions:
                goal_box = goals[pname] if pname in goals else None
                goals[pname] = sim_viz.plot_timevarying_goals(ax=ax, player_name=pname, goal_box=goal_box, t=t)

        adjust_axes_limits(
            ax=ax, plot_limits=plot_limits, players_states={p: log_entry.state for p, log_entry in log_at_t.items()}
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
    with tqdm(total=frame_count, unit="frame") as t:
        anim.save(
            file_path,
            dpi=dpi,
            writer="ffmpeg",
            fps=fps,
            progress_callback=lambda *_: t.update(1),
            # extra_args=["-g", "1", "-keyint_min", str(interval_seconds)]
        )
    ax.clear()


def adjust_axes_limits(
    ax: Axes,
    plot_limits: Union[str, PlayerName, Sequence[Sequence[float]]],
    players_states: Optional[Mapping[PlayerName, X]] = None,
):
    if plot_limits == "auto":
        if not players_states:
            raise ZValueError('Plotting with "auto" option requires players positions')
        players_limits = approximate_bounding_box_players(obj_list=list(players_states.values()))
        if players_limits is not None:
            ax.axis(
                xmin=players_limits[0][0] - 5.0,
                xmax=players_limits[0][1] + 5.0,
                ymin=players_limits[1][0] - 5.0,
                ymax=players_limits[1][1] + 5.0,
            )
        else:
            ax.autoscale()
    elif isinstance(plot_limits, str):
        # str instead of "PlayerName" since https://github.com/python/mypy/issues/3325
        try:
            state = players_states[plot_limits]
            slack = 35
            v_scaling = 1.5
            # we shift the center of the image forward according to the velocity vector of the player
            velocity_v = v_scaling * np.array([state.vx * np.cos(state.psi), state.vx * np.sin(state.psi)])
            buffer = slack / 3
            velocity_v = np.clip(velocity_v, -slack + buffer, slack - buffer)
            x_c, y_c = state.x + velocity_v[0], state.y + velocity_v[1]
            ax.axis(xmin=x_c - slack, xmax=x_c + slack, ymin=y_c - slack, ymax=y_c + slack)
        except AssertionError:
            ax.autoscale()

    elif isinstance(plot_limits, Sequence):
        # plotlimits are expected to be seq of seq of floats
        ax.set_xlim(plot_limits[0][0], plot_limits[0][1])
        ax.set_ylim(plot_limits[1][0], plot_limits[1][1])
    else:
        raise ZValueError(f"Plot limits {plot_limits} not recognized")
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
