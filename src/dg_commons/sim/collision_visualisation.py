from enum import IntEnum
from typing import Mapping

import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch

from dg_commons import PlayerName
from dg_commons.sim import CollisionReport, SimLog
from dg_commons.sim.collision_utils import velocity_of_P_given_A
from dg_commons.sim.simulator_structures import LogEntry


class _Zorders(IntEnum):
    PLAYER_NAME = 90
    VEL_BEFORE = 80
    VEL_AFTER = 81
    IMPACT_LOCATION = 40
    IMPACT_LOCATION_NAME = 40
    IMPACT_POINT = 82
    IMPACT_NORMAL = 85
    DEBUG = 100


def plot_collision(collision_report: CollisionReport, sim_log: SimLog):
    fig = plt.gca().get_figure()

    log_entries: Mapping[PlayerName, LogEntry] = sim_log.at_interp(collision_report.at_time)
    # common impact point and normals
    imp_point = collision_report.impact_point.coords[0]
    n = 0.2 * collision_report.impact_normal
    plt.plot(*imp_point, "o", zorder=_Zorders.IMPACT_POINT)
    n_color = "r"
    plt.arrow(imp_point[0], imp_point[1], n[0], n[1], ec=n_color, fc=n_color, alpha=0.9, zorder=_Zorders.IMPACT_NORMAL)
    # players
    name = "Dark2"
    cmap: ListedColormap = get_cmap(name)
    colors = list(cmap.colors)
    col_before, col_after = "darkorange", "seagreen"
    for i, (player, p_report) in enumerate(collision_report.players.items()):
        p_color = colors[i]
        # vehicle outline
        footprint = p_report.footprint
        plt.plot(*footprint.exterior.xy, color=p_color)
        xc, yc = log_entries[player].state.x, log_entries[player].state.y  # footprint.centroid.coords[0]
        plt.text(
            xc, yc, f"{player}", horizontalalignment="center", verticalalignment="center", zorder=_Zorders.PLAYER_NAME
        )
        # velocity vectors
        vel_scale = 0.3
        vel = vel_scale * p_report.velocity[0]
        vel_after = vel_scale * p_report.velocity_after[0]
        arr_width = 0.01
        head_width = arr_width * 5
        plt.arrow(
            xc,
            yc,
            vel[0],
            vel[1],
            width=arr_width,
            head_width=head_width,
            ec=col_before,
            fc=col_before,
            alpha=0.8,
            zorder=_Zorders.VEL_BEFORE,
        )
        plt.arrow(
            xc,
            yc,
            vel_after[0],
            vel_after[1],
            width=arr_width,
            head_width=head_width,
            ec=col_after,
            fc=col_after,
            alpha=0.8,
            zorder=_Zorders.VEL_AFTER,
        )
        # plot rotational velocity
        arrow_shift = 0.1
        arrow_patch = FancyArrowPatch(
            (xc - arrow_shift, yc),
            (xc + arrow_shift, yc),
            connectionstyle=f"arc3,rad={arrow_shift}",
            color="k",
            zorder=_Zorders.DEBUG,
        )
        fig.patches.extend([arrow_patch])
        plt.text(
            xc,
            yc + 4 * arrow_shift,
            f"{p_report.velocity[1]:.3f}",
            horizontalalignment="center",
            verticalalignment="center",
            zorder=_Zorders.VEL_BEFORE,
            color=col_before,
        )
        plt.text(
            xc,
            yc + 2 * arrow_shift,
            f"{p_report.velocity_after[1]:.3f}",
            horizontalalignment="center",
            verticalalignment="center",
            zorder=_Zorders.VEL_AFTER,
            color=col_after,
        )
        # Velocities at collision point
        ap = np.array(imp_point) - np.array([xc, yc])
        omega = p_report.velocity[1]
        vel_atP = vel_scale * velocity_of_P_given_A(1 / vel_scale * vel, omega, ap)
        plt.arrow(
            imp_point[0],
            imp_point[1],
            vel_atP[0],
            vel_atP[1],
            width=arr_width,
            head_width=head_width,
            ec=p_color,
            fc=p_color,
            alpha=0.8,
            zorder=_Zorders.DEBUG,
        )
        # impact locations
        for loc in p_report.locations:
            loc_str, loc_shape = loc
            plt.fill(*loc_shape.exterior.xy, fc="cyan", ec="darkblue", alpha=0.4, zorder=_Zorders.IMPACT_LOCATION)
            xc, yc = loc_shape.centroid.coords[0]
            plt.text(
                xc,
                yc,
                f"{loc_str}",
                horizontalalignment="center",
                verticalalignment="center",
                zorder=_Zorders.IMPACT_LOCATION_NAME,
            )

    before_patch = mpatches.Patch(color=col_before, label="before")
    after_patch = mpatches.Patch(color=col_after, label="after")
    plt.legend(handles=[before_patch, after_patch])

    fig.set_tight_layout(True)
    plt.axis("equal")
    plt.draw()
    return
