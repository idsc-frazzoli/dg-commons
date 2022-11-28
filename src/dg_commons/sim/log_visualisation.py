from dataclasses import fields
from typing import Set

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from dg_commons import PlayerName
from dg_commons.sim import SimLog, PlayerLog

__all__ = ["plot_players_logs", "plot_player_log"]


def plot_players_logs(sim_log: SimLog, fig: Figure = None, only_players: Set[PlayerName] = None):
    fig = plt.gca().get_figure() if fig is None else fig

    for pn, log in sim_log.items():
        if only_players is not None and pn not in only_players:
            continue

        plot_player_log(log, fig)


def plot_player_log(log: PlayerLog, fig: Figure = None):
    fig = plt.gca().get_figure() if fig is None else fig

    state_fields = fields(log.states.values[0])
    n_states = len(state_fields)
    cmds_fields = fields(log.commands.values[0])
    n_inputs = len(cmds_fields)
    rows_number = n_states + n_inputs

    for i, field in enumerate(state_fields):
        field_value = [getattr(s, field.name) for s in log.states.values]
        ts = log.states.timestamps
        fig.add_subplot(rows_number, 1, i + 1)
        plt.plot(ts, field_value, label=field.name)
        plt.grid()
        plt.legend()

    for i, field in enumerate(cmds_fields):
        field_value = [getattr(c, field.name) for c in log.commands.values]
        ts = log.commands.timestamps
        fig.add_subplot(rows_number, 1, n_states + i + 1)
        plt.step(ts, field_value, label=field.name)
        plt.grid()
        plt.legend()

    plt.draw()
