from itertools import chain
from typing import FrozenSet

from dg_commons import PlayerName
from dg_commons.sim.simulator import SimContext, Simulator

__all__ = ["run_simulation", "who_collided"]


def run_simulation(sim_context: SimContext) -> SimContext:
    sim = Simulator()
    sim.run(sim_context)
    return sim_context


def who_collided(sim_context: SimContext) -> FrozenSet[PlayerName]:
    """Return the list of players that collided with the AV"""
    return frozenset(chain.from_iterable(map(lambda report: report.players.keys(), sim_context.collision_reports)))
