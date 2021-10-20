from dataclasses import dataclass, replace
from typing import Mapping, Tuple, Dict, List

import numpy as np
from geometry import T2value
from shapely.geometry import Point, Polygon
from zuper_commons.types import ZValueError

from dg_commons import PlayerName
from dg_commons.sim import SimTime, ImpactLocation

__all__ = [
    "IMPACT_EVERYWHERE",
    "IMPACT_FRONT",
    "IMPACT_BACK",
    "IMPACT_LEFT",
    "IMPACT_RIGHT",
    "CollisionReportPlayer",
    "CollisionReport",
    "combine_collision_reports",
]

IMPACT_EVERYWHERE = ImpactLocation("everywhere")
IMPACT_FRONT = ImpactLocation("front")
IMPACT_BACK = ImpactLocation("back")
IMPACT_LEFT = ImpactLocation("left")
IMPACT_RIGHT = ImpactLocation("right")


@dataclass(frozen=True, unsafe_hash=True)
class CollisionReportPlayer:
    locations: List[Tuple[ImpactLocation, Polygon]]
    """ Location of the impact """
    at_fault: bool
    """ At fault is defined as...."""
    footprint: Polygon
    """ Footprint of impact"""
    velocity: Tuple[T2value, float]
    """ velocity before impact [m/s],[rad/s] """
    velocity_after: Tuple[T2value, float]
    """ velocity after impact [m/s],[rad/s] """
    energy_delta: float
    """ Kinetic energy lost in the collision [J] """


@dataclass(frozen=True, unsafe_hash=True)
class CollisionReport:
    players: Mapping[PlayerName, CollisionReportPlayer]
    """ Relative velocity defined as v_a-v_b in global RF [m/s] """
    impact_point: Point
    """Point of impact"""
    impact_normal: np.ndarray
    """Normal of impact"""
    at_time: SimTime
    """ Sim time at which the collision occurred"""


def combine_collision_reports(r1: CollisionReport, r2: CollisionReport) -> CollisionReport:
    """This function "sums" collision reports.
    While the simulation generates a collision report at every simulation step, oftentimes it's convenient to
    reduce them to an "accident" report.
    """
    if r1.players.keys() != r2.players.keys():
        raise ZValueError("Cannot combine collision reports with different players", report1=r1, report2=r2)
    # impact point, normal are propagated according to the first report
    first_report, second_report = (r1, r2) if r1.at_time <= r2.at_time else (r2, r1)
    combined_players_report: Dict[PlayerName, CollisionReportPlayer] = {}
    for p in first_report.players:
        r1_player, r2_player = first_report.players[p], second_report.players[p]
        combined_players_report[p] = replace(
            r1_player,
            locations=r1_player.locations + r2_player.locations,
            velocity_after=r2_player.velocity_after,
            energy_delta=r1_player.energy_delta + r2_player.energy_delta,
        )
    return replace(first_report, players=combined_players_report)
