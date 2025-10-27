from dataclasses import dataclass
from typing import Optional, Type

from dg_commons import PlayerName
from dg_commons.maps.lanes import DgLanelet
from dg_commons.sim import DrawableTrajectoryType, SimObservations
from dg_commons.sim.agents import TAgent

__all__ = ["AV", "OnGetExtraAV"]

AV = PlayerName("AV")


@dataclass(unsafe_hash=True, frozen=True)
class OnGetExtraAV:
    my_type: Type[TAgent]
    sim_obs: Optional[SimObservations] = None
    plan: Optional[DrawableTrajectoryType] = None
    ref_lane: Optional[DgLanelet] = None
