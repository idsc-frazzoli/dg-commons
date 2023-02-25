from decimal import Decimal
from typing import NewType, Sequence

from dg_commons import DgSampledSequence, X, Color

__all__ = ["SimTime", "ImpactLocation", "DrawableTrajectoryType"]

SimTime = Decimal
"""The time of the simulation time"""
ImpactLocation = NewType("ImpactLocation", str)
"""Imapact location in human readable format"""
DrawableTrajectoryType = Sequence[tuple[DgSampledSequence[X], Color]]
"""The interface for the supported visualisation of trajectories.
Each trajectory shall come paired with a color, setting to None will pick the agent's color """
