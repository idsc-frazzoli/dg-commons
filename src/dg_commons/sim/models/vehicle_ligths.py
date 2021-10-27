from dataclasses import dataclass
from typing import NewType, AbstractSet, Mapping, Tuple

import numpy as np

from dg_commons import Color, fd

__all__ = [
    "LightsCmd",
    "NO_LIGHTS",
    "LIGHTS_HEADLIGHTS",
    "LIGHTS_TURN_LEFT",
    "LIGHTS_TURN_RIGHT",
    "LIGHTS_HAZARD",
    "LightsValues",
    "LightsColors",
    "lightscmd2phases",
    "get_phased_lights",
    "red",
    "red_more",
]

LightsCmd = NewType("LightsCmd", str)
""" The type of light commands. """
NO_LIGHTS = LightsCmd("none")
""" Lights are off. """
LIGHTS_HEADLIGHTS = LightsCmd("headlights")
""" The front lights are on. """
LIGHTS_TURN_LEFT = LightsCmd("turn_left")
""" Blinkers turn left """
LIGHTS_TURN_RIGHT = LightsCmd("turn_right")
""" Blinkers turn right """
LIGHTS_HAZARD = LightsCmd("hazard")
"""Hazard lights"""
LightsValues: AbstractSet[LightsCmd] = frozenset(
    {NO_LIGHTS, LIGHTS_HEADLIGHTS, LIGHTS_TURN_LEFT, LIGHTS_TURN_RIGHT, LIGHTS_HAZARD}
)
""" All possible lights command value. Only one of these commands can be given at each instant"""


@dataclass
class LightsColors:
    front_left: Color
    front_right: Color
    back_left: Color
    back_right: Color


white = (0.9, 0.9, 0.9)
yellow = (1.0, 1.0, 0.0)
red = (0.5, 0.0, 0.0)
red_more = (1.0, 0.0, 0.0)
orange = (1.0, 0.5, 0.3)

phase2colors: Mapping[LightsCmd, LightsColors] = fd(
    {
        NO_LIGHTS: LightsColors(back_left=red, back_right=red, front_left=white, front_right=white),
        LIGHTS_HEADLIGHTS: LightsColors(back_left=red, back_right=red, front_left=yellow, front_right=yellow),
        LIGHTS_TURN_LEFT: LightsColors(
            back_left=orange,
            back_right=red,
            front_left=orange,
            front_right=white,
        ),
        LIGHTS_TURN_RIGHT: LightsColors(
            back_left=red,
            back_right=orange,
            front_left=white,
            front_right=orange,
        ),
        LIGHTS_HAZARD: LightsColors(
            back_left=orange,
            back_right=orange,
            front_left=orange,
            front_right=orange,
        ),
    }
)

lightscmd2phases: Mapping[LightsCmd, Tuple[LightsCmd, ...]] = fd(
    {
        NO_LIGHTS: (NO_LIGHTS,),
        LIGHTS_HEADLIGHTS: (LIGHTS_HEADLIGHTS,),
        LIGHTS_TURN_LEFT: (LIGHTS_TURN_LEFT, NO_LIGHTS),
        LIGHTS_TURN_RIGHT: (LIGHTS_TURN_RIGHT, NO_LIGHTS),
        LIGHTS_HAZARD: (LIGHTS_HAZARD, NO_LIGHTS),
    }
)

PHASE_SLOW: float = 0.4


def get_phased_lights(phases, t: float, phase_period: float = PHASE_SLOW) -> LightsColors:
    phase_index = int(np.round(t / phase_period))
    phase = phases[phase_index % len(phases)]
    led_commands = phase2colors[phase]
    return led_commands
