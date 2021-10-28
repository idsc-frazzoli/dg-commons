from dg_commons_dev.behavior.behavior_types import Situation, SituationParams
from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np
from dg_commons_dev.behavior.utils import SituationObservations
from dg_commons.sim.models import kmh2ms, extract_vel_from_state
from math import pi


# TODO:  there is still a lot to do here

@dataclass
class YieldDescription:
    """ Important parameters describing an emergency """

    is_yield: bool

    drac: float = 0

    def __post_init__(self):
        if self.is_yield:
            assert self.drac is not None


@dataclass
class YieldParams(SituationParams):
    min_vel: Union[List[float], float] = kmh2ms(5)
    """Emergency only to vehicles that are at least moving at.."""
    min_dist: Union[List[float], float] = 7
    """Evaluate whether to yield only for vehicles within x [m]"""


class Yield(Situation[SituationObservations, YieldDescription]):
    """ Yield situation """
    def __init__(self, params: YieldParams, safety_time_braking):
        self.params = params
        self.safety_time_braking = safety_time_braking
        self.obs: Optional[SituationObservations] = None
        self.yield_situation: Optional[YieldDescription] = None

    def update_observations(self, new_obs: SituationObservations):
        self.obs = new_obs
        my_name = new_obs.my_name
        agents = new_obs.agents
        agents_rel_pose = new_obs.rel_poses

        for other_name, _ in agents.items():
            if other_name == my_name:
                continue
            rel = agents_rel_pose[other_name]
            other_vel = extract_vel_from_state(agents[other_name].state)
            rel_distance = np.linalg.norm(rel.p)
            coming_from_the_right: bool = pi / 4 <= rel.theta <= pi * 3 / 4 and \
                                          other_vel > self.params.min_vel
            if coming_from_the_right and rel_distance < self.params.min_dist:
                self.yield_situation = YieldDescription(True)
        self.yield_situation = YieldDescription(False)

    def is_true(self) -> bool:
        assert self.obs is not None
        return self.yield_situation.is_yield

    def infos(self) -> YieldDescription:
        assert self.obs is not None
        return self.yield_situation
