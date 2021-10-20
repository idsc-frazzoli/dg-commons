from dataclasses import dataclass
from math import pi
from typing import Optional, Tuple, MutableMapping, Dict

import numpy as np
from geometry import SE2value

from dg_commons import PlayerName, relative_pose, SE2Transform, valmap
from dg_commons.controllers.pid import PIDParam, PID
from dg_commons.sim.models import extract_pose_from_state, kmh2ms, extract_vel_from_state
from dg_commons.sim.simulator_structures import PlayerObservations

__all__ = ["SpeedControllerParam", "SpeedController", "SpeedBehavior"]


@dataclass
class SpeedControllerParam(PIDParam):
    """Default values are tuned roughly for a default car model"""

    kP: float = 4
    kI: float = 0.01
    kD: float = 0.1
    antiwindup: Tuple[float, float] = (-2, 2)
    setpoint_minmax: Tuple[float, float] = (-kmh2ms(10), kmh2ms(150))
    output_minmax: Tuple[float, float] = (-8, 5)  # acc minmax


class SpeedController(PID):
    """Low-level controller for reference tracking of speed"""

    def __init__(self, params: Optional[PIDParam] = None):
        params = SpeedControllerParam() if params is None else params
        super(SpeedController, self).__init__(params)


@dataclass
class SpeedBehaviorParam:
    nominal_speed: float = kmh2ms(40)
    """Nominal desired speed"""
    yield_distance: float = 7
    """Evaluate whether to yield only for vehicles within x [m]"""
    minimum_yield_vel: float = kmh2ms(5)
    """yield only to vehicles that are at least moving at.."""
    safety_time_braking: float = 1.5
    """Evaluates safety distance from vehicle in front based on distance covered in this delta time"""


class SpeedBehavior:
    """Determines the reference speed"""

    def __init__(self, my_name: Optional[PlayerName] = None):
        self.params: SpeedBehaviorParam = SpeedBehaviorParam()
        self.my_name: PlayerName = my_name
        self.agents: Optional[MutableMapping[PlayerName, PlayerObservations]] = None
        self.speed_ref: float = 0
        """ The speed reference"""

    def update_observations(self, agents: MutableMapping[PlayerName, PlayerObservations]):
        self.agents = agents

    def get_speed_ref(self, at: float) -> (float, bool):
        """Check if there is anyone on the right too close, then brake.
        @:return candidate speed ref and whether or not the situation requires an emergency takeover
        (e.g. collision avoidance)
        """

        mypose = extract_pose_from_state(self.agents[self.my_name].state)

        def rel_pose(other_obs: PlayerObservations) -> SE2Transform:
            other_pose: SE2value = extract_pose_from_state(other_obs.state)
            return SE2Transform.from_SE2(relative_pose(mypose, other_pose))

        agents_rel_pose: Dict[PlayerName, SE2Transform] = valmap(rel_pose, self.agents)
        yield_to_anyone: bool = self.is_there_anyone_to_yield_to(agents_rel_pose)
        emergency_situation: bool = self.is_emergency_subroutine_needed(agents_rel_pose)
        if yield_to_anyone or emergency_situation:
            self.speed_ref = 0
        else:
            self.speed_ref = self.cruise_control(agents_rel_pose)
        return self.speed_ref, emergency_situation

    def is_there_anyone_to_yield_to(self, agents_rel_pose: Dict[PlayerName, SE2Transform]) -> bool:
        """
        If someone is approaching from the right or someone is in front of us we yield
        """

        for other_name, _ in self.agents.items():
            if other_name == self.my_name:
                continue
            rel = agents_rel_pose[other_name]
            other_vel = extract_vel_from_state(self.agents[other_name].state)
            rel_distance = np.linalg.norm(rel.p)
            # todo improve with SPOT predictions
            coming_from_the_right: bool = (
                pi / 4 <= rel.theta <= pi * 3 / 4 and other_vel > self.params.minimum_yield_vel
            )
            if coming_from_the_right and rel_distance < self.params.yield_distance:
                return True
        return False

    def is_emergency_subroutine_needed(self, agents_rel_pose: Dict[PlayerName, SE2Transform]) -> bool:
        myvel = self.agents[self.my_name].state.vx
        for other_name, _ in self.agents.items():
            if other_name == self.my_name:
                continue
            rel = agents_rel_pose[other_name]
            other_vel = extract_vel_from_state(self.agents[other_name].state)
            rel_distance = np.linalg.norm(rel.p)
            coming_from_the_left: bool = (
                -3 * pi / 4 <= rel.theta <= -pi / 4 and other_vel > self.params.minimum_yield_vel
            )
            in_front_of_me: bool = rel.p[0] > 0 and -1.2 <= rel.p[1] <= 1.2
            coming_from_the_front: bool = 3 * pi / 4 <= abs(rel.theta) <= pi * 5 / 4 and in_front_of_me
            if (coming_from_the_left and rel_distance < self.params.yield_distance) or (
                coming_from_the_front and rel_distance < self.params.safety_time_braking * (myvel + other_vel)
            ):
                return True
        return False

    def cruise_control(self, agents_rel_pose: Dict[PlayerName, SE2Transform]) -> float:
        """
        If someone is in front with the same orientation, then apply the two seconds rule to adapt reference velocity
         that allows maintaining a safe distance between the vehicles
        """
        myvel = self.agents[self.my_name].state.vx
        candidate_speed_ref = [
            self.params.nominal_speed,
        ]
        for other_name, _ in self.agents.items():
            if other_name == self.my_name:
                continue
            rel = agents_rel_pose[other_name]
            rel_dist = np.linalg.norm(rel.p)
            other_vel = self.agents[other_name].state.vx
            in_front_of_me: bool = rel.p[0] > 0.5 and abs(rel.p[1]) <= 1.2 and abs(rel.theta) < pi / 6
            # safety distance at current speed + difference of how it will be in the next second
            dist_to_keep = self._get_min_safety_dist(myvel) + max(myvel - other_vel, 0)
            if in_front_of_me and rel_dist < dist_to_keep:
                speed_ref = float(np.clip(other_vel - max(myvel - other_vel, 0), 0, kmh2ms(130)))
                candidate_speed_ref.append(speed_ref)
            else:
                candidate_speed_ref.append(self.params.nominal_speed)
        return min(candidate_speed_ref)

    def _get_min_safety_dist(self, vel: float):
        """The distance covered in x [s] travelling at vel"""
        return vel * self.params.safety_time_braking
