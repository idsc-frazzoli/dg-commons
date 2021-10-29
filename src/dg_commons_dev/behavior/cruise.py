from dg_commons_dev.behavior.behavior_types import Situation, SituationParams
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, MutableMapping
from dg_commons.sim.models import kmh2ms, extract_vel_from_state
from dg_commons_dev.behavior.utils import SituationObservations, \
    occupancy_prediction, SituationPolygons, Polygon, PlayerObservations
from dg_commons import PlayerName, X


@dataclass
class CruiseDescription:
    """ Important parameters describing a cruise situation """
    is_cruise: bool = True
    is_following: bool = False

    speed_ref: Optional[float] = None

    my_player: Optional[PlayerName] = None
    other_player: Optional[PlayerName] = None

    def __post_init__(self):
        if self.is_cruise:
            assert self.speed_ref is not None
            assert self.my_player is not None
        if self.is_following:
            assert self.other_player is not None


@dataclass
class CruiseParams(SituationParams):
    nominal_speed: Union[List[float], float] = kmh2ms(40)
    """Nominal desired speed"""

    min_safety_distance: float = 5.0
    """ Min distance to keep from vehicle ahead at v = 0 """

    kp_back: float = 2.0
    kp_forward: float = 1.0
    """ kp for computing speed ref when 
    1) I am too close to ref vehicle 
    2) I am not closer than min distance and the ref vehicle is moving slower than nominal_speed """


class Cruise(Situation[SituationObservations, CruiseDescription]):
    """ Cruise situation """

    def __init__(self, params: CruiseParams, safety_time_braking, plot=False):
        self.params = params
        self.safety_time_braking = safety_time_braking

        self.obs: Optional[SituationObservations] = None
        self.cruise_situation: Optional[CruiseDescription] = None
        self.polygon_plotter = SituationPolygons(plot=plot)

    def update_observations(self, new_obs: SituationObservations)\
            -> Tuple[List[Polygon], List[SituationPolygons.PolygonClass]]:
        self.obs = new_obs
        my_name: PlayerName = new_obs.my_name
        agents: MutableMapping[PlayerName, PlayerObservations] = new_obs.agents

        my_state: X = agents[my_name].state
        my_vel: float = my_state.vx
        my_occupancy: Polygon = agents[my_name].occupancy
        my_polygon, _ = occupancy_prediction(agents[my_name].state, self._get_look_ahead_time(my_vel))
        self.polygon_plotter.plot_polygon(my_polygon, SituationPolygons.PolygonClass(dangerous_zone=True))

        self.cruise_situation = CruiseDescription(is_cruise=True, is_following=False,
                                                  speed_ref=self.params.nominal_speed, my_player=my_name)
        for other_name, _ in agents.items():
            if other_name == my_name:
                continue
            other_state: X = agents[other_name].state
            other_vel: float = extract_vel_from_state(other_state)
            other_occupancy: Polygon = agents[other_name].occupancy

            intersection: Polygon = my_polygon.intersection(other_occupancy)

            if not intersection.is_empty:
                distance: float = my_occupancy.distance(other_occupancy)
                min_distance: float = self._get_min_safety_dist(my_vel)
                if distance < min_distance:
                    speed_ref: float = \
                        other_vel + (distance - min_distance) / min_distance * other_vel * self.params.kp_back
                elif distance > min_distance and other_vel < self.params.nominal_speed:
                    speed_ref: float = \
                        other_vel + (distance - min_distance) / min_distance * other_vel * self.params.kp_forward
                else:
                    speed_ref: float = self.params.nominal_speed
                self.cruise_situation = CruiseDescription(True, is_following=True, speed_ref=speed_ref,
                                                          my_player=my_name, other_player=other_name)

                other_occupancy, _ = occupancy_prediction(agents[other_name].state, 0.1)
                my_occupancy, _ = occupancy_prediction(agents[my_name].state, 0.1)
                self.polygon_plotter.plot_polygon(my_occupancy, SituationPolygons.PolygonClass(following=True))
                self.polygon_plotter.plot_polygon(other_occupancy, SituationPolygons.PolygonClass(following=True))
                # This is only for plotting purposes

        return self.polygon_plotter.next_frame()

    def _get_min_safety_dist(self, vel: float) -> float:
        """The distance covered in x [s] travelling at vel"""
        return vel * self.safety_time_braking * 2 + self.params.min_safety_distance

    def _get_look_ahead_time(self, vel: float) -> float:
        if vel == 0:
            return self.safety_time_braking * 2
        else:
            return (self._get_min_safety_dist(vel) + 5) / vel

    def is_true(self) -> bool:
        assert self.obs is not None
        return self.cruise_situation.is_cruise

    def infos(self) -> CruiseDescription:
        assert self.obs is not None
        return self.cruise_situation

    def simulation_ended(self):
        pass
