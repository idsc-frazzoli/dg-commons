from dg_commons_dev.behavior.behavior_types import Situation, SituationParams
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, MutableMapping
from dg_commons_dev.behavior.utils import SituationObservations, \
    occupancy_prediction, entry_exit_t, SituationPolygons, Polygon, PlayerObservations
from dg_commons.sim.models import kmh2ms, extract_vel_from_state
from dg_commons.sim.models.vehicle import VehicleParameters
from dg_commons import PlayerName, X


@dataclass
class EmergencyDescription:
    """ Important parameters describing an emergency """

    is_emergency: bool = False

    drac: Optional[Tuple[float, float]] = None
    """ deceleration to avoid a crash """
    ttc: Optional[float] = None
    """ time-to-collision """
    pet: Optional[float] = None
    """ post encroachment time """

    """ Reference: https://sumo.dlr.de/docs/Simulation/Output/SSM_Device.html """

    my_player: PlayerName = None
    """ My PlayerName """
    other_player: PlayerName = None
    """ Other Playername """

    def __post_init__(self):
        if self.is_emergency:
            assert self.ttc is not None
            assert self.drac is not None
            assert self.pet is not None


@dataclass
class EmergencyParams(SituationParams):
    min_dist: Union[List[float], float] = 7
    """Evaluate emergency only for vehicles within x [m]"""
    min_vel: Union[List[float], float] = kmh2ms(5)
    """emergency only to vehicles that are at least moving at.."""


class Emergency(Situation[SituationObservations, EmergencyDescription]):
    """
    Emergency situation, provides tools for:
     1) establishing whether an emergency is occurring
     2) computing important parameters describing the emergency situation
    """

    def __init__(self, params: EmergencyParams, safety_time_braking: float,
                 vehicle_params: VehicleParameters = VehicleParameters.default_car(),
                 plot: bool = False):
        self.params = params
        self.safety_time_braking = safety_time_braking
        self.acc_limits: Tuple[float, float] = vehicle_params.acc_limits

        self.obs: SituationObservations
        self.emergency_situation: EmergencyDescription = EmergencyDescription()
        self.polygon_plotter = SituationPolygons(plot=plot)
        self.counter = 0

    def update_observations(self, new_obs: SituationObservations) \
            -> Tuple[List[Polygon], List[SituationPolygons.PolygonClass]]:
        self.obs = new_obs
        my_name: PlayerName = new_obs.my_name
        agents: MutableMapping[PlayerName, PlayerObservations] = new_obs.agents

        my_state: X = agents[my_name].state
        my_vel: float = my_state.vx
        my_occupancy: Polygon = agents[my_name].occupancy
        my_polygon, _ = occupancy_prediction(agents[my_name].state, self.safety_time_braking)

        for other_name, _ in agents.items():
            if other_name == my_name:
                continue
            other_state: X = agents[other_name].state
            other_vel: float = extract_vel_from_state(other_state)
            other_occupancy: Polygon = agents[other_name].occupancy
            other_polygon, _ = occupancy_prediction(agents[other_name].state, self.safety_time_braking)

            intersection: Polygon = my_polygon.intersection(other_polygon)
            if intersection.is_empty:
                self.emergency_situation = EmergencyDescription(False)
            else:
                my_entry_time, my_exit_time = entry_exit_t(intersection, my_state, my_occupancy,
                                                           self.safety_time_braking, my_vel, tol=0.01)
                other_entry_time, other_exit_time = entry_exit_t(intersection, other_state, other_occupancy,
                                                                 self.safety_time_braking, other_vel, tol=0.01)

                collision_score: float = 0
                collision_max: float = 1  # if collision_score > collision_max then there is an emergency

                def pet_score(pet: float):
                    pet_min = self.safety_time_braking
                    if pet < pet_min:
                        return 1.0
                    else:
                        return 0.0

                def ttc_score(ttc: float):
                    ttc_min = self.safety_time_braking
                    if ttc < ttc_min:
                        return 1.0
                    else:
                        return 0.0

                def drac_score(drac: float):
                    drac_max = self.acc_limits[1]
                    if drac_max < drac:
                        return 1.0
                    else:
                        return 0.0

                pet: float = other_entry_time - my_exit_time if my_exit_time < other_exit_time else \
                    my_entry_time - other_exit_time
                self.emergency_situation.my_player = my_name
                self.emergency_situation.pet = pet
                collision_score += pet_score(pet)

                pot1, pot2 = my_exit_time - other_entry_time > 0, other_exit_time - my_entry_time > 0
                if pot1 or pot2:
                    ttc = other_entry_time if my_entry_time < other_entry_time else my_entry_time
                    self.emergency_situation.ttc = ttc
                    collision_score += ttc_score(ttc)
                    drac1: float = 2 * (other_vel - other_vel * other_entry_time / my_exit_time) / my_exit_time \
                        if pot1 else 0.0
                    drac2: float = 2 * (my_vel - my_vel * my_entry_time / other_exit_time) / other_exit_time \
                        if pot2 else 0.0
                    drac: float = max(drac1, drac2)
                    collision_score += drac_score(drac)
                    self.emergency_situation.drac = [drac1, drac2]
                    if collision_max < collision_score:
                        self.emergency_situation.is_emergency = True
                        self.emergency_situation.other_player = other_name

                        other_occupancy, _ = occupancy_prediction(agents[other_name].state, 0.1)
                        my_occupancy, _ = occupancy_prediction(agents[my_name].state, 0.1)
                        self.polygon_plotter.plot_polygon(my_occupancy,
                                                          SituationPolygons.PolygonClass(collision=True))
                        self.polygon_plotter.plot_polygon(other_occupancy,
                                                          SituationPolygons.PolygonClass(collision=True))

        return self.polygon_plotter.next_frame()
        # This is for plotting purposes, can be ignored

    def _get_min_safety_dist(self, vel: float) -> float:
        """The distance covered in x [s] travelling at vel"""
        return vel * self.safety_time_braking

    def is_true(self) -> bool:
        assert self.obs is not None
        return self.emergency_situation.is_emergency

    def infos(self) -> EmergencyDescription:
        assert self.obs is not None
        return self.emergency_situation

    def simulation_ended(self):
        """ Called when the simulation ends """
        pass
