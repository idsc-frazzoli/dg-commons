from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping, Optional, List, MutableMapping

from geometry import translation_angle_from_SE2
import numpy as np

from dg_commons import PlayerName, DgSampledSequence, X
from dg_commons.sim import SimTime, CollisionReport, logger, CollisionReportPlayer
from dg_commons.sim.collision import impact_locations_from_polygons
from dg_commons.sim.collision_utils import CollisionException, compute_impact_geometry, velocity_of_P_given_A, \
    check_who_is_at_fault, kinetic_energy, rot_velocity_after_collision, velocity_after_collision, \
    compute_impulse_response
from dg_commons.sim.models.obstacles_dyn import DynObstacleModel
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator_structures import *
from dg_commons.time import time_function


@dataclass
class LightSimContext:
    """
    The simulation context that does not keep track of everything,
    handle with care as it is passed around by reference, it is a mutable object.
    """

    dg_scenario: DgScenario
    """A driving games scenario"""
    models: MutableMapping[PlayerName, SimModel]
    """The simulation models for each player"""
    param: SimParameters
    """The simulation parameters"""
    traj: Mapping[PlayerName, DgSampledSequence[X]] = field(default_factory=dict)
    """The propagated trajectory of each player"""
    sim_terminated: bool = False
    "Whether the simulation has terminated"
    time: SimTime = SimTime(0)
    "The clock for the simulator, keeps track of the current instant"
    collision_reports: List[CollisionReport] = field(default_factory=list)
    "The log of collision reports"
    first_collision_ts: SimTime = SimTime("Infinity")
    "The first collision time"

    def __post_init__(self):
        assert self.models.keys() == self.traj.keys()
        for pname in self.models.keys():
            assert issubclass(type(self.models[pname]), SimModel)


class LightSimulator:
    """
    A light simulator has a loop made of 2 main step:
        _ A update function that load the new states of all the models
        - A post-update function that checks the new states of all the models and resolves collisions
    """

    @time_function
    def run(self, light_sim_context: LightSimContext):

        while not light_sim_context.sim_terminated:
            self.update(light_sim_context)
            self.post_update(light_sim_context)

    @staticmethod
    def update(light_sim_context: LightSimContext):
        """Avoid the real step of the simulation"""
        t = light_sim_context.time
        for player_name, model in light_sim_context.models.items():
            new_state: X = light_sim_context.traj[player_name].at_or_previous(t)
            model.set_state(new_state)
        return

    def post_update(self, light_sim_context: LightSimContext):
        """
        Here all the operations that happen after we have stepped the simulation, e.g. collision checking
        """
        # after all the computations advance simulation time
        light_sim_context.time += light_sim_context.param.dt
        # collision checking
        collision_environment = self._check_collisions_with_environment(light_sim_context)
        collision_players = self._check_collisions_among_players(light_sim_context)
        # check if the simulation is over
        self._maybe_terminate_simulation(light_sim_context)
        return

    @staticmethod
    def _maybe_terminate_simulation(light_sim_context: LightSimContext):
        """Evaluates if the simulation needs to terminate based on the expiration of times.
        The simulation is considered terminated if:
        - the maximum time has expired
        - the minimum time after the first collision has expired
        """
        termination_condition: bool = (
                light_sim_context.time > light_sim_context.param.max_sim_time
                or light_sim_context.time > light_sim_context.first_collision_ts + light_sim_context.param.sim_time_after_collision
        )
        light_sim_context.sim_terminated = termination_condition

    @staticmethod
    def _check_collisions_with_environment(light_sim_context: LightSimContext) -> bool:
        """Check collisions of the players with the environment"""
        from dg_commons.sim.collision import resolve_collision_with_environment  # import here to avoid circular imports

        env_obstacles = light_sim_context.dg_scenario.strtree_obstacles
        collision = False
        for p, p_model in light_sim_context.models.items():
            p_shape = p_model.get_footprint()
            items = env_obstacles.query_items(p_shape)
            for idx in items:
                candidate = light_sim_context.dg_scenario.static_obstacles[idx]
                if p_shape.intersects(candidate.shape):
                    try:
                        report: Optional[CollisionReport] = resolve_collision_with_environment(
                            p, p_model, candidate, light_sim_context.time
                        )
                    except CollisionException as e:
                        logger.warn(f"Failed to resolve collision of {p} with environment because:\n{e.args}")
                        report = None
                    if report is not None and not isinstance(p_model, DynObstacleModel):
                        logger.info(f"Player {p} collided with the environment")
                        collision = True
                        light_sim_context.collision_reports.append(report)
                        if light_sim_context.time < light_sim_context.first_collision_ts:
                            light_sim_context.first_collision_ts = light_sim_context.time
        return collision

    @staticmethod
    def _check_collisions_among_players(light_sim_context: LightSimContext) -> bool:
        """
        This checks only collision location at the current step, tunneling effects and similar are ignored
        :param light_sim_context:
        :return: True if at least one collision happened, False otherwise
        """
        from dg_commons.sim.collision import resolve_collision  # import here to avoid circular imports

        collision = False
        for p1, p2 in combinations(light_sim_context.models, 2):
            a_shape = light_sim_context.models[p1].get_footprint()
            b_shape = light_sim_context.models[p2].get_footprint()
            if a_shape.intersects(b_shape):
                try:
                    report: Optional[CollisionReport] = light_resolve_collision(p1, p2, light_sim_context)
                except CollisionException as e:
                    logger.warn(f"Failed to resolve collision between {p1} and {p2} because:\n{e.args}")
                    report = None
                if report is not None:
                    logger.info(f"Detected a collision between {p1} and {p2}")
                    collision = True
                    if report.at_time < light_sim_context.first_collision_ts:
                        light_sim_context.first_collision_ts = report.at_time
                    light_sim_context.collision_reports.append(report)
        return collision


def light_resolve_collision(a: PlayerName, b: PlayerName, light_sim_context: LightSimContext) -> Optional[
    CollisionReport]:
    """
    Resolves the collision between A and B using the impulse method.
    Sources:
        - http://www.chrishecker.com/images/e/e7/Gdmphys3.pdf
        - https://research.ncl.ac.uk/game/mastersdegree/gametechnologies/previousinformation/physics6collisionresponse/
    :returns A CollisionReport or None if the collision does not need to be solved (the two bodies are already separating)
    """
    a_model: SimModel = light_sim_context.models[a]
    b_model: SimModel = light_sim_context.models[b]
    a_shape = a_model.get_footprint()
    b_shape = b_model.get_footprint()
    # Compute collision geometry
    impact_normal, impact_point = compute_impact_geometry(a_shape, b_shape)
    a_cog = translation_angle_from_SE2(a_model.get_pose())[0]
    b_cog = translation_angle_from_SE2(b_model.get_pose())[0]
    r_ap = np.array(impact_point.coords[0]) - np.array(a_cog)
    r_bp = np.array(impact_point.coords[0]) - np.array(b_cog)  # b_shape.centroid.coords[0])
    a_vel, a_omega = a_model.get_velocity(in_model_frame=False)
    b_vel, b_omega = b_model.get_velocity(in_model_frame=False)
    a_vel_atP = velocity_of_P_given_A(a_vel, a_omega, r_ap)
    b_vel_atP = velocity_of_P_given_A(b_vel, b_omega, r_bp)
    rel_velocity_atP = a_vel_atP - b_vel_atP

    if np.dot(rel_velocity_atP, impact_normal) < 0:
        logger.debug(f"Not solving the collision between {a}, {b} since they are already separating")
        return None

    # Update state of the model, it has collided
    a_model.has_collided = True
    b_model.has_collided = True

    # Compute collision locations
    a_locations = impact_locations_from_polygons(a_model, b_shape)
    b_locations = impact_locations_from_polygons(b_model, a_shape)

    # Check who is at fault
    if light_sim_context.dg_scenario.scenario:
        who_is_at_fault = check_who_is_at_fault(
            {a: a_model.get_pose(), b: b_model.get_pose()},
            impact_point=impact_point,
            lanelet_network=light_sim_context.dg_scenario.scenario.lanelet_network,
        )
        a_fault, b_fault = who_is_at_fault[a], who_is_at_fault[b]
    else:
        a_fault, b_fault = False, False

    # Compute impulse resolution
    a_geom = a_model.model_geometry
    b_geom = b_model.model_geometry
    j_n = compute_impulse_response(
        n=impact_normal, vel_ab=rel_velocity_atP, r_ap=r_ap, r_bp=r_bp, a_geom=a_geom, b_geom=b_geom
    )
    # Apply impulse to models
    a_vel_after = velocity_after_collision(impact_normal, a_vel, a_geom.m, j_n)
    b_vel_after = velocity_after_collision(-impact_normal, b_vel, b_geom.m, j_n)
    a_omega_after = rot_velocity_after_collision(r_ap, impact_normal, a_omega, a_geom.Iz, j_n)
    b_omega_after = rot_velocity_after_collision(r_bp, -impact_normal, b_omega, b_geom.Iz, j_n)
    a_model.set_velocity(a_vel_after, a_omega_after, in_model_frame=False)
    b_model.set_velocity(b_vel_after, b_omega_after, in_model_frame=False)

    # Log reports
    a_kenergy_delta = kinetic_energy(a_vel_after, a_geom.m) - kinetic_energy(a_vel, a_geom.m)
    b_kenergy_delta = kinetic_energy(b_vel_after, b_geom.m) - kinetic_energy(b_vel, b_geom.m)
    # todo rotational energy
    a_report = CollisionReportPlayer(
        locations=a_locations,
        at_fault=a_fault,
        footprint=a_shape,
        velocity=(a_vel, a_omega),
        velocity_after=(a_vel_after, a_omega_after),
        energy_delta=a_kenergy_delta,
    )
    b_report = CollisionReportPlayer(
        locations=b_locations,
        at_fault=b_fault,
        footprint=b_shape,
        velocity=(b_vel, b_omega),
        velocity_after=(b_vel_after, b_omega_after),
        energy_delta=b_kenergy_delta,
    )
    return CollisionReport(
        players={a: a_report, b: b_report},
        impact_point=impact_point,
        impact_normal=impact_normal,
        at_time=light_sim_context.time,
    )
