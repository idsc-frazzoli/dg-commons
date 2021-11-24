from typing import List, Optional, Tuple

import numpy as np
from geometry import translation_angle_from_SE2
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from dg_commons import PlayerName
from dg_commons.sim import ImpactLocation, CollisionReport, logger, SimModel, SimTime
from dg_commons.sim.collision_structures import CollisionReportPlayer
from dg_commons.sim.collision_utils import (
    compute_impact_geometry,
    velocity_after_collision,
    kinetic_energy,
    compute_impulse_response,
    rot_velocity_after_collision,
    velocity_of_P_given_A,
    CollisionException,
    chek_who_is_at_fault,
)
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.simulator import SimContext


def impact_locations_from_polygons(a_model: SimModel, b_shape: BaseGeometry) -> List[Tuple[ImpactLocation, Polygon]]:
    """
    Checks the impact locations of A based on its mesh and the footprint of B
    :return:
    """
    locations: List[Tuple[ImpactLocation, Polygon]] = []
    a_mesh = a_model.get_mesh()
    for loc, loc_shape in a_mesh.items():
        if b_shape.intersects(loc_shape):
            locations.append((loc, loc_shape))
    if not locations:
        raise CollisionException(f"Detected a collision but unable to find the impact location for model {a_model}")
    return locations


def resolve_collision(a: PlayerName, b: PlayerName, sim_context: SimContext) -> Optional[CollisionReport]:
    """
    Resolves the collision between A and B using the impulse method.
    Sources:
        - http://www.chrishecker.com/images/e/e7/Gdmphys3.pdf
        - https://research.ncl.ac.uk/game/mastersdegree/gametechnologies/previousinformation/physics6collisionresponse/
    :returns A CollisionReport or None if the collision does not need to be solved (the two bodies are already separating)
    """
    a_model: SimModel = sim_context.models[a]
    b_model: SimModel = sim_context.models[b]
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
    who_is_at_fault = chek_who_is_at_fault(
        {a: a_model.get_pose(), b: b_model.get_pose()},
        impact_point=impact_point,
        lanelet_network=sim_context.dg_scenario.lanelet_network,
    )
    a_fault, b_fault = who_is_at_fault[a], who_is_at_fault[b]

    # Compute impulse resolution
    a_geom = a_model.get_geometry()
    b_geom = b_model.get_geometry()
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
        at_time=sim_context.time,
    )


def resolve_collision_with_environment(
    a: PlayerName, a_model: SimModel, b_obstacle: StaticObstacle, time: SimTime
) -> Optional[CollisionReport]:
    a_shape = a_model.get_footprint()
    b_shape = b_obstacle.shape
    # Compute collision geometry
    impact_normal, impact_point = compute_impact_geometry(a_shape, b_shape)
    a_cog = translation_angle_from_SE2(a_model.get_pose())[0]
    r_ap = np.array(impact_point.coords[0]) - np.array(a_cog)
    a_vel, a_omega = a_model.get_velocity(in_model_frame=False)
    a_vel_atP = velocity_of_P_given_A(a_vel, a_omega, r_ap)
    rel_velocity_atP = a_vel_atP

    if np.dot(rel_velocity_atP, impact_normal) < 0:
        logger.debug(f"Not solving the collision,  since they are already separating")
        return None

    # Update state of the model, it has collided
    a_model.has_collided = True

    # Compute collision locations
    a_locations = impact_locations_from_polygons(a_model, b_shape)

    # If you collide with the environment it is your fault
    a_fault = True

    # Compute impulse resolution
    a_geom = a_model.get_geometry()
    b_geom = b_obstacle.geometry
    r_bp = np.array([1, 1])  # irrelevant since it will disappear divided by infinity
    j_n = compute_impulse_response(
        n=impact_normal, vel_ab=rel_velocity_atP, r_ap=r_ap, r_bp=r_bp, a_geom=a_geom, b_geom=b_geom
    )
    # Apply impulse to models
    a_vel_after = velocity_after_collision(impact_normal, a_vel, a_geom.m, j_n)
    a_omega_after = rot_velocity_after_collision(r_ap, impact_normal, a_omega, a_geom.Iz, j_n)
    a_model.set_velocity(a_vel_after, a_omega_after, in_model_frame=False)

    # Log reports
    a_kenergy_delta = kinetic_energy(a_vel_after, a_geom.m) - kinetic_energy(a_vel, a_geom.m)
    # todo rotational energy
    a_report = CollisionReportPlayer(
        locations=a_locations,
        at_fault=a_fault,
        footprint=a_shape,
        velocity=(a_vel, a_omega),
        velocity_after=(a_vel_after, a_omega_after),
        energy_delta=a_kenergy_delta,
    )

    return CollisionReport(
        players={a: a_report},
        impact_point=impact_point,
        impact_normal=impact_normal,
        at_time=time,
    )
