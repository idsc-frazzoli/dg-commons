from typing import Mapping, MutableMapping
import numpy as np
from shapely import distance
from shapely.ops import nearest_points
from shapely.geometry import Polygon, Point
from shapely.affinity import translate
from geometry import SE2_from_xytheta
from dg_commons import apply_SE2_to_shapely_geo
from dg_commons import PlayerName, X
from dg_commons.sim import CollisionReport
from dg_commons.sim.goals import TPlanningGoal
from dg_commons.sim.simulator import LogEntry
from dg_commons.sim.simulator_structures import SimLog, SimModel
from dg_commons.seq.sequence import Timestamp


def has_collision(cr_list: list[CollisionReport]) -> bool:
    """Check whether the ego vehicle is involved in a collision."""
    ego_name = PlayerName("Ego")
    collided_players: set[PlayerName] = set()
    for cr in cr_list:
        collided_players.update((cr.players.keys()))
    has_collided = True if ego_name in collided_players else False
    return has_collided


def get_min_dist(logs: SimLog, models: MutableMapping[PlayerName, SimModel],
                 missions: Mapping[PlayerName, TPlanningGoal], ego_name: PlayerName,
                 t_range: tuple[Timestamp|None, Timestamp|None] = (None, None)) -> tuple[float, PlayerName, Timestamp]:
    """
    Get the minimum distance between the ego and any other agent throughout the simulation.
    Only timesteps within t_range are considered.
    """
    timesteps = logs[ego_name].states.timestamps
    min_dist = np.inf
    min_dist_agent = None
    min_dist_t = None
    for t in timesteps:
        if t_range[0] is not None and t < t_range[0]:
            continue
        if t_range[1] is not None and t > t_range[1]:
            break
        min_dist_at_t, min_dist_agent_at_t = get_min_dist_at_t(logs, models, missions, t, ego_name)
        if min_dist_at_t < min_dist:
            min_dist = min_dist_at_t
            min_dist_agent = min_dist_agent_at_t
            min_dist_t = t
    return min_dist, min_dist_agent, min_dist_t


def get_min_ttc_max_drac(logs: SimLog, models: MutableMapping[PlayerName, SimModel],
                         missions: Mapping[PlayerName, TPlanningGoal],
                         ego_name: PlayerName, t_range: tuple[Timestamp|None, Timestamp|None] = (None, None)) -> tuple[float, PlayerName, Timestamp]:
    """
    Get te minimum time-to-collision(ttc) and maximum deceleration-rate-to-avoid-collision(drac) throughout the
    simulation.
    Only timesteps within t_range are considered.
    """
    timesteps = logs[ego_name].states.timestamps
    min_ttc = np.inf
    min_ttc_agent = None
    min_ttc_t = None
    max_drac = -np.inf
    max_drac_agent = None
    max_drac_t = None
    for t in timesteps:
        if t_range[0] is not None and t < t_range[0]:
            continue
        if t_range[1] is not None and t > t_range[1]:
            break
        ttc_at_t, ttc_agent_at_t, drac_at_t, drac_agent_at_t = get_ttc_drac_at_t(logs, models, missions, t, ego_name)
        if ttc_at_t < min_ttc:
            min_ttc = ttc_at_t
            min_ttc_agent = ttc_agent_at_t
            min_ttc_t = t
        if drac_at_t > max_drac:
            max_drac = drac_at_t
            max_drac_agent = drac_agent_at_t
            max_drac_t = t
    return min_ttc, min_ttc_agent, min_ttc_t, max_drac, max_drac_agent, max_drac_t


def get_min_dist_at_t(logs: SimLog, models: MutableMapping[PlayerName, SimModel],
                      missions: Mapping[PlayerName, TPlanningGoal], t: Timestamp,
                      ego_name: PlayerName) -> tuple[float, PlayerName]:
    """
    Compute the minimum distance between the ego vehicle and the agents at current timestep.
    :param logs: simulation log
    :param models: models of all agents
    :param missions: missions of all agents
    :param t: sim time
    :param ego_name:
    :return: min dist, the closest agent from ego
    """
    logs_at_t = logs.at_interp(t)
    logs_at_t = _remove_finished_players(logs_at_t, missions)
    if ego_name not in logs_at_t.keys():
        # ego accomplished its goal
        return np.inf, None
    ego_state = logs_at_t[ego_name].state
    ego_model = models[ego_name]
    min_dist = np.inf
    min_dist_agent = None
    for name, log in logs_at_t.items():
        if name == ego_name:
            continue
        agent_state = log.state
        agent_model = models[name]
        dist, _ = _get_dist(ego_state, agent_state, ego_model, agent_model)
        if dist < min_dist:
            min_dist = dist
            min_dist_agent = name
    return min_dist, min_dist_agent


def get_ttc_drac_at_t(logs: SimLog, models: MutableMapping[PlayerName, SimModel],
                      missions: Mapping[PlayerName, TPlanningGoal], t: Timestamp, ego_name: PlayerName) -> tuple[float, PlayerName, float, PlayerName]:
    """
    Compute the minimum time-to-collision and the maximum deceleration-rate-to-avoid-collision for the ego vehicle
    against all agents at current timestep.
    :param logs: simulation log
    :param models: models of all agents
    :param missions: missions of all agents
    :param t: sim time
    :param ego_name:
    :return: ttc, the opponent agent involved in ttc, drac, the opponent agent involved in drac
    """
    # time to collision
    logs_at_t = logs.at_interp(t)
    logs_at_t = _remove_finished_players(logs_at_t, missions)
    if ego_name not in logs_at_t.keys():
        # ego accomplished its goal
        return np.inf, None, -np.inf, None
    ego_state = logs_at_t[ego_name].state
    ego_model = models[ego_name]
    ego_pos = np.array([ego_state.x, ego_state.y])
    ego_vel = ego_state.vx * np.array([np.cos(ego_state.psi), np.sin(ego_state.psi)])
    min_time = np.inf
    min_ttc_agent = None
    max_drac = -np.inf
    max_drac_agent = None
    for name, log in logs_at_t.items():
        if name == ego_name:
            continue
        agent_state = log.state
        agent_pos = np.array([agent_state.x, agent_state.y])
        agent_model = models[name]
        agent_vel = agent_state.vx * np.array([np.cos(agent_state.psi), np.sin(agent_state.psi)])
        dist_center = np.linalg.norm(agent_pos - ego_pos)
        rel_vel_along_dist = np.dot(agent_vel - ego_vel, (agent_pos - ego_pos) / dist_center)
        if rel_vel_along_dist > 0 or np.abs(dist_center / rel_vel_along_dist) > 5.0:
            # two agents are leaving each other or far enough
            continue
        ttc, ego_dtc, _ = _get_ttc(ego_state, agent_state, ego_model, agent_model)
        if ttc < min_time:
            min_time = ttc
            min_ttc_agent = name

        drac = ego_state.vx ** 2 / (2 * ego_dtc) if ego_dtc > 0 else np.inf
        if drac > max_drac:
            max_drac = drac
            max_drac_agent = name

    return min_time, min_ttc_agent, max_drac, max_drac_agent


def _get_dist(state1: X, state2: X, model1: SimModel, model2: SimModel) -> tuple[float, tuple[Point, Point]]:
    """get the minimum distance between two vehicles, considering their geometry"""
    pose1 = SE2_from_xytheta([state1.x, state1.y, state1.psi])
    poly1 = apply_SE2_to_shapely_geo(model1.vg.outline_as_polygon, pose1)
    pose2 = SE2_from_xytheta([state2.x, state2.y, state2.psi])
    poly2 = apply_SE2_to_shapely_geo(model2.vg.outline_as_polygon, pose2)
    nearest_pts = nearest_points(poly1, poly2)
    dist = distance(nearest_pts[0], nearest_pts[1])
    return dist, nearest_pts


def _get_ttc(state1: X, state2: X, model1: SimModel, model2: SimModel) -> tuple[float, float, float]:
    """
    Compute the time-to-collision and distance-to-collisions of the two vehicles, considering their geometry.
    :param poly1: geometry of the first agent
    :param poly2: geometry of the second agent
    :param state1: current state of the first agent
    :param state2: current state of the second agent
    :return: time-to-collision, distance-to-collision for agent1. distance-to-collision for agent2
    """
    pose1 = SE2_from_xytheta([state1.x, state1.y, state1.psi])
    poly1 = apply_SE2_to_shapely_geo(model1.vg.outline_as_polygon, pose1)
    pose2 = SE2_from_xytheta([state2.x, state2.y, state2.psi])
    poly2 = apply_SE2_to_shapely_geo(model2.vg.outline_as_polygon, pose2)
    ttc, dist_to_collision1, dist_to_collision2 = _get_ttc_of_poly_and_state(poly1, poly2, state1, state2)
    return ttc, dist_to_collision1, dist_to_collision2


def _get_ttc_of_poly_and_state(poly1: Polygon, poly2: Polygon, state1: X, state2: X) -> tuple[float, float, float]:
    """
    Compute the time-to-collision and distance-to-collisions of the two polygons, considering their geometry.
    :param poly1: geometry of the first agent
    :param poly2: geometry of the second agent
    :param state1: current state of the first agent
    :param state2: current state of the second agent
    :return: time-to-collision, distance-to-collision for agent1. distance-to-collision for agent2
    """
    time = 0
    max_time = 3.0
    timestep = 0.1
    v1 = state1.vx * np.array([np.cos(state1.psi), np.sin(state1.psi)])
    v2 = state2.vx * np.array([np.cos(state2.psi), np.sin(state2.psi)])
    delta_pos = (v2 - v1) * timestep
    while not poly1.intersects(poly2):
        if time > max_time:
            return np.inf, np.inf, np.inf
        # move poly2
        poly2 = translate(poly2, xoff=delta_pos[0], yoff=delta_pos[1])
        time += timestep
    return time, state1.vx * time, state2.vx * time


def _remove_finished_players(log_at_t: Mapping[PlayerName, LogEntry], missions: Mapping[PlayerName, TPlanningGoal]) -> \
        Mapping[PlayerName, LogEntry]:
    """
    Remove the agents that have acomplished their goal.
    :param log_at_t: collection of players and their log(state, command, etc.) at time t
    :param missions: the goal of each player
    :return: a now collection of running players and their logs.
    """
    log_new = {}
    for name, log in log_at_t.items():
        mission = missions[name]
        state = log.state
        if not mission.is_fulfilled(state):
            log_new[name] = log
    return log_new
