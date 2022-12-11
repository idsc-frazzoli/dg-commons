from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from decimal import Decimal
from itertools import combinations
from time import perf_counter
from typing import Dict, List, Mapping, MutableMapping, Optional

from dg_commons import PlayerName, U, fd
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import CollisionReport, SimTime, logger
from dg_commons.sim.agents.agent import Agent, TAgent
from dg_commons.sim.collision_utils import CollisionException
from dg_commons.sim.models.obstacles_dyn import DynObstacleModel
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.sim_perception import IdObsFilter, ObsFilter
from dg_commons.sim.simulator_structures import *
from dg_commons.sim.simulator_structures import InitSimObservations
from dg_commons.time import time_function


@dataclass
class SimContext:
    """
    The simulation context that keeps track of everything,
    handle with care as it is passed around by reference, it is a mutable object.
    """

    dg_scenario: DgScenario
    """A driving games scenario"""
    models: MutableMapping[PlayerName, SimModel]
    """The simulation models for each player"""
    players: MutableMapping[PlayerName, TAgent]
    """The players in the simulation (Agents mapping observations to commands)"""
    param: SimParameters
    """The simulation parameters"""
    missions: Mapping[PlayerName, PlanningGoal] = field(default_factory=dict)
    """The ultimate goal of each player, it can be specified only for a subset of the players"""
    sensors: Mapping[PlayerName, ObsFilter] = field(default_factory=lambda: defaultdict(lambda: IdObsFilter()))
    """The sensors for each player, if not specified the default is the identity filter returning full observations"""
    log: SimLog = field(default_factory=SimLog)
    "The loggers for observations, commands, and extra information"
    time: SimTime = SimTime(0)
    "The clock for the simulator, keeps track of the current instant"
    seed: int = 0
    "The seed for reproducible randomness"
    sim_terminated: bool = False
    "Whether the simulation has terminated"
    collision_reports: List[CollisionReport] = field(default_factory=list)
    "The log of collision reports"
    first_collision_ts: SimTime = SimTime("Infinity")
    "The first collision time"
    description: str = ""
    "A string description for the specific simulation context"

    def __post_init__(self):
        assert self.models.keys() == self.players.keys()
        # players with a mission must be a subset of the players
        if self.missions is not None:
            assert all([player in self.models for player in self.missions])
            assert all([issubclass(type(self.missions[p]), PlanningGoal) for p in self.missions])
        assert isinstance(self.dg_scenario, DgScenario), self.dg_scenario
        for pname in self.models.keys():
            assert issubclass(type(self.models[pname]), SimModel)
            assert issubclass(type(self.players[pname]), Agent)


class Simulator:
    """
    A simulator has a loop made of 3 main steps:
        - A pre-update function creating the observations for the agents
        - An update function which asks the agents the commands and applies them to the dynamics of each model
        - A post-update function that checks the new states of all the models and resolves collisions
    """

    # fixme check if this is okay once you have multiple simulators running together
    last_observations: SimObservations = SimObservations(players=fd({}), time=Decimal(0))
    last_get_commands_ts: SimTime = SimTime("-Infinity")
    last_commands: Dict[PlayerName, U] = {}
    simlogger: Dict[PlayerName, PlayerLogger] = {}

    @time_function
    def run(self, sim_context: SimContext):
        logger.info("Beginning simulation.")
        # initialize the simulation
        for player_name, player in sim_context.players.items():
            scenario = deepcopy(sim_context.dg_scenario)
            init_obs = InitSimObservations(
                my_name=player_name,
                seed=sim_context.seed,
                dg_scenario=scenario,
                goal=deepcopy(sim_context.missions.get(player_name)),
            )
            player.on_episode_init(init_obs)
            self.simlogger[player_name] = PlayerLogger()
        # actual simulation loop
        while not sim_context.sim_terminated:
            self.pre_update(sim_context)
            self.update(sim_context)
            self.post_update(sim_context)
        logger.info("Completed simulation. Writing logs...")
        for player_name in sim_context.models:
            sim_context.log[player_name] = self.simlogger[player_name].as_sequence()
        logger.info("Writing logs terminated.")

    def pre_update(self, sim_context: SimContext):
        """Prior to stepping the simulation we compute the observations for each agent"""
        players_observations: Dict[PlayerName, PlayerObservations] = {}
        for player_name, model in sim_context.models.items():
            # todo not always necessary to update observations
            player_obs = PlayerObservations(state=model.get_state(), occupancy=model.get_footprint())
            players_observations.update({player_name: player_obs})
        self.last_observations = replace(
            self.last_observations, players=fd(players_observations), time=sim_context.time
        )

        logger.debug(f"Pre update function, sim time {sim_context.time}")
        logger.debug(f"Last observations:\n{self.last_observations}")
        return

    def update(self, sim_context: SimContext):
        """The real step of the simulation"""
        # fixme this can be parallelized later with ProcessPoolExecutor?
        t = sim_context.time
        update_commands: bool = (t - self.last_get_commands_ts) >= sim_context.param.dt_commands
        for player_name, agent in sim_context.players.items():
            state = sim_context.models[player_name].get_state()
            self.simlogger[player_name].states.add(t=t, v=state)
            if update_commands:
                p_observations = sim_context.sensors[player_name].sense(
                    sim_context.dg_scenario, self.last_observations, player_name
                )
                tic = perf_counter()
                cmds = agent.get_commands(p_observations)
                toc = perf_counter()
                self.last_commands[player_name] = cmds
                self.simlogger[player_name].commands.add(t=t, v=cmds)
                self.simlogger[player_name].info.add(t=t, v=toc - tic)
                extra = agent.on_get_extra()
                if extra is not None:
                    self.simlogger[player_name].extra.add(t=t, v=extra)
            cmds = self.last_commands[player_name]
            model = sim_context.models[player_name]
            model.update(cmds, dt=sim_context.param.dt)
            logger.debug(f"Update function, sim time {sim_context.time:.2f}, player: {player_name}")
            logger.debug(f"New state {model.get_state()} reached applying {cmds}")
        if update_commands:
            self.last_get_commands_ts = t
        return

    def post_update(self, sim_context: SimContext):
        """
        Here all the operations that happen after we have stepped the simulation, e.g. collision checking
        """
        # after all the computations advance simulation time
        sim_context.time += sim_context.param.dt
        # collision checking
        collision_enviroment = self._check_collisions_with_environment(sim_context)
        collision_players = self._check_collisions_among_players(sim_context)
        # check if the simulation is over
        self._maybe_terminate_simulation(sim_context)
        # remove finished players
        self._remove_finished_players(sim_context)
        return

    @staticmethod
    def _maybe_terminate_simulation(sim_context: SimContext):
        """Evaluates if the simulation needs to terminate based on the expiration of times.
        The simulation is considered terminated if:
        - the maximum time has expired
        - the minimum time after the first collision has expired
        - all missions have been fulfilled
        """
        # missions_completed: bool = (
        #     all(m.is_fulfilled(sim_context.models[p].get_state()) for p, m in sim_context.missions.items())
        #     if sim_context.missions
        #     else False
        # )
        termination_condition: bool = (
            sim_context.time > sim_context.param.max_sim_time
            or sim_context.time > sim_context.first_collision_ts + sim_context.param.sim_time_after_collision
            or not sim_context.players
        )
        sim_context.sim_terminated = termination_condition

    @staticmethod
    def _check_collisions_with_environment(sim_context: SimContext) -> bool:
        """Check collisions of the players with the environment"""
        from dg_commons.sim.collision import (
            resolve_collision_with_environment,  # import here to avoid circular imports
        )

        env_obstacles = sim_context.dg_scenario.strtree_obstacles
        collision = False
        for p, p_model in sim_context.models.items():
            p_shape = p_model.get_footprint()
            items = env_obstacles.query_items(p_shape)
            for idx in items:
                candidate = sim_context.dg_scenario.static_obstacles[idx]
                if p_shape.intersects(candidate.shape):
                    try:
                        report: Optional[CollisionReport] = resolve_collision_with_environment(
                            p, p_model, candidate, sim_context.time
                        )
                    except CollisionException as e:
                        logger.warn(f"Failed to resolve collision of {p} with environment because:\n{e.args}")
                        report = None
                    if report is not None and not isinstance(p_model, DynObstacleModel):
                        logger.info(f"Player {p} collided with the environment")
                        collision = True
                        sim_context.collision_reports.append(report)
                        if sim_context.time < sim_context.first_collision_ts:
                            sim_context.first_collision_ts = sim_context.time
        return collision

    @staticmethod
    def _check_collisions_among_players(sim_context: SimContext) -> bool:
        """
        This checks only collision location at the current step, tunneling effects and similar are ignored
        :param sim_context:
        :return: True if at least one collision happened, False otherwise
        """
        from dg_commons.sim.collision import (
            resolve_collision,  # import here to avoid circular imports
        )

        collision = False
        for p1, p2 in combinations(sim_context.models, 2):
            a_shape = sim_context.models[p1].get_footprint()
            b_shape = sim_context.models[p2].get_footprint()
            if a_shape.intersects(b_shape):
                try:
                    report: Optional[CollisionReport] = resolve_collision(p1, p2, sim_context)
                except CollisionException as e:
                    logger.warn(f"Failed to resolve collision between {p1} and {p2} because:\n{e.args}")
                    report = None
                if report is not None:
                    logger.info(f"Detected a collision between {p1} and {p2}")
                    collision = True
                    if report.at_time < sim_context.first_collision_ts:
                        sim_context.first_collision_ts = report.at_time
                    sim_context.collision_reports.append(report)
        return collision

    @staticmethod
    def _remove_finished_players(sim_context: SimContext):
        """We remove players that complete their mission"""
        for p, m in sim_context.missions.items():
            # if p is still active
            if p in sim_context.players:
                p_state = sim_context.models[p].get_state()
                if m.is_fulfilled(p_state):
                    sim_context.players.pop(p)
