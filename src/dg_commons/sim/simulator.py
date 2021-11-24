from dataclasses import dataclass, field
from decimal import Decimal
from itertools import combinations
from typing import Mapping, Optional, List, Dict

from dg_commons import PlayerName, U
from dg_commons.sim import SimTime, CollisionReport, logger
from dg_commons.sim.agents.agent import Agent, TAgent
from dg_commons.sim.collision_utils import CollisionException
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator_structures import *
from dg_commons.time import time_function


@dataclass
class SimContext:
    """The simulation context that keeps track of everything, handle with care as it is passed around by reference and
    it is a mutable object"""

    dg_scenario: DgScenario
    models: Mapping[PlayerName, SimModel]
    players: Mapping[PlayerName, TAgent]
    param: SimParameters
    log: SimLog = field(default_factory=SimLog)
    time: SimTime = SimTime(0)
    seed: int = 0
    sim_terminated: bool = False
    collision_reports: List[CollisionReport] = field(default_factory=list)
    first_collision_ts: SimTime = SimTime("Infinity")

    def __post_init__(self):
        assert self.models.keys() == self.players.keys()
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
    last_observations: Optional[SimObservations] = SimObservations(players={}, time=Decimal(0))
    last_get_commands_ts: SimTime = SimTime("-Infinity")
    last_commands: Dict[PlayerName, U] = {}
    simlogger: Dict[PlayerName, PlayerLogger] = {}

    @time_function
    def run(self, sim_context: SimContext):
        logger.info("Beginning simulation.")
        for player_name, player in sim_context.players.items():
            player.on_episode_init(player_name)
            self.simlogger[player_name] = PlayerLogger()
        while not sim_context.sim_terminated:
            self.pre_update(sim_context)
            self.update(sim_context)
            self.post_update(sim_context)
        logger.info("Completed simulation. Writing logs...")
        for player_name in sim_context.players:
            sim_context.log[player_name] = self.simlogger[player_name].as_sequence()
        logger.info("Writing logs terminated.")

    def pre_update(self, sim_context: SimContext):
        """Prior to stepping the simulation we compute the observations for each agent"""
        self.last_observations.time = sim_context.time
        self.last_observations.players = {}
        for player_name, model in sim_context.models.items():
            # todo not always necessary to update observations
            player_obs = PlayerObservations(state=model.get_state(), occupancy=model.get_footprint())
            self.last_observations.players.update({player_name: player_obs})
        logger.debug(f"Pre update function, sim time {sim_context.time}")
        logger.debug(f"Last observations:\n{self.last_observations}")
        return

    def update(self, sim_context: SimContext):
        """The real step of the simulation"""
        t = sim_context.time
        update_commands: bool = (t - self.last_get_commands_ts) >= sim_context.param.dt_commands
        # fixme this can be parallelized later with ProcessPoolExecutor?
        for player_name, model in sim_context.models.items():
            if update_commands:
                actions = sim_context.players[player_name].get_commands(self.last_observations)
                self.last_commands[player_name] = actions
                self.simlogger[player_name].actions.add(t=t, v=actions)
                extra = sim_context.players[player_name].on_get_extra()
                if extra is not None:
                    self.simlogger[player_name].extra.add(t=t, v=extra)
            cmds = self.last_commands[player_name]
            model.update(cmds, dt=sim_context.param.dt)
            self.simlogger[player_name].states.add(t=t, v=model.get_state())
            logger.debug(f"Update function, sim time {sim_context.time:.2f}, player: {player_name}")
            logger.debug(f"New state {model.get_state()} reached applying {cmds}")
        if update_commands:
            self.last_get_commands_ts = t
        return

    def post_update(self, sim_context: SimContext):
        """
        Here all the operations that happen after we have stepped the simulation, e.g. collision checking
        """
        collision_enviroment = self._check_collisions_with_environment(sim_context)
        collision_players = self._check_collisions_among_players(sim_context)
        # after all the computations advance simulation time
        sim_context.time += sim_context.param.dt
        self._maybe_terminate_simulation(sim_context)
        return

    @staticmethod
    def _maybe_terminate_simulation(sim_context: SimContext):
        """Evaluates if the simulation needs to terminate based on the expiration of times"""
        termination_condition: bool = (
            sim_context.time > sim_context.param.max_sim_time
            or sim_context.time > sim_context.first_collision_ts + sim_context.param.sim_time_after_collision
        )
        sim_context.sim_terminated = termination_condition

    @staticmethod
    def _check_collisions_with_environment(sim_context: SimContext) -> bool:
        """Check collisions of the players with the environment"""
        from dg_commons.sim.collision import resolve_collision_with_environment  # import here to avoid circular imports

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
                    if report is not None:
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
        from dg_commons.sim.collision import resolve_collision  # import here to avoid circular imports

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
