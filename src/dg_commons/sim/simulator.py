from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from decimal import Decimal
from itertools import combinations
from time import perf_counter
from typing import Mapping, MutableMapping, Optional

from dg_commons import PlayerName, U, fd
from dg_commons.sim import CollisionReport, SimTime, logger
from dg_commons.sim.agents.agent import Agent, TAgent, GlobalPlanner
from dg_commons.sim.collision_utils import CollisionException
from dg_commons.sim.goals import PlanningGoal, TPlanningGoal
from dg_commons.sim.models.obstacles_dyn import DynObstacleModel
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.sim_perception import IdObsFilter, ObsFilter

from dg_commons.sim.simulator_structures import *
from dg_commons.sim.shared_goals import SharedPolygonGoalsManager
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
    missions: Mapping[PlayerName, TPlanningGoal] = field(default_factory=dict)
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
    collision_reports: list[CollisionReport] = field(default_factory=list)
    "The log of collision reports"
    first_collision_ts: SimTime = SimTime("Infinity")
    "The first collision time"
    description: str = ""
    "A string description for the specific simulation context"
    global_planner: Optional[GlobalPlanner] = None
    "Optional global planner for on_episode_init"
    shared_goals_manager: Optional[SharedPolygonGoalsManager] = None
    "Optional manager for shared goals and collection points"

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
    last_commands: dict[PlayerName, U] = {}
    simlogger: dict[PlayerName, PlayerLogger] = {}
    disabled_players: list[PlayerName] = []
    """List of players that have been disabled due to collisions"""

    @time_function
    def run(self, sim_context: SimContext):
        logger.info("~~~~~> Beginning simulation")
        # initialize the simulation
        init_obs = {}
        for player_name, player in sim_context.players.items():
            init_obs[player_name] = InitSimObservations(
                my_name=player_name,
                seed=sim_context.seed,
                dg_scenario=deepcopy(sim_context.dg_scenario),
                goal=deepcopy(sim_context.missions.get(player_name)),
                model_geometry=sim_context.models[player_name].model_geometry,
                model_params=sim_context.models[player_name].model_params,
                initial_state=None,
            )
            player.on_episode_init(init_obs[player_name])

            # NOTE: The inital state is not exposed to the agent at initialization.
            # But it need to be exposed to the global planner later. We set it here after on_episode_init.
            init_obs[player_name] = replace(
                init_obs[player_name], initial_state=sim_context.models[player_name].get_state()
            )
            self.simlogger[player_name] = PlayerLogger()

        if sim_context.global_planner is not None:
            goals = sim_context.shared_goals_manager.all_goals if sim_context.shared_goals_manager is not None else None
            collection_points = (
                sim_context.shared_goals_manager.collection_points
                if sim_context.shared_goals_manager is not None
                else None
            )
            init_global_obs = InitSimGlobalObservations(
                players_obs=init_obs,
                seed=sim_context.seed,
                dg_scenario=deepcopy(sim_context.dg_scenario),
                goals=deepcopy(goals),
                collection_points=deepcopy(collection_points),
            )
            serialzied_global_plan = sim_context.global_planner.send_plan(init_global_obs)
            if not isinstance(serialzied_global_plan, str):
                raise TypeError(f"Global planner returned a plan of type {type(serialzied_global_plan)}, expected str")
            for player_name, player in sim_context.players.items():
                player.on_receive_global_plan(serialzied_global_plan)

        # actual simulation loop
        while not sim_context.sim_terminated:
            self.pre_update(sim_context)
            self.update(sim_context)
            self.post_update(sim_context)
        logger.info("<~~~~~ Completed simulation")
        for player_name in sim_context.models:
            sim_context.log[player_name] = self.simlogger[player_name].as_sequence()
        logger.debug("Writing logs terminated.")

    def pre_update(self, sim_context: SimContext):
        """Prior to stepping the simulation we compute the observations for each agent.
        Note that observations are generated only about active players.
        """

        # we update the observations only when we will need to use them
        if self._need_to_update_commands(sim_context):
            players_observations: dict[PlayerName, PlayerObservations] = {}
            for player_name in sim_context.players:
                model = sim_context.models[player_name]

                collected_goal = None
                if sim_context.shared_goals_manager is not None:
                    collected_goal = sim_context.shared_goals_manager.agent_carrying.get(player_name)

                player_obs = PlayerObservations(
                    state=model.get_state(), occupancy=model.get_footprint(), collected_goal_id=collected_goal
                )
                players_observations.update({player_name: player_obs})

            # Get available shared goals if manager exists
            available_goals_obs = None
            if sim_context.shared_goals_manager is not None:
                available_goals = sim_context.shared_goals_manager.get_available_goals()
                available_goals_obs = {}
                for goal in available_goals:
                    goal_obs = SharedGoalObservation(occupancy=goal.polygon)
                    available_goals_obs[goal.goal_id] = goal_obs

            self.last_observations = replace(
                self.last_observations,
                players=fd(players_observations),
                time=sim_context.time,
                available_goals=fd(available_goals_obs) if available_goals_obs else None,
            )

            logger.debug(f"Pre update function, sim time {sim_context.time}")
            logger.debug(f"Last observations:\n{self.last_observations}")
        return

    def update(self, sim_context: SimContext):
        """The real step of the simulation"""
        # fixme this can be parallelized later with ProcessPoolExecutor?
        t = sim_context.time
        for player_name, agent in sim_context.players.items():
            # if hasattr(agent, "_capacity"):
            #     self._ensure_agent_within_capacity(agent, player_name)
            state = sim_context.models[player_name].get_state()
            self.simlogger[player_name].states.add(t=t, v=state)
            if self._need_to_update_commands(sim_context):
                if player_name in self.disabled_players:
                    continue
                p_observations = sim_context.sensors[player_name].sense(
                    sim_context.dg_scenario, self.last_observations, player_name
                )
                tic = perf_counter()
                cmds = agent.get_commands(p_observations)
                extra = agent.on_get_extra()
                toc = perf_counter()
                self.last_commands[player_name] = cmds
                self.simlogger[player_name].commands.add(t=t, v=cmds)
                self.simlogger[player_name].info.add(t=t, v=toc - tic)
                if extra is not None:
                    self.simlogger[player_name].extra.add(t=t, v=extra)
            cmds = self.last_commands[player_name]
            model = sim_context.models[player_name]
            model.update(cmds, dt=sim_context.param.dt)
            logger.debug(f"Update function, sim time {sim_context.time:.2f}, player: {player_name}")
            logger.debug(f"New state {model.get_state()} reached applying {cmds}")
        if self._need_to_update_commands(sim_context):
            self.last_get_commands_ts = t
        return

    def post_update(self, sim_context: SimContext):
        """
        Here all the operations that happen after we have stepped the simulation, e.g. collision checking
        """
        # after all the computations advance simulation time
        sim_context.time += sim_context.param.dt
        # update shared goals manager
        if sim_context.shared_goals_manager is not None:
            self._update_shared_goals_manager(sim_context)
        # check if the simulation is over
        self._maybe_terminate_simulation(sim_context)
        if sim_context.sim_terminated:
            return
        # collision checking
        _ = self._check_collisions_with_environment(sim_context)
        _ = self._check_collisions_among_players(sim_context)
        # update disabled players
        self._update_disabled_players(sim_context)
        return

    @staticmethod
    def _maybe_terminate_simulation(sim_context: SimContext):
        """Evaluates if the simulation needs to terminate.
        The simulation is considered terminated if:
        - All objects have been collected and delivered to the collection area
        - The time limit is reached
        - A robot collides with an obstacle or another robot
        """
        termination_condition: bool = False

        # Check if time limit is reached
        if sim_context.time > sim_context.param.max_sim_time:
            termination_condition = True

        # Check if enough time has passed since the first collision
        if sim_context.time > sim_context.first_collision_ts + sim_context.param.sim_time_after_collision:
            termination_condition = True

        # Check if all objects have been collected and delivered
        if sim_context.shared_goals_manager is not None and sim_context.shared_goals_manager.is_all_goals_delivered():
            termination_condition = True

        # Terminate if all players have collided
        if all(model.has_collided for model in sim_context.models.values()):
            termination_condition = True

        sim_context.sim_terminated = termination_condition

    @staticmethod
    def _check_collisions_with_environment(sim_context: SimContext) -> bool:
        """Check collisions of the players with the environment"""
        from dg_commons.sim.collision import (
            resolve_collision_with_environment,  # import here to avoid circular imports
        )

        env_obstacles = sim_context.dg_scenario.strtree_obstacles
        collision = False
        for p in sim_context.players:
            if isinstance(sim_context.models[p], DynObstacleModel) and (
                sim_context.models[p].tag == "asteroid" or sim_context.models[p].tag == "satellite"
            ):
                # dynamic obstacles that impact everywhere do not collide with the environment
                continue
            p_model = sim_context.models[p]
            p_shape = p_model.get_footprint()
            items = env_obstacles.query(p_shape, predicate="intersects")
            for idx in items:
                sobstacle = sim_context.dg_scenario.static_obstacles[idx]
                try:
                    report: Optional[CollisionReport] = resolve_collision_with_environment(
                        p, p_model, sobstacle, sim_context.time
                    )
                except CollisionException as e:
                    logger.warn(f"Failed to resolve collision of {p} with environment because:\n{e.args}")
                    report = CollisionReport.get_empty(players={p: None}, at_time=sim_context.time)
                if report is not None and not isinstance(p_model, DynObstacleModel):
                    logger.debug(f"Player {p} collided with the environment")
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
        for p1, p2 in combinations(sim_context.players, 2):
            if (
                isinstance(sim_context.models[p1], DynObstacleModel)
                and isinstance(sim_context.models[p2], DynObstacleModel)
                and sim_context.models[p1].tag == "asteroid"
                and sim_context.models[p2].tag == "asteroid"
            ):
                continue
            a_shape = sim_context.models[p1].get_footprint()
            b_shape = sim_context.models[p2].get_footprint()
            if a_shape.intersects(b_shape):
                try:
                    report: Optional[CollisionReport] = resolve_collision(p1, p2, sim_context)
                except CollisionException as e:
                    logger.warn(f"Failed to resolve collision between {p1} and {p2} because:\n{e.args}")
                    report = CollisionReport.get_empty(players={p1: None, p2: None}, at_time=sim_context.time)
                if report is not None:
                    logger.debug(f"Detected a collision between {p1} and {p2}")
                    collision = True
                    if report.at_time < sim_context.first_collision_ts:
                        sim_context.first_collision_ts = report.at_time
                    sim_context.collision_reports.append(report)
        return collision

    def _remove_finished_players(self, sim_context: SimContext):
        """We remove players that complete their mission"""
        for p, m in sim_context.missions.items():
            # if p is still active
            if p in sim_context.players:
                p_state = sim_context.models[p].get_state()
                if m.is_fulfilled(p_state, sim_context.time):
                    t = sim_context.time
                    self.simlogger[p].states.add(t=t, v=p_state)
                    sim_context.players.pop(p)

    def _update_disabled_players(self, sim_context: SimContext):
        """We update the list of disabled players"""
        for pname, pmodel in sim_context.models.items():
            if pmodel.has_collided:
                if pname not in self.disabled_players:
                    self.disabled_players.append(pname)
                    logger.info(f"Player {pname} has been disabled due to collision at time {sim_context.time:.2f}s")

    # @staticmethod
    # def _ensure_agent_within_capacity(agent: Agent, player_name: PlayerName) -> None:
    #     """Verify that an agent does not exceed its declared capacity."""
    #     get_current_load = getattr(agent, "get_current_load", None)
    #     get_capacity = getattr(agent, "get_capacity", None)
    #     if callable(get_current_load) and callable(get_capacity):
    #         current_load = get_current_load()
    #         capacity = get_capacity()
    #         if current_load > capacity:
    #             raise RuntimeError(f"Agent '{player_name}' load {current_load} exceeds capacity {capacity}")

    @staticmethod
    def _update_shared_goals_manager(sim_context: SimContext):
        """Update shared goals manager if present"""
        agents_states = {pn: sim_context.models[pn].get_state() for pn in sim_context.players}
        events = sim_context.shared_goals_manager.update(agents_states, sim_context.time)
        if events["goals_collected"]:
            logger.info(f"Goals collected: {events['goals_collected']} at time {sim_context.time:.2f}s")
        if events["goals_delivered"]:
            logger.info(f"Goals delivered: {events['goals_delivered']} at time {sim_context.time:.2f}s")

    def _need_to_update_commands(self, sim_context: SimContext) -> bool:
        """Checks if we need to update the commands of the players"""
        return (sim_context.time - self.last_get_commands_ts) >= sim_context.param.dt_commands
