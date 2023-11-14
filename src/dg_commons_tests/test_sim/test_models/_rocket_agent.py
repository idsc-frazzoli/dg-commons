from dg_commons import U, DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.rocket import RocketCommands, RocketState


class RocketAgent(Agent):
    cmds_plan: DgSampledSequence[RocketCommands]
    state_traj: DgSampledSequence[RocketState]
    myname: PlayerName

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        # todo compute a plan
        self.myname = init_sim_obs.my_name
        planner = RocketPlanner()
        cmds_plan, state_traj = planner.compute()

    def get_commands(self, sim_obs: SimObservations) -> RocketCommands:

        mystate = sim_obs.players[self.myname].state
        my_realstate = self.state_traj.at_interp(sim_obs.time)
        if mystate - my_realstate > 0.1:
            # update initial state for the planner
            cmds_plan, state_traj = planner.compute()

        # ZOH
        cmds = self.cmds_plan.at_or_previous(sim_obs.time)
        return cmds
