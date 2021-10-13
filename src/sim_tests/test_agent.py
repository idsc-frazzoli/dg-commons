from decimal import Decimal as D

import numpy as np

from dg_commons import DgSampledSequence
from sim.agents.agent import NPAgent
from sim.simulator_structures import SimObservations


def test_npagent():
    cmds = DgSampledSequence[float](timestamps=[0, 1, 2, 3, 4, 5], values=[0, 1, 2, 3, 4, 5])

    agent = NPAgent(cmds)
    ts_list = [D(i) for i in np.linspace(0, 6, 20)]
    sim_obs = SimObservations({}, D(0))
    for ts in ts_list:
        sim_obs.time = ts
        cmds = agent.get_commands(sim_obs=sim_obs)
        print(f"At {ts:.2f} agent cmds: {cmds}")
