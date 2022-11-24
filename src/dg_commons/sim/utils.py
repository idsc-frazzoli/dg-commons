from dg_commons.sim.simulator import SimContext, Simulator

__all__ = ["run_simulation"]


def run_simulation(sim_context: SimContext) -> SimContext:
    sim = Simulator()
    sim.run(sim_context)
    return sim_context
