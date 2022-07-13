from math import pi

from dg_commons.sim.models.vehicle import VehicleState, VehicleCommands, VehicleModel
from dg_commons.sim.scenarios.npagent_from_dyn_obstacle import reconstruct_input_transition


def test_npagent_from_dynobstacle():
    x0: VehicleState = VehicleState(0, 0, pi, 0, 0)
    x1: VehicleState = VehicleState(5, 5, pi / 2, 3, 1)
    dt = 5
    reconstruct_input_transition(
        x0,
        x1,
        VehicleCommands,
        vehicle_dynamics=VehicleModel,
    )
