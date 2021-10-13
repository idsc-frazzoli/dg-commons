import numpy as np

from sim.models.vehicle import VehicleState, VehicleCommands
from sim.models.vehicle_dynamic import VehicleStateDyn, VehicleParameters, VehicleModelDyn


def test_vehicle_state_01():
    npstate = np.array([1, 2, 3, 4, 5])
    vstate = VehicleState.from_array(npstate)
    print(vstate)
    np.testing.assert_array_equal(npstate, vstate.as_ndarray())


def test_vehicle_commands_01():
    npcmds = np.array([1, 2])
    vcommands = VehicleCommands.from_array(npcmds)
    print(vcommands)
    np.testing.assert_array_equal(npcmds, vcommands.as_ndarray())


def test_vehicledyn_state_01():
    npstate = np.array([1, 2, 3, 4, 5, 6, 9.2])
    vstate = VehicleStateDyn.from_array(npstate)
    print(vstate)
    np.testing.assert_array_equal(npstate, vstate.as_ndarray())


def test_dyn_params():
    vp = VehicleParameters.default_car()
    print(vp)


def test_vehicledyn_inheritance():
    npstate = np.array([1, 2, 3, 4, 5, 6, 9.2])
    vstate = VehicleStateDyn.from_array(npstate)
    cardyn = VehicleModelDyn.default_car(vstate)

    print(cardyn.get_state())
    print(cardyn.update(VehicleCommands(6.0, 6.0), 0.3))
    print(cardyn.get_velocity(in_model_frame=False))
    cardyn.set_velocity(np.array([0, 2]), 5, in_model_frame=False)
    print(cardyn.get_velocity(in_model_frame=False))
