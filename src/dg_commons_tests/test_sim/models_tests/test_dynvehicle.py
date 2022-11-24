import os
from decimal import Decimal as D

from numpy import deg2rad

from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.models import kmh2ms
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import SimContext, Simulator
from dg_commons_tests import OUT_TESTS_DIR
from dg_commons_tests.test_sim.test_sim import generate_report

P1, P2, P3 = (
    PlayerName("P1"),
    PlayerName("P2"),
    PlayerName("P3"),
)


def get_vehicle_dyn_scenario() -> SimContext:
    scenario_name = "USA_Lanker-1_1_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)
    # road_boundary_obstacle, road_boundary_sg_rectangles = boundary.create_road_boundary_obstacle(scenario)

    x0_p1 = VehicleStateDyn(x=0, y=0, psi=deg2rad(0), vx=kmh2ms(50), delta=0)
    x0_p2 = VehicleStateDyn(x=25, y=-10, psi=deg2rad(90), vx=kmh2ms(0), delta=-1)
    models = {
        P1: VehicleModelDyn.default_car(x0_p1),
        P2: VehicleModelDyn.default_bicycle(x0_p2),
    }

    cmds_p1 = DgSampledSequence[VehicleCommands](
        timestamps=[0, 1, 2, 3, 4, 5],
        values=[
            VehicleCommands(acc=5, ddelta=1),
            VehicleCommands(acc=5, ddelta=1),
            VehicleCommands(acc=5, ddelta=-5),
            VehicleCommands(acc=5, ddelta=4),
            VehicleCommands(acc=5, ddelta=-3),
            VehicleCommands(acc=5, ddelta=-3),
        ],
    )
    cmds_p2 = DgSampledSequence[VehicleCommands](
        timestamps=[0, 1, 2, 3, 4, 5],
        values=[
            VehicleCommands(acc=0, ddelta=0),
            VehicleCommands(acc=5, ddelta=0),
            VehicleCommands(acc=5, ddelta=-1),
            VehicleCommands(acc=3, ddelta=-1),
            VehicleCommands(acc=-5, ddelta=-3),
            VehicleCommands(acc=0, ddelta=3),
        ],
    )

    players = {P1: NPAgent(cmds_p1), P2: NPAgent(cmds_p2)}
    return SimContext(
        dg_scenario=DgScenario(scenario=scenario, use_road_boundaries=True),
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(4), max_sim_time=D(10)),
    )


def test_vehicle_dynamics_sim():
    sim_context = get_vehicle_dyn_scenario()
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    report = generate_report(sim_context)
    # save report
    report_file = os.path.join(OUT_TESTS_DIR, f"vehicle_dyn.html")
    report.to_html(report_file)
