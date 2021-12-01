import os
from decimal import Decimal as D

from numpy import deg2rad
from reprep import Report, MIME_GIF

from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.models.spacecraft import SpacecraftState, SpacecraftModel, SpacecraftCommands
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import SimContext, Simulator
from dg_commons.sim.simulator_animation import create_animation
from dg_commons_tests import OUT_TESTS_DIR

P1, P2, P3 = (
    PlayerName("P1"),
    PlayerName("P2"),
    PlayerName("P3"),
)


def get_simple_scenario() -> SimContext:
    scenario_name = "USA_Lanker-1_1_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)
    # road_boundary_obstacle, road_boundary_sg_rectangles = boundary.create_road_boundary_obstacle(scenario)

    x0_p1 = VehicleStateDyn(x=0, y=0, theta=deg2rad(60), vx=5, delta=0)
    x0_p2 = VehicleStateDyn(x=24, y=6, theta=deg2rad(150), vx=6, delta=0)
    x0_p3 = SpacecraftState(x=10, y=5, psi=deg2rad(-150), vx=5, vy=1, dpsi=1)
    models = {
        P1: VehicleModelDyn.default_car(x0_p1),
        P2: VehicleModelDyn.default_bicycle(x0_p2),
        P3: SpacecraftModel.default(x0_p3),
    }

    static_vehicle = DgSampledSequence[VehicleCommands](
        timestamps=[
            0,
        ],
        values=[
            VehicleCommands(acc=0, ddelta=0),
        ],
    )
    moving_vehicle = DgSampledSequence[VehicleCommands](
        timestamps=[0, 1, 2, 3, 4, 5],
        values=[
            VehicleCommands(acc=0, ddelta=0),
            VehicleCommands(acc=2, ddelta=+1),
            VehicleCommands(acc=1, ddelta=-0.5),
            VehicleCommands(acc=3, ddelta=+0.4),
            VehicleCommands(acc=-5, ddelta=-3),
            VehicleCommands(acc=0, ddelta=-3),
        ],
    )
    spacecraft_dynamic = DgSampledSequence[SpacecraftCommands](
        timestamps=[0, 1, 2, 3, 4, 5],
        values=[
            SpacecraftCommands(acc_left=0, acc_right=0),
            SpacecraftCommands(acc_left=2, acc_right=+1),
            SpacecraftCommands(acc_left=1, acc_right=-0.5),
            SpacecraftCommands(acc_left=3, acc_right=+0.4),
            SpacecraftCommands(acc_left=-5, acc_right=-3),
            SpacecraftCommands(acc_left=0, acc_right=-3),
        ],
    )
    players = {P1: NPAgent(moving_vehicle), P2: NPAgent(static_vehicle), P3: NPAgent(spacecraft_dynamic)}
    return SimContext(
        dg_scenario=DgScenario(scenario),
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(1), max_sim_time=D(7)),
    )


def generate_report(sim_context: SimContext) -> Report:
    r = Report("EpisodeVisualisation")
    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Animation", MIME_GIF) as fn:
        create_animation(file_path=fn, sim_context=sim_context, figsize=(16, 8), dt=30, dpi=120, plot_limits="auto")
    return r


def test_simple_simulation():
    sim_context = get_simple_scenario()
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    report = generate_report(sim_context)
    # save report
    report_file = os.path.join(OUT_TESTS_DIR, f"simple_sim.html")
    report.to_html(report_file)
