import os
from decimal import Decimal as D

from geometry import SE2_from_xytheta
from numpy import deg2rad
from shapely.geometry import LinearRing, Polygon

from dg_commons import PlayerName, DgSampledSequence, apply_SE2_to_shapely_geo
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import Simulator, SimContext
from dg_commons_tests import OUT_TESTS
from dg_commons_tests.test_sim.test_sim import generate_report

Ego = PlayerName("ego")


def get_maze_scenario() -> SimContext:
    x0_p1 = VehicleStateDyn(x=7, y=4, theta=deg2rad(60), vx=2, delta=0)
    models = {Ego: VehicleModelDyn.default_car(x0_p1)}

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
    boundaries = LinearRing([(0, 0), (0, 50), (50, 50), (50, 0), (0, 0)])
    poly1 = Polygon([[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]])
    poly2 = apply_SE2_to_shapely_geo(poly1, SE2_from_xytheta((7, 15, deg2rad(30))))

    static_obstacles = {0: StaticObstacle(boundaries), 1: StaticObstacle(poly1), 2: StaticObstacle(poly2)}
    players = {Ego: NPAgent(moving_vehicle)}

    return SimContext(
        dg_scenario=DgScenario(static_obstacles=static_obstacles, use_road_boundaries=True),
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(1), max_sim_time=D(4)),
    )


def test_sim_static_obstacles():
    sim_context = get_maze_scenario()
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    report = generate_report(sim_context)
    # save report
    report_file = os.path.join(OUT_TESTS, f"maze_sim.html")
    report.to_html(report_file)
