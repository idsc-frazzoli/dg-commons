import os
from decimal import Decimal as D

from geometry import SE2_from_xytheta
from numpy import deg2rad
from shapely.geometry import LinearRing, Polygon

from dg_commons import PlayerName, DgSampledSequence, apply_SE2_to_shapely_geo
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.models.obstacles import StaticObstacle, DynObstacleParameters, ObstacleGeometry
from dg_commons.sim.models.obstacles_dyn import DynObstacleModel, DynObstacleState, DynObstacleCommands
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import Simulator, SimContext
from dg_commons_tests import OUT_TESTS_DIR
from dg_commons_tests.test_sim.test_sim import generate_report

Ego = PlayerName("ego")
DObs1 = PlayerName("DObs1")


def get_maze_scenario() -> SimContext:
    x0_p1 = VehicleStateDyn(x=7, y=4, theta=deg2rad(60), vx=2, delta=0)

    dobs_shape = Polygon([[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]])
    x0_dobs1: DynObstacleState = DynObstacleState(x=7, y=9, psi=deg2rad(67), vx=2, vy=-3, dpsi=-0.5)
    og_dobs1: ObstacleGeometry = ObstacleGeometry(m=100, Iz=100, e=0.3)
    op_dops1: DynObstacleParameters = DynObstacleParameters(vx_limits=(-10, 10), acc_limits=(-1, 1))

    models = {
        Ego: VehicleModelDyn.default_car(x0_p1),
        DObs1: DynObstacleModel(x0_dobs1, dobs_shape, og_dobs1, op_dops1),
    }

    dyn_obstacle_commands = DgSampledSequence[DynObstacleCommands](
        timestamps=[0],
        values=[DynObstacleCommands(acc_x=0, acc_y=0, acc_psi=0)],
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
    boundaries = LinearRing([(0, 0), (0, 50), (50, 50), (50, 0), (0, 0)])
    poly1 = Polygon([[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]])
    poly2 = apply_SE2_to_shapely_geo(poly1, SE2_from_xytheta((7, 15, deg2rad(30))))

    static_obstacles = {0: StaticObstacle(boundaries), 1: StaticObstacle(poly1), 2: StaticObstacle(poly2)}
    players = {Ego: NPAgent(moving_vehicle), DObs1: NPAgent(dyn_obstacle_commands)}

    return SimContext(
        dg_scenario=DgScenario(static_obstacles=static_obstacles, use_road_boundaries=True),
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.3"), sim_time_after_collision=D(15), max_sim_time=D(15)),
    )


def test_sim_static_obstacles():
    sim_context = get_maze_scenario()
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    report = generate_report(sim_context)
    # save report
    report_file = os.path.join(OUT_TESTS_DIR, f"maze_sim.html")
    report.to_html(report_file)
