import os
from decimal import Decimal as D

from geometry import SE2_from_xytheta
from matplotlib import pyplot as plt
from numpy import deg2rad
from reprep import Report, MIME_PNG
from shapely.geometry import LinearRing, Polygon
from zuper_commons.text import pretty_msg

from dg_commons import PlayerName, DgSampledSequence, apply_SE2_to_shapely_geo
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.collision_visualisation import plot_collision
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


def get_simple_scenario() -> SimContext:
    x0_p1 = VehicleStateDyn(x=7, y=4, psi=deg2rad(60), vx=2, delta=0)

    dobs_shape = Polygon([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    x0_dobs1: DynObstacleState = DynObstacleState(x=7, y=11, psi=deg2rad(0), vx=3, vy=-1, dpsi=1)
    og_dobs1: ObstacleGeometry = ObstacleGeometry(m=1000, Iz=1000, e=0.2)
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
            VehicleCommands(acc=2, ddelta=+0.1),
            VehicleCommands(acc=1, ddelta=-0.5),
            VehicleCommands(acc=3, ddelta=+0.4),
            VehicleCommands(acc=5, ddelta=-1),
            VehicleCommands(acc=0, ddelta=-3),
        ],
    )
    size = 30
    boundaries = LinearRing([(0, 0), (0, size), (size, size), (size, 0), (0, 0)])
    poly1 = Polygon([[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]])
    poly2 = apply_SE2_to_shapely_geo(poly1, SE2_from_xytheta((7, 15, deg2rad(30))))

    static_obstacles = {0: StaticObstacle(boundaries), 1: StaticObstacle(poly1), 2: StaticObstacle(poly2)}
    players = {Ego: NPAgent(moving_vehicle), DObs1: NPAgent(dyn_obstacle_commands)}

    return SimContext(
        dg_scenario=DgScenario(static_obstacles=static_obstacles, use_road_boundaries=True),
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.3"), sim_time_after_collision=D(10), max_sim_time=D(10)),
    )


def get_collisions_report(sim_context: SimContext) -> Report:
    r = Report("AccidentsReport")
    for i, col_report in enumerate(sim_context.collision_reports):
        acc_id = "-".join(list(col_report.players.keys()))
        r.text(f"Accident-{acc_id}-{i}-report", text=pretty_msg(col_report.__str__()))
        coll_fig = r.figure(cols=5)
        with coll_fig.plot(f"Collision-{i}", MIME_PNG) as _:
            plot_collision(col_report, sim_log=sim_context.log)
        plt.close()
    return r


def test_sim_with_obstacles():
    sim_context = get_simple_scenario()
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    report = generate_report(sim_context)
    report.add_child(get_collisions_report(sim_context))
    # save report
    report_file = os.path.join(OUT_TESTS_DIR, f"sim_with_obstacles.html")
    report.to_html(report_file)
