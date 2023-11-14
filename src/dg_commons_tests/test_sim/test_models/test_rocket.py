import os
from decimal import Decimal as D

from numpy import deg2rad
from shapely import LineString, Point

from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.models import kmh2ms
from dg_commons.sim.models.obstacles import StaticObstacle, DynObstacleParameters, ObstacleGeometry
from dg_commons.sim.models.obstacles_dyn import DynObstacleModel, DynObstacleState, DynObstacleCommands
from dg_commons.sim.models.rocket import RocketState, RocketCommands, RocketModel
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.utils import run_simulation
from dg_commons_tests import OUT_TESTS_DIR
from dg_commons_tests.test_sim.test_sim import generate_report

P1, P2, P3, P4 = (
    PlayerName("P1"),
    PlayerName("P2"),
    PlayerName("P3"),
    PlayerName("P4"),
)


def get_rocket_simcontext() -> SimContext:
    x0_p1 = RocketState(x=5, y=5, psi=deg2rad(45), vx=kmh2ms(50), vy=kmh2ms(0), dpsi=0)
    x0_p2 = DynObstacleState(x=25, y=10, psi=deg2rad(90), vx=kmh2ms(10), vy=kmh2ms(0), dpsi=0)
    satellite_shape = Point(0, 0).buffer(4)

    models = {
        P1: RocketModel.default(x0_p1),
        P2: DynObstacleModel(
            x0_p2,
            shape=satellite_shape,
            og=ObstacleGeometry(m=5, Iz=50, e=0.5),
            op=DynObstacleParameters(vx_limits=(-10, 10), acc_limits=(-10, 10)),
        ),
    }

    cmds_p1 = DgSampledSequence[RocketCommands](
        timestamps=[0, 1, 2, 3, 4, 5],
        values=[
            RocketCommands(acc_left=5, acc_right=1),
            RocketCommands(acc_left=5, acc_right=1),
            RocketCommands(acc_left=5, acc_right=-5),
            RocketCommands(acc_left=5, acc_right=4),
            RocketCommands(acc_left=5, acc_right=-3),
            RocketCommands(acc_left=5, acc_right=-3),
        ],
    )
    cmds_p2 = DgSampledSequence[DynObstacleCommands](
        timestamps=[0],
        values=[
            DynObstacleCommands(acc_x=0, acc_y=0, acc_psi=0),
        ],
    )
    players = {P1: NPAgent(cmds_p1), P2: NPAgent(cmds_p2)}

    # some boundaries
    boundaries = LineString([(0, 0), (0, 100), (100, 100), (100, 0), (0, 0)])
    # some static circular obstacles
    planet1 = Point(50, 50).buffer(10)
    planet2 = Point(30, 30).buffer(3)
    planet3 = Point(70, 80).buffer(5)

    static_obstacles: list[StaticObstacle] = [StaticObstacle(shape=s) for s in [boundaries, planet1, planet2, planet3]]

    return SimContext(
        dg_scenario=DgScenario(static_obstacles=static_obstacles),
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(4), max_sim_time=D(10)),
    )


def test_rocket_n_planet_sim():
    sim_context = get_rocket_simcontext()
    # run simulation
    run_simulation(sim_context)
    report = generate_report(sim_context)
    # save report
    report_file = os.path.join(OUT_TESTS_DIR, "rocket.html")
    report.to_html(report_file)
