import os
from decimal import Decimal as D

from numpy import deg2rad, pi
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

def get_planet_simcontext() -> SimContext:
    x0_p1 = RocketState(x=-8, y=-8, theta= pi, m=2.1, vx=0, vy=0, dtheta= 0.0, phi= 0.0)

    satellite_shape = Point(0, 0).buffer(4)

    models = {
        P1: RocketModel.default(x0_p1),
    }

    cmds_p1 = DgSampledSequence[RocketCommands](
        timestamps=[0, 1, 2, 3, 4, 5],
        values=[
            RocketCommands(F_left=5, F_right=1,  dphi=deg2rad(4)),
            RocketCommands(F_left=5, F_right=1,  dphi=deg2rad(4)),
            RocketCommands(F_left=5, F_right=-5, dphi=deg2rad(4)),
            RocketCommands(F_left=5, F_right=4,  dphi=deg2rad(4)),
            RocketCommands(F_left=5, F_right=-3, dphi=deg2rad(4)),
            RocketCommands(F_left=5, F_right=-3, dphi=deg2rad(4)),
        ],
    )
    players = {P1: NPAgent(cmds_p1)}

    # some boundaries
    boundaries = LineString([(-10, -10), (-10, 10), (10, 10), (10, -10), (-10, -10)])
    # some static circular obstacles
    planet1 = Point(5, 4).buffer(3)
    planet2 = Point(5, -4).buffer(3)
    planet3 = Point(0, 0).buffer(2)

    static_obstacles: list[StaticObstacle] = [StaticObstacle(shape=s) for s in [boundaries, planet1, planet2, planet3]]

    return SimContext(
        dg_scenario=DgScenario(static_obstacles=static_obstacles),
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(4), max_sim_time=D(10)),
    )

def get_planet_and_satellite_simcontext() -> SimContext:
    x0_p1 = RocketState(x=-8, y=-8, theta= pi, m=2.1, vx=0, vy=0, dtheta= 0.0, phi= 0.0)
    x0_planet1 = DynObstacleState(x=5.0, y=4.0, psi=0, vx=0, vy=0, dpsi=0)
    x0_planet2 = DynObstacleState(x=5.0, y=-4.0, psi=0, vx=0, vy=0, dpsi=0)
    x0_planet3 = DynObstacleState(x=0.0, y=0.0, psi=0, vx=0, vy=0, dpsi=0)

    satellite_shape = Point(0, 0).buffer(4)

    models = {
        P1: RocketModel.default(x0_p1),
        P2: DynObstacleModel(
            x0_planet1,
            shape=satellite_shape,
            og=ObstacleGeometry(m=5, Iz=50, e=0.5),
            op=DynObstacleParameters(vx_limits=(-10, 10), acc_limits=(-10, 10)),
        ),
    }

    cmds_p1 = DgSampledSequence[RocketCommands](
        timestamps=[0, 1, 2, 3, 4, 5],
        values=[
            RocketCommands(F_left=5, F_right=1,  dphi=deg2rad(4)),
            RocketCommands(F_left=5, F_right=1,  dphi=deg2rad(4)),
            RocketCommands(F_left=5, F_right=-5, dphi=deg2rad(4)),
            RocketCommands(F_left=5, F_right=4,  dphi=deg2rad(4)),
            RocketCommands(F_left=5, F_right=-3, dphi=deg2rad(4)),
            RocketCommands(F_left=5, F_right=-3, dphi=deg2rad(4)),
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
    boundaries = LineString([(-10, -10), (-10, 10), (10, 10), (10, -10), (-10, -10)])
    # some static circular obstacles
    planet1 = Point(5, 4).buffer(3)
    planet2 = Point(5, -4).buffer(3)
    planet3 = Point(0, 0).buffer(2)

    static_obstacles: list[StaticObstacle] = [StaticObstacle(shape=s) for s in [boundaries, planet1, planet2, planet3]]

    return SimContext(
        dg_scenario=DgScenario(static_obstacles=static_obstacles),
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(4), max_sim_time=D(10)),
    )

def test_rocket_n_planet_sim():
    sim_context = get_planet_simcontext()
    # run simulation
    run_simulation(sim_context)
    report = generate_report(sim_context)
    # save report
    report_file = os.path.join(OUT_TESTS_DIR, "rocket.html")
    report.to_html(report_file)


test_rocket_n_planet_sim()