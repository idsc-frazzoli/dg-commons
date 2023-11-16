import os
from decimal import Decimal as D
from math import cos, sin, pi

from numpy import deg2rad, pi, arctan2
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

(
    P1,
    P2,
) = (PlayerName("PDM4ARocket"), PlayerName("P2"))


def get_planet_simcontext() -> SimContext:
    x0_p1 = RocketState(x=-8, y=-8, psi=pi / 12, m=2.1, vx=0, vy=0, dpsi=0.0, phi=deg2rad(30.0))

    satellite_shape = Point(0, 0).buffer(4)

    models = {
        P1: RocketModel.default(x0_p1),
    }

    cmds_p1 = DgSampledSequence[RocketCommands](
        timestamps=[0, 1, 2, 3, 4, 5],
        values=[
            RocketCommands(F_left=0, F_right=0, dphi=deg2rad(20)),
            RocketCommands(F_left=0.1, F_right=0.1, dphi=deg2rad(20)),
            RocketCommands(F_left=1, F_right=1, dphi=deg2rad(-20)),
            RocketCommands(F_left=0, F_right=2, dphi=deg2rad(-20)),
            RocketCommands(F_left=1, F_right=0, dphi=deg2rad(-20)),
            RocketCommands(F_left=2, F_right=2, dphi=deg2rad(0)),
            # RocketCommands(F_left=5, F_right=5,  dphi=deg2rad(20)),
            # RocketCommands(F_left=5, F_right=5,  dphi=deg2rad(20)),
            # RocketCommands(F_left=5, F_right=5, dphi=deg2rad(0)),
            # RocketCommands(F_left=5, F_right=5,  dphi=deg2rad(-20)),
            # RocketCommands(F_left=5, F_right=5, dphi=deg2rad(-20)),
            # RocketCommands(F_left=5, F_right=5, dphi=deg2rad(0)),
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
    x0_p1 = RocketState(x=-8, y=-8, psi=pi, m=2.1, vx=0, vy=0, dpsi=0.0, phi=0.0)

    # some static circular obstacles
    planet1 = Point(5, 5).buffer(3)

    # TODO: add a class with planets and their respective satellite kids
    # for a circular orbit with radius r and angular velocity w --> v=w*r
    mother_planet = planet1
    d_to_planet = 5
    omega = 1
    theta_planet = -pi / 2
    x = mother_planet.centroid.x + d_to_planet * cos(theta_planet)
    y = mother_planet.centroid.y + d_to_planet * sin(theta_planet)
    curr_psi = pi / 2 + arctan2(y - mother_planet.centroid.y, x - mother_planet.centroid.x)

    satellite_1 = DynObstacleState(x=x, y=y, psi=curr_psi, vx=omega * d_to_planet, vy=0, dpsi=omega)
    satellite_1_shape = Point(0, 0).buffer(1)

    models = {
        P1: RocketModel.default(x0_p1),
        P2: DynObstacleModel(
            satellite_1,
            shape=satellite_1_shape,
            og=ObstacleGeometry(m=5, Iz=50, e=0.5),
            op=DynObstacleParameters(vx_limits=(-100, 100), acc_limits=(-10, 10)),
        ),
    }

    cmds_p1 = DgSampledSequence[RocketCommands](
        timestamps=[0, 1, 2, 3, 4, 5],
        values=[
            RocketCommands(F_left=0, F_right=0, dphi=deg2rad(20)),
            RocketCommands(F_left=0.1, F_right=0.1, dphi=deg2rad(20)),
            RocketCommands(F_left=1, F_right=1, dphi=deg2rad(-20)),
            RocketCommands(F_left=0, F_right=2, dphi=deg2rad(-20)),
            RocketCommands(F_left=1, F_right=0, dphi=deg2rad(-20)),
            RocketCommands(F_left=2, F_right=2, dphi=deg2rad(0)),
        ],
    )
    centripetal_acc = omega**2 * d_to_planet
    cmds_p2 = DgSampledSequence[DynObstacleCommands](
        timestamps=[0],
        values=[
            DynObstacleCommands(acc_x=0, acc_y=centripetal_acc, acc_psi=0),
        ],
    )
    players = {P1: NPAgent(cmds_p1), P2: NPAgent(cmds_p2)}

    # some boundaries
    l = 20
    boundaries = LineString([(-l, -l), (-l, l), (l, l), (l, -l), (-l, -l)])

    static_obstacles: list[StaticObstacle] = [StaticObstacle(shape=s) for s in [boundaries, planet1]]

    return SimContext(
        dg_scenario=DgScenario(static_obstacles=static_obstacles),
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(4), max_sim_time=D(10)),
    )


def test_rocket_n_planet_sim():
    sim_context = get_planet_and_satellite_simcontext()
    # run simulation
    run_simulation(sim_context)
    report = generate_report(sim_context)
    # save report
    report_file = os.path.join(OUT_TESTS_DIR, "sim_rocket.html")
    report.to_html(report_file)


test_rocket_n_planet_sim()
