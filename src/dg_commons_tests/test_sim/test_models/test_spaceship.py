import os
from decimal import Decimal as D
from math import cos, sin, pi

import matplotlib.pyplot as plt

from numpy import deg2rad, pi, arctan2
from shapely import LineString, Point

from dg_commons import PlayerName, DgSampledSequence
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.models.obstacles import StaticObstacle, DynObstacleParameters, ObstacleGeometry
from dg_commons.sim.models.obstacles_dyn import DynObstacleModel, DynObstacleState, DynObstacleCommands
from dg_commons.sim.models.spaceship import SpaceshipState, SpaceshipCommands, SpaceshipModel
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import SimContext
from dg_commons.sim.utils import run_simulation
from dg_commons_tests import OUT_TESTS_DIR
from dg_commons_tests.test_sim.test_sim import generate_report

P1, P2 = PlayerName("PDM4AR"), PlayerName("P2")


def get_planet_n_satellite_simcontext() -> SimContext:
    x0_p1 = SpaceshipState(x=-8, y=-8, psi=pi / 2, vx=0, vy=0, dpsi=0.0, delta=0.0, m=3)

    # some static circular obstacles
    planet1 = Point(0, 0).buffer(3)

    # TODO: add a class with planets and their respective satellite kids
    # for a circular orbit with radius r and angular velocity w --> v=w*r
    mother_planet = planet1
    orbit_r = 5
    omega = 1
    tau = -pi / 2  # initial angle of satellite w.r.t. mother planet
    x = mother_planet.centroid.x + orbit_r * cos(tau)
    y = mother_planet.centroid.y + orbit_r * sin(tau)
    curr_psi = pi / 2 + arctan2(y - mother_planet.centroid.y, x - mother_planet.centroid.x)

    satellite_1 = DynObstacleState(x=x, y=y, psi=curr_psi, vx=omega * orbit_r, vy=0, dpsi=omega)
    satellite_1_shape = Point(0, 0).buffer(1)

    models = {
        P1: SpaceshipModel.default(x0_p1),
        P2: DynObstacleModel(
            satellite_1,
            shape=satellite_1_shape,
            og=ObstacleGeometry(m=5, Iz=50, e=0.5),
            op=DynObstacleParameters(vx_limits=(-100, 100), acc_limits=(-10, 10)),
        ),
    }

    cmds_p1 = DgSampledSequence[SpaceshipCommands](
        timestamps=[0, 1, 2, 3, 4, 5],
        values=[
            SpaceshipCommands(thrust=0, ddelta=deg2rad(20)),
            SpaceshipCommands(thrust=0.1, ddelta=deg2rad(20)),
            SpaceshipCommands(thrust=1, ddelta=deg2rad(-20)),
            SpaceshipCommands(thrust=0, ddelta=deg2rad(-20)),
            SpaceshipCommands(thrust=1, ddelta=deg2rad(-20)),
            SpaceshipCommands(thrust=2, ddelta=deg2rad(0)),
        ],
    )
    centripetal_acc = omega**2 * orbit_r
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


def test_spaceship_n_planet_sim():
    sim_context = get_planet_n_satellite_simcontext()
    # run simulation
    run_simulation(sim_context)
    report = generate_report(sim_context)
    # save report
    report_file = os.path.join(OUT_TESTS_DIR, "sim_spaceship.html")
    report.to_html(report_file)


if __name__ == "__main__":
    test_spaceship_n_planet_sim()
