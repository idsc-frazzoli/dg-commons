from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal as D
from typing import Sequence

from numpy import deg2rad
from reprep import Report, MIME_GIF
from shapely import Point
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from dg_commons import PlayerName, DgSampledSequence, fd, X
from dg_commons.sim import SimParameters, SimTime
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.goals import PolygonGoal, PlanningGoal
from dg_commons.sim.log_visualisation import plot_player_log
from dg_commons.sim.models.spacecraft import SpacecraftState, SpacecraftModel, SpacecraftCommands
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons.sim.simulator import SimContext, Simulator
from dg_commons.sim.simulator_animation import create_animation
from dg_commons.sim.utils import run_simulation
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

    x0_p1 = VehicleStateDyn(x=0, y=0, psi=deg2rad(60), vx=5, delta=0)
    x0_p2 = VehicleStateDyn(x=24, y=6, psi=deg2rad(150), vx=6, delta=0)
    x0_p3 = SpacecraftState(x=10, y=5, psi=deg2rad(0), vx=0, vy=0, dpsi=0)
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
            SpacecraftCommands(acc_left=-1, acc_right=6),
            SpacecraftCommands(acc_left=-1, acc_right=6),
            SpacecraftCommands(acc_left=1, acc_right=6),
            SpacecraftCommands(acc_left=6, acc_right=-2),
            SpacecraftCommands(acc_left=6, acc_right=-2),
        ],
    )
    players = {P1: NPAgent(moving_vehicle), P2: NPAgent(static_vehicle), P3: NPAgent(spacecraft_dynamic)}
    p2_goal_poly = Polygon([[0, 13], [5, 13], [5, 20], [0, 20], [0, 13]])
    missions = {P2: PolygonGoal(p2_goal_poly)}
    return SimContext(
        dg_scenario=DgScenario(scenario=scenario, use_road_boundaries=True),
        models=models,
        players=players,
        param=SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(4), max_sim_time=D(6)),
        missions=fd(missions),
    )


def generate_report(sim_context: SimContext) -> Report:
    r = Report("EpisodeVisualisation")
    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Animation", MIME_GIF) as fn:
        create_animation(file_path=fn, sim_context=sim_context, figsize=(16, 8), dt=50, dpi=120, plot_limits="auto")

    # state/commands plots
    for pn in sim_context.log.keys():
        with r.subsection(f"Player-{pn}-log") as sub:
            with sub.plot(f"{pn}-log", figsize=(20, 15)) as pylab:
                plot_player_log(log=sim_context.log[pn], fig=pylab.gcf())

    return r


def test_simple_simulation():
    sim_context = get_simple_scenario()
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    report = generate_report(sim_context)
    # save report
    report_file = os.path.join(OUT_TESTS_DIR, f"sim_simple.html")
    report.to_html(report_file)


@dataclass(frozen=True)
class TVgoal(PlanningGoal):
    x0: Sequence[float, float]
    v0: Sequence[float, float]
    is_static: bool = False

    def is_fulfilled(self, state: X, at: SimTime | float = 0) -> bool:
        p_pos = Point(state.x, state.y)
        return self.get_plottable_geometry(at).contains(p_pos)

    def get_plottable_geometry(self, at: SimTime | float = 0) -> BaseGeometry:
        x1, y1 = self._get_position_at(at)
        return Point(x1, y1).buffer(0.5)

    def _get_position_at(self, at: SimTime | float = 0) -> Sequence[float, float]:
        x1 = self.x0[0] + self.v0[0] * float(at)
        y1 = self.x0[1] + self.v0[1] * float(at)
        return x1, y1


def test_simulation_tvgoal():
    # Testing time-varying goals
    sim_context = get_simple_scenario()
    sim_params2 = SimParameters(dt=D("0.01"), dt_commands=D("0.1"), sim_time_after_collision=D(1), max_sim_time=D(3))
    sim_context.param = sim_params2
    tv_goal1 = TVgoal(x0=(15, 5), v0=(3, 3))
    tv_goal2 = TVgoal(x0=(3, 3), v0=(-3, 4))
    missions = {P1: tv_goal1, P2: tv_goal2}
    sim_context.missions = fd(missions)
    sim_context = run_simulation(sim_context)
    report = generate_report(sim_context)
    # save report
    report_file = os.path.join(OUT_TESTS_DIR, f"sim_tvgoals.html")
    report.to_html(report_file)
