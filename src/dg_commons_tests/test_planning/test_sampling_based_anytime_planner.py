import itertools
from decimal import Decimal
import os

import numpy as np
from geometry import SE2_from_xytheta
from matplotlib import pyplot as plt
from numpy import deg2rad
from reprep import Report, MIME_GIF
from shapely.geometry import Polygon
from commonroad.scenario.obstacle import StaticObstacle as CRStaticObstacle

from dg_commons import PlayerName, apply_SE2_to_shapely_geo, DgSampledSequence
from dg_commons.controllers.speed import SpeedBehavior, SpeedBehaviorParam
from dg_commons.maps import LaneCtrPoint, DgLanelet
from dg_commons.planning import PolygonGoal, Trajectory
from dg_commons.planning.sampling_algorithms.anytime_rrt_dubins import AnytimeRRTDubins
from dg_commons.planning.sampling_algorithms.cl_rrt_star import ClosedLoopRRTStar
from dg_commons.sim import SimParameters
from dg_commons.sim.agents import NPAgent
from dg_commons.sim.agents.plan_lane_follower import PlanLFAgent
from dg_commons.sim.agents.real_time_plan_lane_follower import RealTimePlanLFAgent
from dg_commons.sim.models import kmh2ms
from dg_commons.sim.models.obstacles_dyn import DynObstacleModel, DynObstacleState, DynObstacleCommands
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.scenarios import load_commonroad_scenario, DgScenario
from dg_commons.sim.simulator import SimContext, Simulator
from dg_commons.sim.simulator_animation import create_animation
from dg_commons.sim.simulator_visualisation import plot_trajectories
from dg_commons_tests import OUT_TESTS_DIR
from dg_commons.sim.models.obstacles import StaticObstacle, DynObstacleParameters, ObstacleGeometry

P1 = PlayerName("P1")

DObs1 = PlayerName("DObs1")

def _viz(trajectories, name=""):
    # viz
    fig = plt.figure(figsize=(10, 7), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect("equal")
    traj_lines, traj_points = plot_trajectories(ax=ax, trajectories=trajectories)
    # set the limits
    ax.set_xlim([50, 100])
    ax.set_ylim([-2, 6])
    # ax.autoscale(True, axis="both", tight=True)
    plt.gca().relim(visible_only=True)
    # ax.autoscale_view()
    # plt.draw()
    file_name = os.path.join(OUT_TESTS_DIR, f"{name}_test.png")
    plt.savefig(file_name)


def static_object_cr2dg(static_obstacle: CRStaticObstacle) -> StaticObstacle:
    position = static_obstacle.initial_state.position
    orientation = static_obstacle.initial_state.orientation
    poly1 = Polygon(static_obstacle.obstacle_shape.vertices)
    poly2 = apply_SE2_to_shapely_geo(poly1, SE2_from_xytheta((position[0], position[1], orientation)))
    return StaticObstacle(poly2)


def generate_report(sim_context: SimContext) -> Report:
    r = Report("EpisodeVisualisation")
    gif_viz = r.figure(cols=1)
    with gif_viz.data_file("Animation", MIME_GIF) as fn:
        create_animation(file_path=fn, sim_context=sim_context, figsize=(16, 8), dt=30, dpi=120, plot_limits="auto")
    return r


def get_simple_scenario() -> SimContext:
    scenario_name = "ZAM_Tutorial_Urban-3_2"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)
    static_obstacles = {id: static_object_cr2dg(static_obstacle) for id, static_obstacle in
                        enumerate(scenario.static_obstacles)}
    dobs_shape = Polygon([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]])
    x0_dobs1: DynObstacleState = DynObstacleState(x=90, y=3, psi=deg2rad(0), vx=-6, vy=0, dpsi=0)
    og_dobs1: ObstacleGeometry = ObstacleGeometry(m=1000, Iz=1000, e=0.2)
    op_dops1: DynObstacleParameters = DynObstacleParameters(vx_limits=(-10, 10), acc_limits=(-1, 1))
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    initial_position = planning_problem.initial_state.position
    initial_velocitiy = planning_problem.initial_state.velocity
    # initial_velocitiy = 0.0
    initial_position[0] = initial_position[0] - 5
    initial_orientation = planning_problem.initial_state.orientation
    x0_p1 = VehicleStateDyn(x=initial_position[0], y=initial_position[1], theta=initial_orientation,
                            vx=initial_velocitiy, delta=0)
    vp = VehicleParameters.default_car()
    vg = VehicleGeometry.default_car()
    goal_state = planning_problem.goal.state_list[0]
    goal_state = VehicleStateDyn(x=goal_state.position.center[0], y=goal_state.position.center[1],
                                 theta=goal_state.position.orientation,
                                 vx=0.0, delta=0)
    goal_poly = Polygon(vg.outline)
    goal_poly = apply_SE2_to_shapely_geo(goal_poly, SE2_from_xytheta((85, 0, goal_state.theta)))
    goal = PolygonGoal(goal_poly)
    dg_scenario_planner = DgScenario(scenario=scenario, static_obstacles=static_obstacles, use_road_boundaries=True)
    planner = AnytimeRRTDubins(player_name=P1, scenario=dg_scenario_planner,
                               planningProblem=planning_problem, initial_vehicle_state=x0_p1, goal=goal,
                               goal_state=goal_state, max_iter=2000, goal_sample_rate=70, expand_dis=10.0,
                               path_resolution=0.25, curvature=1.0,
                               search_until_max_iter=True, seed=5, expand_iter=20)

    goal = {
        P1: goal,
    }

    models = {
        P1: VehicleModelDyn.default_car(x0_p1),
        DObs1: DynObstacleModel(x0_dobs1, dobs_shape, og_dobs1, op_dops1),
    }
    dyn_obstacle_commands = DgSampledSequence[DynObstacleCommands](
        timestamps=[0],
        values=[DynObstacleCommands(acc_x=0, acc_y=0, acc_psi=0)],
    )

    speed_params = SpeedBehaviorParam(nominal_speed=kmh2ms(10))
    speed_behavior = SpeedBehavior()
    speed_behavior.params = speed_params
    players = {P1: RealTimePlanLFAgent(planner=planner, dt_plan=Decimal("1.0"), dt_expand_tree=Decimal("0.1"),
                                       return_extra=True, speed_behavior=speed_behavior),
               DObs1: NPAgent(dyn_obstacle_commands)}
    # players = {P1: NPAgent(p_cmds)}

    return SimContext(
        dg_scenario=DgScenario(scenario=scenario, static_obstacles=static_obstacles),
        models=models,
        players=players,
        missions=goal,
        param=SimParameters(dt=Decimal("0.01"), dt_commands=Decimal("0.01"), sim_time_after_collision=Decimal(1),
                            max_sim_time=Decimal(14)),
    )


def test_sampling_based_planner():
    sim_context = get_simple_scenario()
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    report = generate_report(sim_context)
    # save report
    report_file = os.path.join(OUT_TESTS_DIR, f"simple_sim.html")
    report.to_html(report_file)


if __name__ == '__main__':
    test_sampling_based_planner()
