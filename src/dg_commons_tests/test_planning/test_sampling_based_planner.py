import itertools
from decimal import Decimal
import os

import numpy as np
from geometry import SE2_from_xytheta
from matplotlib import pyplot as plt
from reprep import Report, MIME_GIF
from shapely.geometry import Polygon
from commonroad.scenario.obstacle import StaticObstacle as CRStaticObstacle

from dg_commons import PlayerName, apply_SE2_to_shapely_geo
from dg_commons.controllers.speed import SpeedBehavior, SpeedBehaviorParam
from dg_commons.maps import LaneCtrPoint, DgLanelet
from dg_commons.planning import PolygonGoal, Trajectory
from dg_commons.planning.sampling_algorithms.cl_rrt_star import ClosedLoopRRTStar
from dg_commons.planning.sampling_algorithms.rrt_dubins import RRTDubins
from dg_commons.planning.sampling_algorithms.rrt_star_dubins import RRTStarDubins
from dg_commons.planning.sampling_algorithms.rrt_star_reeds_shepp import RRTStarReedsShepp
from dg_commons.sim import SimParameters
from dg_commons.sim.agents.plan_lane_follower import PlanLFAgent
from dg_commons.sim.models import kmh2ms
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.scenarios import load_commonroad_scenario, DgScenario
from dg_commons.sim.simulator import SimContext, Simulator
from dg_commons.sim.simulator_animation import create_animation
from dg_commons.sim.simulator_visualisation import plot_trajectories
from dg_commons_tests import OUT_TESTS_DIR
from dg_commons.sim.models.obstacles import StaticObstacle

P1 = PlayerName("P1")


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
    planner = RRTStarDubins(player_name=P1, scenario=dg_scenario_planner,
                            initial_vehicle_state=x0_p1, goal=goal,
                            goal_state=goal_state, max_iter=2000, goal_sample_rate=10, expand_dis=10.0,
                            path_resolution=0.25, path_length=3.0, curvature=1.0, goal_yaw_th=np.deg2rad(30.0), goal_xy_th=2.0,
                            connect_circle_dist=30.0, search_until_max_iter=True, seed=0)
    # planner = RRTDubins(player_name=P1, scenario=dg_scenario_planner,
    #                     initial_vehicle_state=x0_p1, goal=goal,
    #                     goal_state=goal_state, max_iter=2000, goal_sample_rate=10, expand_dis=10.0,
    #                     path_resolution=0.25, path_length=3.0, curvature=1.0, goal_yaw_th=np.deg2rad(30.0), goal_xy_th=2.0,
    #                     search_until_max_iter=True, seed=0)
    # planner = RRTStarReedsShepp(player_name=P1, scenario=dg_scenario_planner,
    #                             initial_vehicle_state=x0_p1, goal=goal,
    #                             goal_state=goal_state, max_iter=1000, goal_sample_rate=10, expand_dis=10.0,
    #                             path_resolution=0.25, path_length=3.0, curvature=1.0, goal_yaw_th=np.deg2rad(30.0),
    #                             goal_xy_th=2.0, connect_circle_dist=30.0, search_until_max_iter=True, seed=0)
    # planner = ClosedLoopRRTStar(scenario=dg_scenario_planner,
    #                             initial_vehicle_state=x0_p1, goal=goal,
    #                             goal_state=goal_state, max_iter=1000, goal_sample_rate=10, expand_dis=10.0,
    #                             path_resolution=0.25, curvature=1.0, goal_yaw_th=np.deg2rad(30.0), goal_xy_th=2.0,
    #                             connect_circle_dist=30.0, search_until_max_iter=False, seed=0,
    #                             target_speed=initial_velocitiy,
    #                             yaw_th=np.deg2rad(3.0), xy_th=0.5, invalid_travel_ratio=5.0)
    goal = {
        P1: goal,
    }

    models = {
        P1: VehicleModelDyn.default_car(x0_p1),
    }
    path = planner.planning()
    ctr_points = []
    timestamp = 0.0
    v_state_list = []
    timestamp_list = []
    for p in path:
        ctr_points.append(LaneCtrPoint(p, r=0.01))
        v_state = VehicleState(x=p.p[0], y=p.p[1], theta=p.theta, vx=0, delta=0)
        timestamp_list.append(timestamp)
        v_state_list.append(v_state)
        timestamp += 1

    trajectory = Trajectory(timestamps=timestamp_list, values=v_state_list)
    _viz([trajectory], "opt_rrt_star_dubins")
    lane = DgLanelet(ctr_points)
    all_path_list = [node.path for node in planner.node_list if node.path]
    all_queries_list = []
    for p in all_path_list:
        vs_list = []
        ts_list = []
        ts = 0.0
        for s in p:
            vs = VehicleState(x=s.p[0], y=s.p[1], theta=s.theta, vx=0, delta=0)
            ts_list.append(ts)
            vs_list.append(vs)
            ts += 1
        tra = Trajectory(timestamps=ts_list, values=vs_list)
        all_queries_list.append(tra)
    _viz(all_queries_list, "all_rrt_star_dubins")
    speed_params = SpeedBehaviorParam(nominal_speed=kmh2ms(10))
    speed_behavior = SpeedBehavior()
    speed_behavior.params = speed_params
    players = {P1: PlanLFAgent(trajectories=[trajectory], all_queries_list=all_queries_list,
                               lane=lane, return_extra=True, speed_behavior=speed_behavior)}
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
