import itertools
from decimal import Decimal
import os

from geometry import SE2_from_xytheta
from reprep import Report, MIME_GIF
from shapely.geometry import Polygon

from dg_commons import PlayerName, apply_SE2_to_shapely_geo
from dg_commons.dynamics import BicycleDynamics
from dg_commons.maps import LaneCtrPoint, DgLanelet
from dg_commons.planning import MPGParam, MotionPrimitivesGenerator, PolygonGoal
from dg_commons.planning.search_algorithms.best_first_search import GreedyBestFirstSearch
from dg_commons.sim import SimParameters
from dg_commons.sim.agents.plan_lane_follower import PlanLFAgent
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons.sim.scenarios import load_commonroad_scenario, DgScenario
from dg_commons.sim.simulator import SimContext, Simulator
from dg_commons.sim.simulator_animation import create_animation
from dg_commons_tests import OUT_TESTS_DIR
from dg_commons_tests.test_planning.test_planning import static_object_cr2dg

P1 = (
    PlayerName("P1"),
)

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
    goal = {
        P1: goal,
    }

    models = {
        P1: VehicleModelDyn.default_car(x0_p1),
    }

    params = MPGParam(dt=Decimal(".1"), n_steps=10, velocity=(0, 10, 5), steering=(-vp.delta_max, vp.delta_max, 5))
    vehicle = BicycleDynamics(vg=vg, vp=vp)
    mpg = MotionPrimitivesGenerator(param=params, vehicle_dynamics=vehicle.successor_ivp, vehicle_param=vp)
    x0 = VehicleState(x=initial_position[0], y=initial_position[1], theta=initial_orientation,
                      vx=initial_velocitiy, delta=0)
    planner = GreedyBestFirstSearch(DgScenario(scenario=scenario, static_obstacles=static_obstacles, use_road_boundaries=True), planning_problem,
                          mpg, x0, goal[P1])
    path, trajectory = planner.execute_search()
    all_queries_list = [f_elements[2].list_trajectories for f_elements in planner.frontier.list_elements]
    all_queries_list = list(itertools.chain(*all_queries_list))
    ctr_points = []
    for p in path:
        for s in p:
            ctr_points.append(LaneCtrPoint(s, r=0.01))
    lane = DgLanelet(ctr_points)
    players = {P1: PlanLFAgent(trajectories=trajectory, all_queries_list=all_queries_list, lane=lane, return_extra=True)}

    return SimContext(
        dg_scenario=DgScenario(scenario=scenario, static_obstacles=static_obstacles),
        models=models,
        players=players,
        missions=goal,
        param=SimParameters(dt=Decimal("0.01"), dt_commands=Decimal("0.01"), sim_time_after_collision=Decimal(1),
                            max_sim_time=Decimal(14)),
    )


def test_search_based_planner():
    sim_context = get_simple_scenario()
    sim = Simulator()
    # run simulations
    sim.run(sim_context)
    report = generate_report(sim_context)
    # save report
    report_file = os.path.join(OUT_TESTS_DIR, f"simple_sim.html")
    report.to_html(report_file)


if __name__ == '__main__':
    test_search_based_planner()
