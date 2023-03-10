from commonroad.visualization.draw_params import MPDrawParams
from commonroad.visualization.mp_renderer import MPRenderer
from matplotlib import pyplot as plt
from shapely.affinity import affine_transform
from shapely.geometry import Polygon

from dg_commons.maps.road_bounds import build_road_boundary_obstacle
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.scenarios import load_commonroad_scenario
from dg_commons.sim.scenarios.structures import DgScenario
from dg_commons_tests import OUT_TESTS_DIR


def test_build_road_bounds():
    scenario_name = "USA_Lanker-1_1_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)
    lane_boundaries, gates = build_road_boundary_obstacle(scenario)
    rnd = MPRenderer(figsize=(20, 20))

    draw_params = MPDrawParams()
    draw_params.lanelet_network.traffic_light.draw_traffic_lights = True
    scenario.draw(rnd, draw_params=draw_params)
    rnd.render()

    for l in lane_boundaries:
        xy = l.coords.xy
        rnd.ax.axes.plot(xy[0], xy[1], color="orange", zorder=100)
    for l in gates:
        xy = l.boundary.coords.xy
        rnd.ax.axes.plot(xy[0], xy[1], color="green", zorder=100)

    rnd.ax.set_facecolor("k")
    plt.show()


def test_road_bounds_dgscenario():
    poly1 = Polygon([(0, 0), (0, 5), (5, 5), (5, 0)])
    poly2 = affine_transform(poly1, [1, 0, 0, 1, 10, 5])
    poly3 = affine_transform(poly1, [1, 0, 0, 1, -10, -5])

    scenario_name = "USA_Lanker-1_1_T-1"
    scenario, planning_problem_set = load_commonroad_scenario(scenario_name)
    polys = list(map(StaticObstacle, [poly1, poly2, poly3]))
    dgscenario = DgScenario(scenario, static_obstacles=polys, use_road_boundaries=True)
    rnd = MPRenderer(figsize=(20, 20))
    draw_params = MPDrawParams()
    draw_params.lanelet_network.traffic_light.draw_traffic_lights = True
    scenario.draw(rnd, draw_params=draw_params)
    rnd.render()
    for do in scenario.dynamic_obstacles:
        do.draw(rnd)
        do_shapely = do.occupancy_at_time(0).shape.shapely_object
        coll_indexes = dgscenario.strtree_obstacles.query(do_shapely, predicate="intersects")
        for idx in coll_indexes:
            col_obj = dgscenario.static_obstacles[idx].shape
            col_obj_peri = col_obj.exterior if isinstance(col_obj, Polygon) else col_obj
            rnd.ax.plot(col_obj_peri.xy[0], col_obj_peri.xy[1], color="purple", zorder=100)
            rnd.ax.plot(do_shapely.exterior.xy[0], do_shapely.exterior.xy[1], color="purple", zorder=100)

    for so in polys:
        shapely_obj = so.shape
        xy = shapely_obj.exterior.xy if isinstance(shapely_obj, Polygon) else shapely_obj.xy
        rnd.ax.axes.plot(xy[0], xy[1], color="orange", zorder=100)
    rnd.ax.set_facecolor("k")

    plt.savefig(OUT_TESTS_DIR + "/road_bounds_dgscenario.png")
    # plt.show()
