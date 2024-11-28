import pickle
import numpy as np
import matplotlib.pyplot as plt
import shapely.affinity
from geometry import SE2_from_xytheta
from dg_commons import apply_SE2_to_shapely_geo, X, PlayerName
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.eval.safety import _get_ttc, _get_dist, get_min_dist, get_min_ttc_max_drac, has_collision
from dg_commons_tests import REPO_DIR


def plot_polygon(polygon: shapely.geometry.Polygon, ax: plt.Axes, color="black", alpha=1.0):
    x1, y1 = polygon.exterior.xy
    ax.plot(x1, y1, color=color, alpha=alpha)


def plot_vehicle_at_t(poly: shapely.geometry.Polygon, state: X, t: float, ax: plt.Axes, color="black", alpha=1.0):
    delta_x = state.vx * np.cos(state.psi) * t
    delta_y = state.vx * np.sin(state.psi) * t
    poly = shapely.affinity.translate(poly, xoff=delta_x, yoff=delta_y)
    plot_polygon(poly, ax, color=color, alpha=alpha)


def test_dist_ttc_drac(debug: bool = False):
    # create shapely geometry from vehicle pose
    state1 = VehicleStateDyn(x=0, y=0.0, psi=-0.72, vx=8.0, delta=0)
    state2 = VehicleStateDyn(x=8.0, y=0.0, psi=-1.5, vx=3.3, delta=0)
    model1 = VehicleModelDyn.default_car(state1)
    model2 = VehicleModelDyn.default_car(state2)
    pose1 = SE2_from_xytheta([state1.x, state1.y, state1.psi])
    poly1 = apply_SE2_to_shapely_geo(model1.vg.outline_as_polygon, pose1)
    pose2 = SE2_from_xytheta([state2.x, state2.y, state2.psi])
    poly2 = apply_SE2_to_shapely_geo(model2.vg.outline_as_polygon, pose2)

    # compute the nearest points
    dist, nearest_pts = _get_dist(state1, state2, model1, model2)
    pt1 = np.array([nearest_pts[0].x, nearest_pts[0].y])
    pt2 = np.array([nearest_pts[1].x, nearest_pts[1].y])
    pts = np.stack([pt1, pt2])

    # simulate the movement, stop when the two vehicles collide
    ttc_sim, dtc1, dtc2 = _get_ttc(state1, state2, model1, model2)
    
    min_dist_expected = 5.17
    ttc_sim_expected = 1.10
    dtc1_expected = 8.80
    dtc2_expected = 3.63
    eps = 1e-2
    assert min_dist_expected - eps <= dist <= min_dist_expected + eps
    assert ttc_sim_expected - eps <= ttc_sim <= ttc_sim_expected + eps
    assert dtc1_expected - eps <= dtc1 <= dtc1_expected + eps
    assert dtc2_expected - eps <= dtc2 <= dtc2_expected + eps
    
    # plot the vehicles
    if debug:
        plt.figure()
        ax = plt.gca()
        plot_polygon(poly1, ax, color="b", alpha=0.5)
        plot_polygon(poly2, ax, color="r", alpha=0.5)

        plt.scatter(pts[:, 0], pts[:, 1], color="red")
        
        # plot the vehicle poses when they collide(in case of no collision, plot the vehicle poses at t=3.0)
        ttc_sim = ttc_sim if not np.isinf(ttc_sim) else 3.0
        plot_vehicle_at_t(poly1, state1, ttc_sim, ax, color="b")
        plot_vehicle_at_t(poly2, state2, ttc_sim, ax, color="r")
        ax.axis("equal")
        plt.show()


def test_safety_eval():
    logs = REPO_DIR / "src/dg_commons_tests/test_eval/logs"
    file = open(logs / "collision_reports.pickle", 'rb')
    collision_reports = pickle.load(file)
    file.close()
    file = open(logs / "log.pickle", 'rb')
    log = pickle.load(file)
    file.close()
    file = open(logs / "missions.pickle", 'rb')
    missions = pickle.load(file)
    file.close()
    file = open(logs / "models.pickle", 'rb')
    models = pickle.load(file)
    file.close()

    ego_name = PlayerName("Ego")

    t_range = (0.2, 3.0)
    min_dist, min_dist_agent, min_dist_t = get_min_dist(log, models, missions, ego_name, t_range)
    min_ttc, min_ttc_agent, min_ttc_t, max_drac, max_drac_agent, max_drac_t = get_min_ttc_max_drac(log, models,
                                                                                                    missions,
                                                                                                   ego_name, t_range)
    # print(
    #     f"Minimum distance is {min_dist:.2f}m from agent {min_dist_agent} at time step {min_dist_t:.2f}s.")
    # if not np.isinf(min_ttc):
    #     print(f"Minimum time-to-collision is {min_ttc:.2f}s from agent {min_ttc_agent} at time step {min_ttc_t:.2f}s.")
    #     print(f"Maximum decelleration rate to avoid collision is {max_drac:.2f}m/s2 from agent {max_drac_agent} at time step {max_drac_t:.2f}s.")
    has_collided = has_collision(collision_reports)

    min_dist_expected = 0.47
    min_dist_agent_expected = "P18"
    min_dist_t_expected = 0.35
    min_ttc_expected = 1.10
    min_ttc_agent_expected = "P18"
    min_ttc_t_expected = 0.25
    max_drac_expected = 6.79
    max_drac_agent_expected = "P18"
    max_drac_t_expected = 0.25
    eps = 1e-2
    assert min_dist_expected - eps <= min_dist <= min_dist_expected + eps
    assert min_dist_agent == min_dist_agent_expected
    assert min_dist_t_expected - eps <= min_dist_t <= min_dist_t_expected + eps
    assert min_ttc_expected - eps <= min_ttc <= min_ttc_expected + eps
    assert min_ttc_agent == min_ttc_agent_expected
    assert min_ttc_t_expected - eps <= min_ttc_t <= min_ttc_t_expected + eps
    assert max_drac_expected - eps <= max_drac <= max_drac_expected + eps
    assert max_drac_agent == max_drac_agent_expected
    assert max_drac_t_expected - eps <= max_drac_t <= max_drac_t_expected + eps
    assert not has_collided

    file = open(logs / "collision_reports_collided.pickle", 'rb')
    collision_reports = pickle.load(file)
    file.close()
    has_collided = has_collision(collision_reports)
    assert has_collided
