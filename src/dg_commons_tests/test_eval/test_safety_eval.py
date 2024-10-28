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


def test_dist_ttc_drac():
    # create shapely geometry from vehicle pose
    # state1 = VehicleStateDyn(x=21.84, y=-2.2, psi=-0.72, vx=10.29, delta=0)
    # state2 = VehicleStateDyn(x=32.11, y=-6.96, psi=-0.71, vx=3.3, delta=0)
    state1 = VehicleStateDyn(x=0, y=0.0, psi=-0.72, vx=8.0, delta=0)
    state2 = VehicleStateDyn(x=8.0, y=0.0, psi=-1.5, vx=3.3, delta=0)
    model1 = VehicleModelDyn.default_car(state1)
    model2 = VehicleModelDyn.default_car(state2)
    pose1 = SE2_from_xytheta([state1.x, state1.y, state1.psi])
    poly1 = apply_SE2_to_shapely_geo(model1.vg.outline_as_polygon, pose1)
    pose2 = SE2_from_xytheta([state2.x, state2.y, state2.psi])
    poly2 = apply_SE2_to_shapely_geo(model2.vg.outline_as_polygon, pose2)
    # plot the vehicles
    plt.figure()
    ax = plt.gca()
    plot_polygon(poly1, ax, color="b", alpha=0.5)
    plot_polygon(poly2, ax, color="r", alpha=0.5)
    ax.axis("equal")

    # compute the nearest points
    dist, nearest_pts = _get_dist(state1, state2, model1, model2)
    pt1 = np.array([nearest_pts[0].x, nearest_pts[0].y])
    pt2 = np.array([nearest_pts[1].x, nearest_pts[1].y])
    pts = np.stack([pt1, pt2])
    plt.scatter(pts[:, 0], pts[:, 1], color="red")
    print("min dist between the vehicle is ", dist, "m")

    # simulate the movement, stop when the two vehicles collide
    ttc_sim, dtc1, dtc2 = _get_ttc(state1, state2, model1, model2)
    print("the simulated time to collision is ", ttc_sim, "s")
    print("dist to collision for vehicle1 is ", dtc1, "m")
    print("dist to collision for vehicle2 is ", dtc2, "m")
    # plot the vehicle poses when they collide(in case of no collision, plot the vehicle poses at t=3.0)
    ttc_sim = ttc_sim if not np.isinf(ttc_sim) else 3.0
    plot_vehicle_at_t(poly1, state1, ttc_sim, ax, color="b")
    plot_vehicle_at_t(poly2, state2, ttc_sim, ax, color="r")
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
    print(
        "Minimum distance is %.2f" % min_dist + "m from agent " + str(min_dist_agent) + " at time step %.2f" % float(
            min_dist_t) + "s.")
    if not np.isinf(min_ttc):
        print("Minimum time-to-collision is %.2f" % min_ttc + "s from agent " + str(
            min_ttc_agent) + " at time step %.2f " % float(min_ttc_t) + "s.")
        print("Maximum decelleration rate to avoid collision is %.2f" % max_drac + "m/s2 from agent " + str(
            max_drac_agent) + " at time step %.2f " % float(max_drac_t) + "s.")
    has_collided = has_collision(collision_reports)
    print("Has collided: ", has_collided)
