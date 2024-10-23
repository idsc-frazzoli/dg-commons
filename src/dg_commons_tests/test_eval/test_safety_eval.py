import numpy as np
import matplotlib.pyplot as plt
import shapely.affinity
from geometry import SE2_from_xytheta
from dg_commons import apply_SE2_to_shapely_geo, X
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.eval.safety import _get_ttc, _get_dist


def plot_polygon(polygon: shapely.geometry.Polygon, ax: plt.Axes, color='black', alpha=1.0):
    x1, y1 = polygon.exterior.xy
    ax.plot(x1, y1, color=color, alpha=alpha)


def plot_vehicle_at_t(poly: shapely.geometry.Polygon, state: X, t: float, ax: plt.Axes, color='black', alpha=1.0):
    delta_x = state.vx * np.cos(state.psi) * t
    delta_y = state.vx * np.sin(state.psi) * t
    poly = shapely.affinity.translate(poly, xoff=delta_x, yoff=delta_y)
    plot_polygon(poly, ax, color=color, alpha=alpha)


def test_safety_eval():
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
    plot_polygon(poly1, ax, color='b', alpha=0.5)
    plot_polygon(poly2, ax, color='r', alpha=0.5)
    ax.axis('equal')

    # compute the nearest points
    dist, nearest_pts = _get_dist(state1, state2, model1, model2)
    pt1 = np.array([nearest_pts[0].x, nearest_pts[0].y])
    pt2 = np.array([nearest_pts[1].x, nearest_pts[1].y])
    pts = np.stack([pt1, pt2])
    plt.scatter(pts[:, 0], pts[:, 1], color='red')
    print('min dist between the vehicle is ', dist, "m")


    # simulate the movement, stop when the two vehicles collide
    ttc_sim, dtc1, dtc2 = _get_ttc(state1, state2, model1, model2)
    print("the simulated time to collision is ", ttc_sim, "s")
    print("dist to collision for vehicle1 is ", dtc1, "m")
    print("dist to collision for vehicle2 is ", dtc2, "m")
    # plot the vehicle poses when they collide(in case of no collision, plot the vehicle poses at t=3.0)
    ttc_sim = ttc_sim if not np.isinf(ttc_sim) else 3.0
    plot_vehicle_at_t(poly1, state1, ttc_sim, ax, color='b')
    plot_vehicle_at_t(poly2, state2, ttc_sim, ax, color='r')

    plt.show()


if __name__ == '__main__':
    test_safety_eval()
    