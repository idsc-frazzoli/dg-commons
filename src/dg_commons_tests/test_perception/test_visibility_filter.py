from math import pi

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

from commonroad_dc.pycrcc import Polygon as CommonRoadPolygon

from dg_commons import SE2Transform
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons.perception.sensor import VisRangeSensor


def sPolygon2crPolygon(shapely_polygon: Polygon) -> CommonRoadPolygon:
    vertices = np.array(list(zip(*shapely_polygon.exterior.xy)))
    # magic numbers inferred from here
    # https://gitlab.lrz.de/tum-cps/commonroad-drivability-checker/-/blob/master/cpp/collision/include/collision/plugins/triangulation/triangulate.h
    return CommonRoadPolygon(vertices, 0.125, 2)


def test_visibility_filter():
    vis = ShapelyViz()
    sensor_pose: SE2Transform = SE2Transform(p=[-2, -1], theta=2.3)
    lidar_fov = Point(sensor_pose.p).buffer(20)
    vis.add_shape(lidar_fov, color="cyan", alpha=0.5)
    obs1 = Polygon([(10, 10), (10, 15), (15, 15), (15, 10)])
    obs2 = Polygon([(-3, -3), (-3, -7), (-8, -10), (-12, -9), (-12, -3)])
    obs3 = Polygon([(-3, 3), (-3, 7), (-8, 10), (-12, 9), (-12, 3)])

    vis.add_shape(obs1, color="black")
    vis.add_shape(obs2, color="black")
    vis.add_shape(obs3, color="black")
    pov_x, pov_y = lidar_fov.centroid.xy

    obs = [sPolygon2crPolygon(o) for o in [obs1, obs2, obs3]]
    lidar2d = VisRangeSensor(field_of_view=pi / 1.4)
    lidar2d.pose = sensor_pose
    sensor_view: Polygon = lidar2d.fov_as_polygon(obs)
    vis.add_shape(sensor_view, color="gray", alpha=0.5)

    # fov_linestring = LineString(lidar_fov.exterior.coords)
    # for edge in obs1.exterior.coords:
    #     projection = fov_linestring.project(Point(edge))

    plt.plot(pov_x, pov_y, "r+")
    plt.gca().set_aspect("equal")
    plt.plot()
    plt.show()


if __name__ == "__main__":
    test_visibility_filter()
