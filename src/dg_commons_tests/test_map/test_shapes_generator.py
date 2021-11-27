import os.path

from geometry import SE2_from_xytheta
from matplotlib import pyplot as plt
from numpy import deg2rad
from shapely.geometry import LinearRing, Polygon

from dg_commons import apply_SE2_to_shapely_geo
from dg_commons.maps.obstacles_shapes import polygon_star, generate_starshaped_polygon
from dg_commons.maps.shapely_viz import ShapelyViz
from dg_commons_tests import OUT_TESTS_DIR


def test_generate_shapes():
    poly = polygon_star(15, 8, 5, 1.5)
    transform = SE2_from_xytheta((3, 3, deg2rad(0)))
    viz = ShapelyViz()
    arena_size = 100
    boundary = LinearRing([[0, 0], [arena_size, 0], [arena_size, arena_size], [0, arena_size]])
    viz.add_shape(boundary, color="r")
    gen_poly = Polygon(generate_starshaped_polygon(50, 50, 10, 0.5, 0.5, 10))
    viz.add_shape(gen_poly, facecolor="gold", edgecolor="r")
    for i in range(10):
        poly = apply_SE2_to_shapely_geo(poly, transform)
        viz.add_shape(poly, facecolor="gold", edgecolor="r")
    viz.ax.autoscale()
    viz.ax.set_aspect("equal")
    f = os.path.join(OUT_TESTS_DIR, "test_generate_shapes.png")
    plt.savefig(f, dpi=300)
