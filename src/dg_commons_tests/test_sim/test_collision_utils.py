import pytest
from shapely.geometry import Polygon, LineString
from shapely.geometry.base import BaseGeometry

from dg_commons.sim.collision_utils import compute_impact_geometry

# Create two intersecting rectangles
rect1 = Polygon([(0, 0), (0, 2), (2, 2), (2, 0)])
rect2 = Polygon([(1, 1), (1, 3), (3, 3), (3, 1)])

# Create a rectangle and a LinearString that intersects it
rect = Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])
line_intersecting = LineString([(2, -1), (2, 5)])

# Create a rectangle and a LinearString that's completely inside it
line_contained = LineString([(1, 1), (1, 3)])

# Create a rectangle and a LinearString that intersects it in two points
line_double_intersecting = LineString([(-1, 1), (2, 3), (5, 1)])

pairs_to_test: list[tuple[BaseGeometry, BaseGeometry]] = [
    (rect1, rect2),
    (rect, line_intersecting),
    (rect, line_contained),
    (rect, line_double_intersecting),
]


@pytest.mark.parametrize("a, b", pairs_to_test)
def test_compute_impact_geometry(a: BaseGeometry, b: BaseGeometry):
    norm, impact_p = compute_impact_geometry(a, b)
    print(norm, impact_p)
