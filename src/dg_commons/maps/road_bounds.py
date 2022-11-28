from typing import List, Tuple

from commonroad.scenario.scenario import Scenario
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


def build_road_boundary_obstacle(scenario: Scenario) -> Tuple[List[LineString], List[Polygon]]:
    """Returns a list of LineString of the scenario that are then used for collision checking.
    The boundaries are computed taking the external perimeter of the scenario and
    removing the entrance and exiting "gates" of the lanes.
    @:return: a tuple containing the road boundaries and the open gates. Both are represented as a lists of LineStrings
    """

    lanelets = scenario.lanelet_network.lanelets
    scenario_bounds: List[LineString] = []
    lane_polygons: List[Polygon] = []
    entrance_exit_gates = []
    for lanelet in lanelets:
        lane_polygons.append(lanelet.polygon.shapely_object.buffer(0.1))
        if len(lanelet.successor) == 0:
            pt1 = lanelet.right_vertices[-1]
            pt2 = lanelet.left_vertices[-1]
            entrance_exit_gates.append(LineString([pt1, pt2]).buffer(0.5))
        if len(lanelet.predecessor) == 0:
            pt1 = lanelet.right_vertices[0]
            pt2 = lanelet.left_vertices[0]
            entrance_exit_gates.append(LineString([pt1, pt2]).buffer(0.5))

    overall_poly = unary_union(lane_polygons)
    for interior in overall_poly.interiors:
        scenario_bounds.append(interior)

    ext_bounds = overall_poly.exterior
    for eeg in entrance_exit_gates:
        ext_bounds = ext_bounds.difference(eeg)
    scenario_bounds += [g for g in ext_bounds.geoms]
    return scenario_bounds, entrance_exit_gates
