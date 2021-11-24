from typing import List

from commonroad.scenario.scenario import Scenario
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


def build_road_boundary_obstacle(scenario: Scenario) -> List[LineString]:
    """Returns a list of LineString of the scenario that are then used for collision checking"""
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

    for ext_bound in ext_bounds:
        scenario_bounds.append(ext_bound)
    return scenario_bounds
