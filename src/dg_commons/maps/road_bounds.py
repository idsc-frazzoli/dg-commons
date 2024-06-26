from commonroad.scenario.scenario import Scenario
from shapely import MultiPolygon
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


def build_road_boundary_obstacle(scenario: Scenario, buffer: float = 0.1) -> tuple[list[LineString], list[Polygon]]:
    """Returns a list of LineString of the scenario that are then used for collision checking.
    The boundaries are computed taking the external perimeter of the scenario and
    removing the entrance and exiting "gates" of the lanes.
    :param scenario: the scenario to build the boundaries for
    :param buffer: the buffer to apply to the lanelets
    @:return: a tuple containing the road boundaries and the open gates. Both are represented as a lists of LineStrings
    """

    lanelets = scenario.lanelet_network.lanelets
    scenario_bounds: list[LineString] = []
    lane_polygons: list[Polygon] = []
    entrance_exit_gates = []
    for lanelet in lanelets:
        lane_polygons.append(lanelet.polygon.shapely_object.buffer(buffer))
        if len(lanelet.successor) == 0:
            pt1 = lanelet.right_vertices[-1]
            pt2 = lanelet.left_vertices[-1]
            entrance_exit_gates.append(LineString([pt1, pt2]).buffer(buffer * 2))
        if len(lanelet.predecessor) == 0:
            pt1 = lanelet.right_vertices[0]
            pt2 = lanelet.left_vertices[0]
            entrance_exit_gates.append(LineString([pt1, pt2]).buffer(buffer * 2))

    overall_poly = unary_union(lane_polygons)
    # if the overall_poly is a Polygon, convert it to a MultiPolygon
    if isinstance(overall_poly, Polygon):
        overall_poly = MultiPolygon(
            [
                overall_poly,
            ]
        )

    for geo in overall_poly.geoms:
        for interior in geo.interiors:
            scenario_bounds.append(interior)

        ext_bounds = geo.exterior
        for eeg in entrance_exit_gates:
            ext_bounds = ext_bounds.difference(eeg)
        scenario_bounds += [g for g in ext_bounds.geoms]
    return scenario_bounds, entrance_exit_gates
