from dataclasses import dataclass, field
from typing import Sequence

from commonroad.scenario.scenario import Scenario
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from dg_commons.maps.road_bounds import build_road_boundary_obstacle


@dataclass
class DgScenario:
    scenario: Scenario
    """A commonroad scenario"""
    static_obstacles: Sequence[BaseGeometry] = field(default_factory=list)
    """A sequence of Shapely geometries"""
    use_road_boundaries: bool = False
    """If True the external boundaries of the road are forced to be obstacles """
    strtree_obstacles: STRtree = field(init=False)

    def __post_init__(self):
        assert isinstance(self.scenario, Scenario), self.scenario
        for sobstacle in self.static_obstacles:
            assert issubclass(type(sobstacle), BaseGeometry), sobstacle
        if self.use_road_boundaries:
            lanelet_bounds = build_road_boundary_obstacle(self.scenario)
            self.static_obstacles += lanelet_bounds
        self.strtree_obstacles = STRtree(self.static_obstacles)

    @property
    def lanelet_network(self):
        """Just for ease of use to avoid dg_scenario.scenario..."""
        return self.scenario.lanelet_network
