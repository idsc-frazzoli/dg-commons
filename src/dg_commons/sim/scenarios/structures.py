from dataclasses import dataclass, field
from typing import Optional, Mapping

from commonroad.scenario.scenario import Scenario
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from dg_commons.maps.road_bounds import build_road_boundary_obstacle
from dg_commons.sim.models.obstacles import StaticObstacle


@dataclass
class DgScenario:
    scenario: Optional[Scenario] = None
    """A commonroad scenario"""
    static_obstacles: Mapping[int, StaticObstacle] = field(default_factory=dict)
    """A sequence of Shapely geometries"""
    use_road_boundaries: bool = False
    """If True the external boundaries of the road are forced to be obstacles """
    strtree_obstacles: STRtree = field(init=False)

    def __post_init__(self):
        if self.scenario:
            assert isinstance(self.scenario, Scenario), self.scenario
        for idx, sobstacle in self.static_obstacles.items():
            assert issubclass(type(sobstacle), StaticObstacle), sobstacle
        if self.use_road_boundaries:
            lanelet_bounds = build_road_boundary_obstacle(self.scenario)
            self.static_obstacles += lanelet_bounds
        obs_shapes = [sobstacle.shape for sobstacle in self.static_obstacles.values()]
        obs_idx = [idx for idx in self.static_obstacles.keys()]
        self.strtree_obstacles = STRtree(obs_shapes, obs_idx, node_capacity=3)

    @property
    def lanelet_network(self):
        """Just for ease of use to avoid dg_scenario.scenario..."""
        return self.scenario.lanelet_network
