from dataclasses import dataclass, field
from typing import Optional, Iterable, Sequence

from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import Scenario
from shapely.strtree import STRtree

from dg_commons import logger
from dg_commons.maps.road_bounds import build_road_boundary_obstacle
from dg_commons.sim.models.obstacles import StaticObstacle

__all__ = ["DgScenario"]


@dataclass(unsafe_hash=True)
class DgScenario:
    """Mainly a thin wrapper around CommonRoad scenarios. Yet it can work also as an empty world."""

    scenario: Optional[Scenario] = None
    """A CommonRoad scenario"""
    static_obstacles: Sequence[StaticObstacle] = None
    """Static obstacles of the scenario and/or extra additional ones"""
    use_road_boundaries: bool = False
    """If True the external boundaries of the road are forced to be obstacles """
    road_boundaries_buffer: float = 0.1
    """Buffer to be added to the lanelets when building the road boundaries"""
    strtree_obstacles: STRtree = field(init=False)
    """Store the obstacles in a spatial index for fast collision detection"""

    def __post_init__(self):
        if self.scenario is not None:
            assert isinstance(self.scenario, Scenario), self.scenario
        static_obstacles = list(self.static_obstacles) if self.static_obstacles is not None else []
        for sobstacle in static_obstacles:
            assert issubclass(type(sobstacle), StaticObstacle), sobstacle
        # add lane boundaries as obstacles after the static obstacles (since we assign random ids)
        if self.use_road_boundaries and self.scenario is not None:
            assert self.road_boundaries_buffer >= 0, self.road_boundaries_buffer
            lanelet_bounds, _ = build_road_boundary_obstacle(self.scenario, buffer=self.road_boundaries_buffer)
            for lanelet_bound in lanelet_bounds:
                static_obstacles.append(StaticObstacle(lanelet_bound))
        elif self.use_road_boundaries and self.scenario is None:
            logger.warn("Road boundaries requested but no scenario provided, ignoring...")
        obs_shapes = [sobstacle.shape for sobstacle in static_obstacles]
        self.strtree_obstacles = STRtree(obs_shapes, node_capacity=3)
        self.static_obstacles = tuple(static_obstacles)

    @property
    def lanelet_network(self) -> Optional[LaneletNetwork]:
        """Just for ease of use to avoid dg_scenario.scenario..."""
        return self.scenario.lanelet_network if self.scenario else None
