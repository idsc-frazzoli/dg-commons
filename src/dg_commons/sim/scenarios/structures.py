from dataclasses import dataclass, field, InitVar
from random import randint
from typing import Optional, MutableMapping, Iterable, Union

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

    scenario: InitVar[Scenario]
    """A CommonRoad scenario"""
    static_obstacles: InitVar[list[StaticObstacle]]
    """Static obstacles to be added to the scenario, this is just an initialization cariable"""
    use_road_boundaries: bool = False
    """If True the external boundaries of the road are forced to be obstacles """
    road_boundaries_buffer: float = 0.1
    """Buffer to be added to the lanelets when building the road boundaries"""
    strtree_obstacles: STRtree = field(init=False)
    """Store the obstacles in a spatial index for fast collision detection"""
    _static_obstacles: tuple[StaticObstacle] = field(default_factory=tuple)
    """An internal registry of the static obstacles"""
    _scenario: Scenario = None
    """An internal registry of the static obstacles"""

    def __post_init__(self, scenario: Scenario = None, static_obstacles: list[StaticObstacle] = None):
        self._scenario = scenario
        if self._scenario is not None:
            assert isinstance(self.scenario, Scenario), self.scenario
        if static_obstacles is None:
            static_obstacles = []
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
        self._static_obstacles = tuple(static_obstacles)

    @property
    def lanelet_network(self) -> Optional[LaneletNetwork]:
        """Just for ease of use to avoid dg_scenario.scenario..."""
        return self.scenario.lanelet_network if self.scenario else None

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    @property
    def static_obstacles(self) -> tuple[StaticObstacle]:
        return self._static_obstacles
