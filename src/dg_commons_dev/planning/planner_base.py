import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from shapely.geometry.base import BaseGeometry
from dataclasses import dataclass
from dg_commons_dev.planning.rrt_utils.sampling import BaseBoundaries
from dg_commons_dev.planning.rrt_utils.utils import Node
from dg_commons_dev.planning.rrt_utils.goal_region import GoalRegion


class Planner(ABC):
    """ Planner interface """
    REF_PARAMS: dataclass

    @abstractmethod
    def planning(self, start: Node, goal: GoalRegion, obstacle_list: List[BaseGeometry],
                 sampling_bounds: BaseBoundaries, search_until_max_iter: bool = False,
                 limit_angles: Tuple[float, float] = (-math.pi, math.pi),
                 min_distance: float = 0.1) -> Optional[List[Node]]:
        """ Find path and returns it as a sequence of nodes """
        pass

    @abstractmethod
    def plot_results(self, create_animation: bool) -> None:
        """ Generate and save plots and animations """
        pass

    @abstractmethod
    def get_width(self) -> None:
        """ Returns security width of car """
        pass
