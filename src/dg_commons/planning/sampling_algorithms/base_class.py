import random
from abc import ABC, abstractmethod
from decimal import Decimal

from shapely.geometry import Polygon, Point
from commonroad.planning.planning_problem import PlanningProblem

from dg_commons import PlayerName
from dg_commons.planning import PlanningGoal
from dg_commons.planning.sampling_algorithms.utils import SamplingArea
from dg_commons.sim import SimParameters
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.scenarios import DgScenario
from dg_commons.sim.simulator import SimContext
from dg_commons.maps.road_bounds import build_road_boundary_obstacle


class SamplingBaseClass(ABC):
    """
    Abstract base class for search based motion planners. The class was adapted to dg-commons
    from the original code from CommonRoad (Cyber-Physical Systems Group, Technical University of Munich):
    https://gitlab.lrz.de/tum-cps/commonroad-search/-/blob/master/SMP/motion_planner/search_algorithms/base_class.py
    """

    def __init__(self, player_name: PlayerName, scenario: DgScenario, planningProblem: PlanningProblem,
                 initial_vehicle_state: VehicleState,
                 goal: PlanningGoal, goal_state: VehicleState, seed: int):
        self.dgscenario: DgScenario = scenario
        self.scenario = scenario.scenario
        self.planningProblem: PlanningProblem = planningProblem
        self.goal = goal
        self.sim_context = SimContext(
            dg_scenario=scenario,
            models={},
            players={},
            param=SimParameters(dt=Decimal("0.01"), dt_commands=Decimal("0.1"), sim_time_after_collision=Decimal(1),
                                max_sim_time=Decimal(7)), )
        self.lanelet_network = self.scenario.lanelet_network
        self.list_obstacles = self.scenario.obstacles
        self.state_initial: VehicleState = initial_vehicle_state
        self.node_list = []
        self.sampling_area = self.get_sampling_area()
        self.goal_state = goal_state
        self.player_name = player_name
        random.seed(seed)


    @abstractmethod
    def planning(self):
        pass

    def get_sampling_area(self) -> SamplingArea:
        lanelet_bounds = build_road_boundary_obstacle(self.scenario)
        polygon_bounds = []
        for lanelet_bound in lanelet_bounds:
            coords = lanelet_bound.coords.xy
            x = coords[0]
            y = coords[1]
            polygon_bounds.append([x[0], y[0]])
            polygon_bounds.append([x[1], y[1]])

        pol = Polygon(polygon_bounds)

        return SamplingArea(pol)

    def sample_point(self) -> Point:
        minx, miny, maxx, maxy = self.sampling_area.area.bounds
        search_flag = True
        num_it = 0
        while search_flag:
            point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if self.sampling_area.area.contains(point):
                return point
            num_it += 1
            if num_it > 100:
                raise Exception('Could not sample point')


