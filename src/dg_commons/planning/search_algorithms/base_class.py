import math
from abc import abstractmethod, ABC
from decimal import Decimal
from typing import List, Tuple, Union, Any, Set

import numpy as np
from commonroad.scenario.lanelet import Lanelet
from commonroad.common.util import Interval
from commonroad.planning.planning_problem import PlanningProblem

from dg_commons.planning import Trajectory, MotionPrimitivesGenerator, PlanningGoal
from dg_commons.planning.search_algorithms.node import PriorityNode
from dg_commons.sim import SimParameters
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.models.vehicle_dynamic import VehicleStateDyn, VehicleModelDyn
from dg_commons.sim.scenarios import DgScenario
from dg_commons.sim.simulator import SimContext


class SearchBaseClass(ABC):
    """
    Abstract base class for search based motion planners. The class was adapted to dg-commons
    from the original code from CommonRoad (Cyber-Physical Systems Group, Technical University of Munich):
    https://gitlab.lrz.de/tum-cps/commonroad-search/-/blob/master/SMP/motion_planner/search_algorithms/base_class.py
    """

    def __init__(self, scenario: DgScenario, planningProblem: PlanningProblem,
                 motion_primitive_generator: MotionPrimitivesGenerator,
                 initial_vehicle_state: VehicleState,
                 goal: PlanningGoal):
        self.dgscenario: DgScenario = scenario
        self.scenario = scenario.scenario
        self.planningProblem: PlanningProblem = planningProblem
        self.motion_primitive_generator: MotionPrimitivesGenerator = motion_primitive_generator
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
        self.motion_primitive_initial = motion_primitive_generator.generate(self.state_initial)
        self.list_ids_lanelets_initial = []
        self.list_ids_lanelets_goal = []
        self.time_desired = None
        self.position_desired = None
        self.velocity_desired = None
        self.orientation_desired = None
        self.distance_initial = None
        self.dict_lanelets_costs = {}
        self.parse_planning_problem()
        self.initialize_lanelets_costs()

    def parse_planning_problem(self) -> None:
        """
        Parses the given planning problem, and computes related attributes.
        """
        assert isinstance(self.planningProblem, PlanningProblem), "Given planning problem is not valid!"

        # get lanelet id of the initial state
        self.list_ids_lanelets_initial = self.scenario.lanelet_network.find_lanelet_by_position(
            [self.planningProblem.initial_state.position])[0]

        # get lanelet id of the goal region, which can be of different types
        self.list_ids_lanelets_goal = None
        if hasattr(self.planningProblem.goal.state_list[0], 'position'):
            if hasattr(self.planningProblem.goal.state_list[0].position, 'center'):
                self.list_ids_lanelets_goal = self.scenario.lanelet_network.find_lanelet_by_position(
                    [self.planningProblem.goal.state_list[0].position.center])[0]

            elif hasattr(self.planningProblem.goal.state_list[0].position, 'shapes'):
                self.list_ids_lanelets_goal = self.scenario.lanelet_network.find_lanelet_by_position(
                    [self.planningProblem.goal.state_list[0].position.shapes[0].center])[0]
                self.planningProblem.goal.state_list[0].position.center = \
                    self.planningProblem.goal.state_list[0].position.shapes[0].center

        # set attributes with given planning problem
        if hasattr(self.planningProblem.goal.state_list[0], 'time_step'):
            self.time_desired = self.planningProblem.goal.state_list[0].time_step
        else:
            self.time_desired = Interval(0, np.inf)

        self.position_desired = None
        if hasattr(self.planningProblem.goal.state_list[0], 'position'):
            if hasattr(self.planningProblem.goal.state_list[0].position, 'vertices'):
                self.position_desired = self.calc_goal_interval(
                    self.planningProblem.goal.state_list[0].position.vertices)

            elif hasattr(self.planningProblem.goal.state_list[0].position, 'center'):
                x = self.planningProblem.goal.state_list[0].position.center[0]
                y = self.planningProblem.goal.state_list[0].position.center[1]
                self.position_desired = [Interval(start=x, end=x), Interval(start=y, end=y)]

        if hasattr(self.planningProblem.goal.state_list[0], 'velocity'):
            self.velocity_desired = self.planningProblem.goal.state_list[0].velocity
        else:
            self.velocity_desired = Interval(0, np.inf)

        if hasattr(self.planningProblem.goal.state_list[0], 'orientation'):
            self.orientation_desired = self.planningProblem.goal.state_list[0].orientation
        else:
            self.orientation_desired = Interval(-math.pi, math.pi)

        # create necessary attributes
        if hasattr(self.planningProblem.goal.state_list[0], 'position'):
            self.distance_initial = SearchBaseClass.distance(self.planningProblem.initial_state.position,
                                                             self.planningProblem.goal.state_list[0].position.center)
        else:
            self.distance_initial = 0

    def initialize_lanelets_costs(self) -> None:
        """
        Initializes the heuristic costs for lanelets. The cost of a lanelet equals to the number
        of lanelets that should be traversed before reaching the goal region. The cost is set to
        -1 if it is not possible to reach the goal region from the lanelet, and 0 if it is within
        the list of goal lanelets.
        """
        # set lanelet costs to -1, except goal lanelet (0)
        for lanelet in self.scenario.lanelet_network.lanelets:
            self.dict_lanelets_costs[lanelet.lanelet_id] = -1

        if self.list_ids_lanelets_goal is not None:
            for ids_lanelet_goal in self.list_ids_lanelets_goal:
                self.dict_lanelets_costs[ids_lanelet_goal] = 0

            # calculate costs for lanelets, this is a recursive method
            for ids_lanelet_goal in self.list_ids_lanelets_goal:
                list_lanelets_visited = []
                lanelet_goal = self.scenario.lanelet_network.find_lanelet_by_id(ids_lanelet_goal)
                self.calc_lanelet_cost(lanelet_goal, 1, list_lanelets_visited)

    @abstractmethod
    def execute_search(self) -> Tuple[Union[None, List[List[VehicleState]]], Union[None, List[Set[Trajectory]]]]:
        """
        The actual search algorithms are implemented in the children classes.
        """
        pass

    @staticmethod
    def calc_goal_interval(vertices: np.ndarray) -> List[Interval]:
        """
        Calculate the maximum Intervals of the goal position given as vertices.
        @param: vertices: vertices which describe the goal position.
        """
        min_x = np.inf
        max_x = -np.inf

        min_y = np.inf
        max_y = -np.inf
        for vertex in vertices:
            if vertex[0] < min_x:
                min_x = vertex[0]
            if vertex[0] > max_x:
                max_x = vertex[0]
            if vertex[1] < min_y:
                min_y = vertex[1]
            if vertex[1] > max_y:
                max_y = vertex[1]
        return [Interval(start=min_x, end=max_x), Interval(start=min_y, end=max_y)]

    def calc_lanelet_cost(self, lanelet_current: Lanelet, dist: int, list_lanelets_visited: List[int]) -> None:
        """
        Calculates distances of all lanelets which can be reached through recursive adjacency/predecessor relationship
        by the current lanelet. This is a recursive implementation.

        :param lanelet_current: the current lanelet object (Often set to the goal lanelet).
        :param dist: the initial distance between 2 adjacent lanelets (Often set to 1). This value will increase
        recursively during the execution of this function.
        :param list_lanelets_visited: list of visited lanelet id. In the iterations, visited lanelets will not be
        considered. This value changes during the recursive implementation.
        """
        if lanelet_current.lanelet_id in list_lanelets_visited:
            return
        else:
            list_lanelets_visited.append(lanelet_current.lanelet_id)

        if lanelet_current.predecessor is not None:
            for pred in lanelet_current.predecessor:
                if self.dict_lanelets_costs[pred] == -1 or self.dict_lanelets_costs[pred] > dist:
                    self.dict_lanelets_costs[pred] = dist

            for pred in lanelet_current.predecessor:
                self.calc_lanelet_cost(self.lanelet_network.find_lanelet_by_id(pred), dist + 1, list_lanelets_visited)

        if lanelet_current.adj_left is not None and lanelet_current.adj_left_same_direction:
            if self.dict_lanelets_costs[lanelet_current.adj_left] == -1 or \
                    self.dict_lanelets_costs[lanelet_current.adj_left] > dist:
                self.dict_lanelets_costs[lanelet_current.adj_left] = dist
                self.calc_lanelet_cost(self.lanelet_network.find_lanelet_by_id(lanelet_current.adj_left), dist + 1,
                                       list_lanelets_visited)

        if lanelet_current.adj_right is not None and lanelet_current.adj_right_same_direction:
            if self.dict_lanelets_costs[lanelet_current.adj_right] == -1 or \
                    self.dict_lanelets_costs[lanelet_current.adj_right] > dist:
                self.dict_lanelets_costs[lanelet_current.adj_right] = dist
                self.calc_lanelet_cost(self.lanelet_network.find_lanelet_by_id(lanelet_current.adj_right), dist + 1,
                                       list_lanelets_visited)

    @staticmethod
    def euclidean_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the euclidean distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        # TODO: check if np.sqrt((pos1-pos2).T @ (pos1-pos2)) is faster
        return np.sqrt((pos1[0] - pos2[0]) * (pos1[0] - pos2[0]) + (pos1[1] - pos2[1]) * (pos1[1] - pos2[1]))

    @staticmethod
    def manhattan_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the manhattan distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def chebyshev_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the chebyshev distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

    @staticmethod
    def canberra_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the canberra distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return abs(pos1[0] - pos2[0]) / (abs(pos1[0]) + abs(pos2[0])) + abs(pos1[1] - pos2[1]) / (
                abs(pos1[1]) + abs(pos2[1]))

    @staticmethod
    def cosine_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the cosine distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return 1 - (pos1[0] * pos2[0] + pos1[1] * pos2[1]) / (
                np.sqrt(pos1[0] ** 2 + pos2[0] ** 2) * np.sqrt(pos1[1] ** 2 + pos2[1] ** 2))

    @staticmethod
    def sum_of_squared_difference(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the squared euclidean distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2

    @staticmethod
    def mean_absolute_error(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns a half of the manhattan distance between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return 0.5 * (abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]))

    @staticmethod
    def mean_squared_error(pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Returns the mean of squared difference between 2 points.

        :param pos1: the first point
        :param pos2: the second point
        """
        return 0.5 * ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    @classmethod
    def distance(cls, pos1: np.ndarray, pos2: np.ndarray = np.zeros(2), distance_type=0) -> float:
        """
        Returns the distance between 2 points, the type is specified by 'type'.

        :param pos1: the first point
        :param pos2: the second point
        :param distance_type: specifies which kind of distance is used:
            1: manhattanDistance,
            2: chebyshevDistance,
            3: sumOfSquaredDifference,
            4: meanAbsoluteError,
            5: meanSquaredError,
            6: canberraDistance,
            7: cosineDistance.
        """
        if distance_type == 0:
            return cls.euclidean_distance(pos1, pos2)
        elif distance_type == 1:
            return cls.manhattan_distance(pos1, pos2)
        elif distance_type == 2:
            return cls.chebyshev_distance(pos1, pos2)
        elif distance_type == 3:
            return cls.sum_of_squared_difference(pos1, pos2)
        elif distance_type == 4:
            return cls.mean_absolute_error(pos1, pos2)
        elif distance_type == 5:
            return cls.mean_squared_error(pos1, pos2)
        elif distance_type == 6:
            return cls.canberra_distance(pos1, pos2)
        elif distance_type == 7:
            return cls.cosine_distance(pos1, pos2)
        return math.inf

    def calc_euclidean_distance(self, current_node: PriorityNode) -> float:
        """
        Calculates the euclidean distance to the desired goal position.

        @param current_node:
        @return:
        """
        if self.position_desired[0].contains(current_node.list_paths[-1][-1].p[0]):
            delta_x = 0.0
        else:
            delta_x = min([abs(self.position_desired[0].start - current_node.list_paths[-1][-1].p[0]),
                           abs(self.position_desired[0].end - current_node.list_paths[-1][-1].p[0])])
        if self.position_desired[1].contains(current_node.list_paths[-1][-1].p[1]):
            delta_y = 0
        else:
            delta_y = min([abs(self.position_desired[1].start - current_node.list_paths[-1][-1].p[1]),
                           abs(self.position_desired[1].end - current_node.list_paths[-1][-1].p[1])])

        return np.sqrt(delta_x ** 2 + delta_y ** 2)

    def reached_goal(self, path: List[VehicleState]) -> bool:
        """
        Goal-test every state of the path and returns true if one of the state satisfies all conditions for the goal
        region: position, orientation, velocity, time.

        :param path: the path to be goal-tested

        """
        for i in range(len(path)):
            if self.goal.is_fulfilled(path[i]):
                return True
        return False

    def check_collision(self, trajectory: Trajectory) -> bool:
        """Check collisions of the planned trajectory with the environment
        :param trajectory: The planned trajectory
        :return: True if at least one collision happened, False otherwise"""
        vehicle_states = trajectory.values
        env_obstacles = self.sim_context.dg_scenario.strtree_obstacles
        collision = False
        for vs in vehicle_states:
            x0_p1 = VehicleStateDyn(x=vs.x, y=vs.y, theta=vs.theta,
                                    vx=vs.vx, delta=vs.delta)
            p_model = VehicleModelDyn.default_car(x0_p1)
            footprint = p_model.get_footprint()
            assert footprint.is_valid
            p_shape = footprint
            items = env_obstacles.query_items(p_shape)
            for idx in items:
                candidate = self.sim_context.dg_scenario.static_obstacles[idx]
                if p_shape.intersects(candidate.shape):
                    collision = True
        return collision
