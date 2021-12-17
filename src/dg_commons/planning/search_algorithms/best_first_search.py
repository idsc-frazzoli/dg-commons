from decimal import Decimal

import numpy as np
from abc import abstractmethod, ABC
from typing import Tuple, List, Union

from commonroad.planning.planning_problem import PlanningProblem

from dg_commons.planning import MotionPrimitivesGenerator, Trajectory, PlanningGoal
from dg_commons.planning.search_algorithms.base_class import SearchBaseClass
from dg_commons.planning.search_algorithms.node import PriorityNode
from dg_commons.planning.search_algorithms.queue import PriorityQueue
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim.scenarios import DgScenario


class BestFirstSearch(SearchBaseClass, ABC):
    """
    Abstract class for Best First Search algorithm. The class was adapted to dg-commons
    from the original code from CommonRoad (Cyber-Physical Systems Group, Technical University of Munich):
    https://gitlab.lrz.de/tum-cps/commonroad-search/-/blob/master/SMP/motion_planner/search_algorithms/best_first_search.py
    """

    def __init__(self, scenario: DgScenario, planningProblem: PlanningProblem,
                 motion_primitive_generator: MotionPrimitivesGenerator,
                 initial_vehicle_state: VehicleState, goal: PlanningGoal):
        super().__init__(scenario=scenario, planningProblem=planningProblem,
                         motion_primitive_generator=motion_primitive_generator,
                         initial_vehicle_state=initial_vehicle_state, goal=goal)
        self.frontier = PriorityQueue()

    @abstractmethod
    def evaluation_function(self, node_current: PriorityNode):
        """
        Function that evaluates f(n) in the inherited classes.
        @param node_current:
        @return: cost
        """
        pass

    def heuristic_function(self, node_current: PriorityNode) -> float:
        """
        Function that evaluates the heuristic cost h(n) in inherited classes.
        The example provided here estimates the time required to reach the goal state from the current node.
        @param node_current: time to reach the goal
        @return:
        """
        if self.reached_goal(node_current.list_trajectories[-1].values):
            return 0.0

        if self.position_desired is None:
            return self.time_desired.start - node_current.list_trajectories[-1].get_end()

        else:
            velocity = node_current.list_trajectories[-1].values[-1].vx

            if np.isclose(velocity, 0):
                return np.inf

            else:
                return self.calc_euclidean_distance(current_node=node_current) / velocity

    def execute_search(self) -> Tuple[Union[None, List[List[VehicleState]]], Union[None, List[Trajectory]]]:
        """
        Implementation of Best First Search (tree search) using a Priority queue.
        The evaluation function of each child class is implemented individually.
        """

        # first node
        node_initial = PriorityNode(list_trajectories=[Trajectory([Decimal(0)], [self.state_initial])],
                                    set_primitives=self.motion_primitive_initial,
                                    depth_tree=0, priority=0)

        # add current node (i.e., current path and primitives) to the frontier
        f_initial = self.evaluation_function(node_initial)
        self.frontier.insert(item=node_initial, priority=f_initial)

        while not self.frontier.empty():
            # pop the last node
            node_current = self.frontier.pop()

            # goal test
            if self.reached_goal(node_current.list_trajectories[-1].values):
                path_solution = node_current.list_paths
                # return solution
                return path_solution, node_current.list_trajectories

            # translate/rotate motion primitive to current position
            t0 = node_current.list_trajectories[-1].get_end()
            x0 = node_current.list_trajectories[-1].at(t0)
            trajectory_translated = self.motion_primitive_generator.generate(
                x0=x0, t0=t0)

            # check all possible successor primitives(i.e., actions) for current node
            for trajectory_primitives in trajectory_translated:

                # check for collision, if is not collision free it is skipped
                if self.check_collision(trajectory=trajectory_primitives):
                    continue

                trajectory_new_list = node_current.list_trajectories + [trajectory_primitives]
                node_child = PriorityNode(list_trajectories=trajectory_new_list,
                                          set_primitives=trajectory_translated,
                                          depth_tree=node_current.depth_tree + 1,
                                          priority=node_current.priority)
                f_child = self.evaluation_function(node_current=node_child)

                # insert the child to the frontier:
                self.frontier.insert(item=node_child, priority=f_child)

        return None, None


class UniformCostSearch(BestFirstSearch):
    """
    Class for Uniform Cost Search (Dijkstra) algorithm.
    """

    def __init__(self, scenario: DgScenario, planningProblem: PlanningProblem,
                 motion_primitive_generator: MotionPrimitivesGenerator,
                 initial_vehicle_state: VehicleState, goal: PlanningGoal):
        super().__init__(scenario=scenario, planningProblem=planningProblem,
                         motion_primitive_generator=motion_primitive_generator,
                         initial_vehicle_state=initial_vehicle_state, goal=goal)

    def evaluation_function(self, node_current: PriorityNode) -> float:
        """
        Evaluation function of UCS is f(n) = g(n)
        """

        # calculate g(n)
        node_current.priority += (len(node_current.list_paths[-1]) - 1) * self.scenario.dt

        return node_current.priority


class GreedyBestFirstSearch(BestFirstSearch):
    """
    Class for Greedy Best First Search algorithm.
    """

    def __init__(self, scenario: DgScenario, planningProblem: PlanningProblem,
                 motion_primitive_generator: MotionPrimitivesGenerator,
                 initial_vehicle_state: VehicleState, goal: PlanningGoal):
        super().__init__(scenario=scenario, planningProblem=planningProblem,
                         motion_primitive_generator=motion_primitive_generator,
                         initial_vehicle_state=initial_vehicle_state, goal=goal)

    def evaluation_function(self, node_current: PriorityNode) -> float:
        """
        Evaluation function of GBFS is f(n) = h(n)
        """

        node_current.priority = self.heuristic_function(node_current=node_current)
        return node_current.priority


class AStarSearch(BestFirstSearch):
    """
    Class for A* Search algorithm.
    """

    def __init__(self, scenario: DgScenario, planningProblem: PlanningProblem,
                 motion_primitive_generator: MotionPrimitivesGenerator,
                 initial_vehicle_state: VehicleState, goal: PlanningGoal):
        super().__init__(scenario=scenario, planningProblem=planningProblem,
                         motion_primitive_generator=motion_primitive_generator,
                         initial_vehicle_state=initial_vehicle_state, goal=goal)

    def evaluation_function(self, node_current: PriorityNode) -> float:
        """
        Evaluation function of A* is f(n) = g(n) + h(n)
        """
        # calculate g(n)
        node_current.priority += (len(node_current.list_paths[-1]) - 1) * self.scenario.dt

        # f(n) = g(n) + h(n)
        return node_current.priority + self.heuristic_function(node_current=node_current)
