from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from shapely.geometry import Point, Polygon
from dg_commons import PlayerName
from dg_commons.geo import PoseState
from dg_commons.sim.models import extract_pose_from_state
from geometry import translation_from_SE2


@dataclass
class SharedPolygonGoal:
    """Represents a shared goal that can be collected by any agent"""
    goal_id: str
    polygon: Polygon
    collected_by: Optional[PlayerName] = None
    collection_time: Optional[float] = None

    def is_collected(self) -> bool:
        return self.collected_by is not None


@dataclass
class CollectionPoint:
    """Represents a collection point where goals should be delivered"""
    point_id: str
    polygon: Polygon
    collected_goals: List[str] = field(default_factory=list)
    collection_times: Dict[str, float] = field(default_factory=dict)


class SharedPolygonGoalsManager:
    """
    Manager for shared goals and collection points in a multi-agent system.

    This manager tracks:
    - Shared goals scattered in the environment
    - Collection points where goals should be delivered
    - Which agent is currently carrying which goal
    """

    def __init__(
        self,
        shared_goals: List[SharedPolygonGoal],
        collection_points: List[CollectionPoint],
    ):
        """
        Initialize the shared goals manager.

        Args:
            shared_goals: List of shared goals available for collection
            collection_points: List of collection points for goal delivery
        """
        self.all_goals: Dict[str, SharedPolygonGoal] = {g.goal_id: g for g in shared_goals}
        self.shared_goals: Dict[str, SharedPolygonGoal] = {g.goal_id: g for g in shared_goals}
        self.collection_points: Dict[str, CollectionPoint] = {cp.point_id: cp for cp in collection_points}
        self.agent_carrying: Dict[PlayerName, Optional[str]] = {}  # Maps agent to goal_id they're carrying

    def update(self, agents_states: Dict[PlayerName, PoseState], simulation_time: float) -> Dict[str, any]:
        """
        Update the state of goals and collection points based on agent positions.

        This method should be called in the simulation loop to check:
        1. If any agent collected a goal (agent enters goal polygon while not carrying)
        2. If any agent delivered a goal (agent enters collection point while carrying)

        Args:
            agents_states: Dictionary mapping agent names to their current states

        Returns:
            Dictionary containing update events:
            - 'goals_collected': List of (agent_name, goal_id) tuples
            - 'goals_delivered': List of (agent_name, goal_id, collection_point_id) tuples
        """
        events = {
            'goals_collected': [],
            'goals_delivered': [],
        }

        # Initialize agent carrying state if not present
        for agent_name in agents_states:
            if agent_name not in self.agent_carrying:
                self.agent_carrying[agent_name] = None

        for agent_name, state in agents_states.items():
            # Extract agent position
            pose = extract_pose_from_state(state)
            xy = translation_from_SE2(pose)
            agent_point = Point(xy)

            # Check if agent is currently carrying a goal
            carrying_goal_id = self.agent_carrying[agent_name]

            if carrying_goal_id is None:
                # Agent not carrying anything - check for goal collection
                for goal_id, goal in self.shared_goals.items():
                    if not goal.is_collected() and goal.polygon.contains(agent_point):
                        # Agent collected this goal
                        goal.collected_by = agent_name
                        goal.collection_time = simulation_time
                        self.all_goals[goal_id].collected_by = agent_name
                        self.all_goals[goal_id].collection_time = simulation_time
                        self.agent_carrying[agent_name] = goal_id
                        events['goals_collected'].append((agent_name, goal_id))
                        break  # Agent can only collect one goal at a time
            else:
                # Agent is carrying a goal - check for delivery at collection points
                for cp_id, collection_point in self.collection_points.items():
                    if collection_point.polygon.contains(agent_point):
                        # Agent delivered the goal
                        collection_point.collected_goals.append(carrying_goal_id)
                        collection_point.collection_times[carrying_goal_id] = simulation_time
                        self.agent_carrying[agent_name] = None
                        # Mark goal as completed (remove from shared goals or mark as delivered)
                        if carrying_goal_id in self.shared_goals:
                            del self.shared_goals[carrying_goal_id]
                        events['goals_delivered'].append((agent_name, carrying_goal_id, cp_id))
                        break  # Agent can only deliver to one point at a time

        return events

    def get_available_goals(self) -> List[SharedPolygonGoal]:
        """Get list of goals that haven't been collected yet"""
        return [g for g in self.shared_goals.values() if not g.is_collected()]

    def get_collected_goals(self) -> List[SharedPolygonGoal]:
        """Get list of goals that are currently being carried"""
        return [g for g in self.shared_goals.values() if g.is_collected()]

    def get_agent_carrying_goal(self, agent_name: PlayerName) -> Optional[str]:
        """Get the goal_id that an agent is currently carrying, if any"""
        return self.agent_carrying.get(agent_name)

    def get_total_goals_collected(self) -> int:
        """Get total number of goals that have been delivered to collection points"""
        return sum(len(cp.collected_goals) for cp in self.collection_points.values())

    def is_all_goals_collected(self) -> bool:
        """Check if all goals have been delivered to collection points"""
        return len(self.shared_goals) == 0