from abc import ABC, abstractmethod
from math import atan
from typing import Iterable

import numpy as np
from commonroad_dc import pycrcc
from commonroad_dc.pycrcc import CollisionObject
from commonroad_dc.pycrcc import Shape as crShape
from dg_commons import PlayerName, SE2Transform
from dg_commons.sim import SimObservations
from dg_commons.sim.scenarios import DgScenario


class VisibilityFilter(ABC):
    @abstractmethod
    def filter(self, scenario: DgScenario, full_obs: SimObservations, pov: PlayerName) -> SimObservations:
        pass


class IdVisFilter(VisibilityFilter):
    """Identity visibility filter"""

    @abstractmethod
    def filter(self, scenario: DgScenario, full_obs: SimObservations, pov: PlayerName) -> SimObservations:
        return full_obs


class Lidar2D:
    pose: SE2Transform = SE2Transform([0, 0], 0)
    range: float = 20.0
    field_of_view: float = 2 * np.pi
    angle_resolution = atan(1 / range)

    def filter(self, scenario: DgScenario, full_obs: SimObservations, pov: PlayerName) -> SimObservations:
        pass

    def view_vertices(self, obstacles: Iterable[crShape]):
        vertices = []

        has_omnidirectional_view = self.field_of_view > 2 * np.pi - self.angle_resolution
        if not has_omnidirectional_view:
            vertices.append(self.pose.p)

        num_ray_angles = int(self.field_of_view / self.angle_resolution) + 1
        ray_angles = self.pose.theta + np.linspace(-self.field_of_view / 2, self.field_of_view / 2, num_ray_angles)

        cc = pycrcc.CollisionChecker()
        for obstacle in obstacles:
            cc.add_collision_object(obstacle)

        for angle in ray_angles:
            ray_end = [self.pose.p[0] + self.range * np.cos(angle), self.pose.p[1] + self.range * np.sin(angle)]

            ray_hits = cc.raytrace(self.pose.p[0], self.pose.p[1], ray_end[0], ray_end[1], False)

            if not ray_hits:
                vertices.append(ray_end)
            else:
                closest_hit = ray_end
                closest_hit_distance = self.range
                for ray_hit in ray_hits:
                    hit_in = ray_hit[0:2]
                    hit_out = ray_hit[2:]
                    distance_to_hit_in = np.hypot(hit_in[0] - self.pose.p[0], hit_in[1] - self.pose.p[1])
                    distance_to_hit_out = np.hypot(hit_out[0] - self.pose.p[0], hit_out[1] - self.pose.p[1])
                    if distance_to_hit_in < closest_hit_distance:
                        closest_hit_distance = distance_to_hit_in
                        closest_hit = hit_in
                    if distance_to_hit_out < closest_hit_distance:
                        closest_hit_distance = distance_to_hit_out
                        closest_hit = hit_out
                vertices.append(closest_hit)

        vertices.append(vertices[0])

        return np.array(vertices)
