from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import atan
from typing import Iterable

import numpy as np
from commonroad_dc import pycrcc
from commonroad_dc.pycrcc import Shape as crShape
from shapely.geometry import Polygon, Point

from dg_commons import SE2Transform


@dataclass
class Sensor(ABC):
    pose: SE2Transform = SE2Transform([0, 0], 0)
    range: float = 20.0
    field_of_view: float = 2 * np.pi

    def __post_init__(self):
        self.angle_resolution = atan(1 / self.range)
        assert self.field_of_view <= 2 * np.pi
        assert self.range >= 0

    @abstractmethod
    def fov_as_polygon(self, obstacles: Iterable[crShape]) -> Polygon:
        pass

    def is_omnidirectional(self) -> bool:
        return self.field_of_view > 2 * np.pi - self.angle_resolution


@dataclass
class FullRangeSensor(Sensor):
    def fov_as_polygon(self, obstacles: Iterable[crShape]) -> Polygon:
        """
        Returns a polygon representing the field of view of the sensor.
        :param obstacles:
        :return:
        """
        x, y = self.pose.p
        if self.is_omnidirectional():
            return Point(x, y).buffer(self.range)
        else:
            raise NotImplementedError("Only omnidirectional sensors are supported for full range sensors.")


@dataclass
class VisRangeSensor(Sensor):
    def fov_as_polygon(self, obstacles: Iterable[crShape]) -> Polygon:
        vertices = []

        if not self.is_omnidirectional():
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
                closest_hit_distance = self.range**2
                for ray_hit in ray_hits:
                    hit_in = ray_hit[0:2]
                    hit_out = ray_hit[2:]
                    distance_to_hit_in = (hit_in[0] - self.pose.p[0]) ** 2 + (hit_in[1] - self.pose.p[1]) ** 2
                    distance_to_hit_out = (hit_out[0] - self.pose.p[0]) ** 2 + (hit_out[1] - self.pose.p[1]) ** 2
                    if distance_to_hit_in < closest_hit_distance:
                        closest_hit_distance = distance_to_hit_in
                        closest_hit = hit_in
                    if distance_to_hit_out < closest_hit_distance:
                        closest_hit_distance = distance_to_hit_out
                        closest_hit = hit_out
                vertices.append(closest_hit)

        vertices.append(vertices[0])
        return Polygon(vertices)
