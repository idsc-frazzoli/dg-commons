from dataclasses import dataclass

from shapely.geometry import Polygon

@dataclass()
class SamplingArea():
    area: Polygon