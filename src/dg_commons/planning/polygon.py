from typing import Type
from dg_commons.seq.sequence import DgSampledSequence
from shapely.geometry import Polygon


__all__ = ["PolygonSequence"]


class PolygonSequence(DgSampledSequence[Polygon]):
    """Container for a polygon as a sampled sequence"""

    @property
    def XT(self) -> Type[Polygon]:
        return Polygon
