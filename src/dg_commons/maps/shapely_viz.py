from typing import Union, Dict

import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    GeometryCollection,
    LinearRing,
    MultiPoint,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

from dg_commons import logger


# todo support for MultiShapes needs to be verified

# todo add support for ellipses


class ShapelyViz:
    def __init__(self, ax: Axes = plt.gca()):
        self.ax: Axes = ax

    def add_shape(self, shape: BaseGeometry, **style_kwargs: Dict):
        """Plot a given shapely object.
        Shapely geometries currently supported are: `GeometryCollection`, `LineString`,
        `LinearRing`, `MultiLineString`, `MultiPoint`, `MultiPolygon`, `Point`, and `Polygon`.

        :param shape: Shape(ly) to plot. Subclass from BaseGeometry.
        :param **style_kwargs: Dictionary of attributes to style the given shape.
        Available style attributes are the one passed to the corresponding matplotlib shapes
        :raise TypeError: When given `shape` type is not currently supported.
        """

        if shape.is_empty:
            logger.info("Given shape is empty, returning.")
            return

        if isinstance(shape, (Point, MultiPoint)):
            self._plot_points(shape, **style_kwargs)
        elif isinstance(shape, (LineString, MultiLineString, LinearRing)):
            self._plot_lines(shape, **style_kwargs)
        elif isinstance(shape, (Polygon, MultiPolygon)):
            self._plot_polys(shape, **style_kwargs)
        elif isinstance(shape, GeometryCollection):
            for poly in shape:
                self.add_shape(poly, **style_kwargs)
        else:
            raise NotImplementedError(f"Given `shape` argument is of an unexpected type [{type(shape).__name__}]")

    def _plot_points(self, shape: Union[Point, MultiPoint], **style_kwargs: Dict):
        """Internal method for plotting given non-empty, Point or MultiPoint `shape` object.
        :param shape: Shape(ly) to plot.
        :param **style_kwargs: Dictionary of attributes to style the given shape.
        :raises TypeError: When given `shape` is not a `Point` or `MultiPoint` shapely geometry.
        """
        if shape.is_empty:
            logger.info("Given shape is empty, returning.")
            return
        # todo
        x, y = [], []
        if isinstance(shape, MultiPoint):
            for point in shape:
                x.append(point.x)
                y.append(point.y)
        elif isinstance(shape, Point):
            x, y = map(list, shape.xy)
        else:
            raise TypeError(f"Given `shape` argument is of an unexpected type [{type(shape).__name__}]")
        circle = mpatches.Circle(xy=(x, y), **style_kwargs)
        self.ax.add_patch(circle)

    def _plot_lines(self, shape: Union[LineString, MultiLineString, LinearRing], **style_kwargs: Dict):
        """Internal method for plotting given non-empty, LineString, MultiLineString, or LinearRing `shape` object.
        :param shape: Shape(ly) to plot.
        :param **style_kwargs (dict): Dictionary of attributes to style the given shape.
        :raises TypeError: When given `shape` is not a `LineString`, `MultiLineString`, or `LinearRing` shapely geometry.
        """
        if shape.is_empty:
            logger.info("Given shape is empty, returning.")
            return

        if isinstance(shape, (LineString, LinearRing)):
            x, y = map(list, shape.xy)
            self.ax.plot(x, y, **style_kwargs)
        elif isinstance(shape, MultiLineString):
            for line in shape:
                x, y = map(list, line.xy)
                self.ax.plot(x, y, **style_kwargs)
        else:
            raise TypeError(f"Given `shape` argument is of an unexpected type [{type(shape).__name__}]")

    def _plot_polys(self, shape: Union[Polygon, MultiPolygon], **style_kwargs: Dict):
        """Internal method for plotting given non-empty, Polygon or MultiPolygon `shape` object.
        :param shape: Shape(ly) to plot.
        :param **style_kwargs (dict): Dictionary of attributes to style the given shape.
        :raises TypeError: When given `shape` is not a `Polygon` or `MultiPolygon` shapely geometry.
        """
        if shape.is_empty:
            logger.info("Given shape is empty, returning.")
            return

        if not isinstance(shape, (Polygon, MultiPolygon)):
            raise TypeError(f"Given `shape` argument is of an unexpected type [{type(shape).__name__}]")

        xs, ys = self._get_poly_coordinates(shape)
        for idx, xy in enumerate(zip(xs, ys)):
            xy_array = list(zip(*xy))
            if idx == 0:
                poly = mpatches.Polygon(xy_array, **style_kwargs)
            else:
                poly = mpatches.Polygon(xy_array, facecolor=self.ax.get_facecolor())
            self.ax.add_patch(poly)

    def _get_poly_coordinates(self, shape: Union[Polygon, MultiPolygon]):
        """Internal method for translating shapely polygon coordinates to bokeh polygon plotting coordinates.
        :param shape: Shape(ly) to plot. Polygon, MultiPolygon
        :raises TypeError: When given `shape` is not a `Polygon` or `MultiPolygon` shapely geometry.
        """
        x, y = [], []
        if isinstance(shape, MultiPolygon):
            for poly in shape:
                poly_x, poly_y = self._get_poly_coordinates(poly)
                x += poly_x
                y += poly_y
            return x, y

        elif isinstance(shape, Polygon):
            extr_x, extr_y = map(list, shape.exterior.xy)
            intr_x, intr_y = [], []
            for i in shape.interiors:
                _x, _y = map(list, i.xy)
                intr_x.append(_x[:-1])
                intr_y.append(_y[:-1])
            combined_x, combined_y = [extr_x[:-1]], [extr_y[:-1]]
            combined_x += intr_x
            combined_y += intr_y
            return combined_x, combined_y

        else:
            raise TypeError(f"Given `shape` argument is of an unexpected type [{type(shape).__name__}]")
