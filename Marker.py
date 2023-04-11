import numpy as np
from shapely.geometry import *
from shapely import affinity
from shapely.ops import unary_union
from gdshelpers.geometry.chip import Cell


class GridMarker:
    def __init__(self, origin=(0, 0), line_width=0.5):
        self._marker = None
        self._points = None
        self._centroids = None
        self.marker_height = None
        self.marker_width = None
        self.origin = origin
        self.line_width = line_width
        self.horizontals = []
        self.verticals = []

    @property
    def marker(self):
        mark = self._marker.envelope.buffer(20).difference(self._marker.buffer(self.line_width))
        return affinity.translate(mark, self.origin[0], self.origin[1])

    @marker.setter
    def marker(self, marker):
        self._marker = marker
        self.marker_height = marker.bounds[3] - marker.bounds[1]
        self.marker_width = marker.bounds[2] - marker.bounds[0]
        return

    @property
    def points(self):
        return affinity.translate(self._points, self.origin[0], self.origin[1])

    @property
    def centres(self):
        geoms = self.marker.envelope.difference(self.marker)
        return [Point(g.centroid) for g in geoms.geoms]

    def create_grid(self, x_min=17, x_pitch=-1, x_num=14, y_min=17, y_pitch=-1, y_num=14, line_width=None):
        if line_width is not None:
            self.line_width = line_width

        start = (0, 0)
        x_starts = [Point(start)]
        y_starts = [Point(start)]

        x_s = 0
        for xx in range(x_num):
            x_s += ((xx + 1) * x_pitch) + x_min
            x_starts.append(Point(np.add(start, (x_s, 0))))

        y_s = 0
        for yy in range(y_num):
            y_s += ((yy + 1) * y_pitch) + y_min
            y_starts.append(Point(np.add(start, (0, y_s))))

        x_lines = []
        y_lines = []

        for start in x_starts:
            end = Point(start.x, y_s)
            x_lines.append(LineString((start, end)))

        for start in y_starts:
            end = Point(x_s, start.y)
            y_lines.append(LineString((start, end)))

        self.marker = MultiLineString((*x_lines, *y_lines))
        self.horizontals = MultiLineString(y_lines)
        self.verticals = MultiLineString(x_lines)
        self._points = MultiPoint([Point(v.intersection(h)) for v in x_lines for h in y_lines])

    def layout_marker(self, gds_cell, offset=(0, 0), centred=False, on_corners=True, on_sides=False, on_center=False):
        x_min, y_min, x_max, y_max = gds_cell.bounds

        if centred is True:
            x_min -= offset[0] + self.marker_width/2
            y_min -= offset[1] + self.marker_height/2
            x_max += offset[0] - self.marker_width/2
            y_max += offset[1] - self.marker_height/2
        else:
            x_min -= offset[0]
            y_min -= offset[1]
            x_max += offset[0]
            y_max += offset[1]

        corners = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        sides = [self.halfway(pnt_1, pnt_2) for pnt_1, pnt_2 in zip(corners, [*corners[1:], corners[0]])]
        center = self.halfway(corners[0], corners[2])

        positions = []
        if on_corners is True:
            [positions.append(pos) for pos in corners]
        if on_sides is True:
            [positions.append(pos) for pos in sides]
        if on_center is True:
            positions.append(center)

        markers = []
        temp = self.origin
        for pos in positions:
            self.origin = pos
            markers.append(self.marker)
        self.origin = temp
        return unary_union(markers)

    @staticmethod
    def halfway(pnt_1, pnt_2):
        return tuple(np.divide(np.add(pnt_1, pnt_2), 2))

"""
grid = GridMarker((100, 300))
grid.create_grid(3, 1, 8, 6, 2, 5, line_width=0.2)

cell = Cell("marker")
cell.add_to_layer(1, grid.marker)
cell.add_to_layer(2, grid.layout_marker(cell, offset=(100, 100), on_corners=True, on_sides=True))
cell.show()"""