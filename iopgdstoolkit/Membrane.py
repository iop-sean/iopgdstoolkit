"""
Author: Sean Bommer
Email: sean.bommer.2014@uni.strath.ac.uk
version: 0.0.1
release date: 20/10/2022

Notes: May be buggy, and documentation is missing in some areas.
I advise that I am asked for assistance getting started as I have not written documentation yet.
"""

from math import pi
import numpy as np
from gdshelpers.geometry.chip import Cell
import matplotlib.pyplot as plt
# ======================================================================
from shapely.geometry import *
from shapely.geometry.polygon import orient
from gdshelpers.parts.port import Port
from gdshelpers.parts.text import Text
from gdshelpers.parts.waveguide import Waveguide
from shapely import affinity as aff
from shapely.ops import unary_union
from shapely.ops import voronoi_diagram
import copy
from multiprocessing import Pool


class Membrane:
    def __init__(self):
        self._origin = (0, 0)
        self._resist = "positive"
        self.fillet = 0
        self.chamfer = 0
        self._devices = []
        self.device_buffer = []
        self.void_buffer = []
        self.etch_regions = []

        self.lines = {
            "pixel":    [],
            "inner":    [],
            "outer":    [],
            "anchors":  [],
            "holes":    [],
            "removed":  [],
        }

        self.geometry = {
            "pixel":    Polygon(),
            "inner":    Polygon(),
            "outer":    Polygon(),
            "anchors":  [],
            "holes":    [],
            "removed":  [],
            "devices":  [],
            "voids":    [],
        }
        self.mesh = []

    # ===============================
    # bunch of setters and getters
    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, new_origin):
        x_offset, y_offset = np.subtract(new_origin, self.origin)
        self.move(x_offset, y_offset)
        self._origin = new_origin

    def move(self, x_off=0, y_off=0):
        def shift_it(polygon):
            polygon = aff.translate(polygon, x_off, y_off)
            return polygon

        for key, values in self.geometry.items():
            temp = self.nested_op([values], shift_it)
            self.geometry[key] = temp[0]

        for key, values in self.lines.items():
            temp = self.nested_op([values], shift_it)
            self.lines[key] = temp[0]
        self._origin = tuple(np.add(self.origin, (x_off, y_off)))

    @property
    def pixel(self):
        pixel = self.geometry["pixel"]
        pixel = self.modify_corners(pixel, self.chamfer, 3)
        pixel = self.modify_corners(pixel, self.fillet, 1)
        return pixel

    @pixel.setter
    def pixel(self, polygon):
        self.lines["pixel"] = self.create_sides(polygon)
        self.geometry["pixel"] = polygon

    @property
    def interiors(self):
        interiors = unary_union(self.geometry["interiors"])
        interiors = self.modify_corners(interiors, self.chamfer, 3)
        interiors = self.modify_corners(interiors, self.fillet, 1)
        return interiors

    @pixel.setter
    def interiors(self, polygons):
        self.lines["interiors"] = [self.create_sides(polygon) for polygon in polygons]
        self.geometry["interiors"] = polygons

    @property
    def pixel_lines(self):
        return unary_union(self.flatten_list(self.lines["pixel"]))

    @property
    def anchors(self):
        return MultiPolygon(self.flatten_list(self.geometry["anchors"]))

    @property
    def anchor_lines(self):
        return unary_union(self.flatten_list(self.lines["anchors"]))

    @property
    def border(self):
        return self.assemble_border()

    @border.setter
    def border(self, polygon):
        self.lines["inner"] = self.create_sides(polygon[0])
        self.lines["outer"] = self.create_sides(polygon[1])
        self.geometry["inner"] = polygon[0]
        self.geometry["outer"] = polygon[1]

    @property
    def border_lines(self):
        return unary_union(self.flatten_list(self.lines["inner"]))

    @property
    def holes(self):
        return unary_union(self.flatten_list(self.geometry["holes"]))

    @property
    def hole_lines(self):
        return unary_union(self.flatten_list(self.lines["holes"]))

    @property
    def removed(self):
        return unary_union(self.flatten_list(self.geometry["removed"]))

    @removed.setter
    def removed(self, polygon):
        self.lines["removed"] = self.create_sides(polygon)
        self.geometry["removed"] = polygon

    @property
    def devices(self):
        return unary_union(self.flatten_list(self.geometry["devices"]))

    @devices.setter
    def devices(self, value):
        self.geometry["devices"] = value

    @property
    def device_boundary(self):
        buff_d, d = self.buffered_parts(self.geometry["devices"], self.device_buffer)
        buff_v, v = self.buffered_parts(self.geometry["voids"], self.void_buffer)
        buffed = unary_union([buff_d, buff_v]).buffer(self.fillet).buffer(-self.fillet)
        return buffed

    @property
    def resist_tone(self):
        return self._resist

    @resist_tone.setter
    def resist_tone(self, value):
        self._resist = value

    def basic_mem(self):
        self.create_pixel(side_lengths=[100, 100], num_sides=4)
        self.create_border(border_gap=10, border_width=50)
        self.create_anchors(widths=[2, 10], sides=[0, 2], position_list=[[0.2, -0.2]])

    # pixel
    def create_pixel(self, side_lengths, num_sides, origin=None):
        """
        creates the pixel by generating a regular polygon, as in the angle between each vertice is equal.
        when num_sides = 4, height and width can be specified as side_lengths = [width, height]
        :param side_lengths: a list of integers or floats, that represent the length of each vertice.
        :param num_sides: integer, 4=rectangle, 6=hexagon etc. if side_lengths < num_sides, side_lengths is repeated.
        :param origin: the centre point on the pixel, if not specified the origin stored in the membrane class is used.
        :return: Shapely Polygon Object
        """
        if origin is None:
            origin = self.origin
        else:
            self.origin = origin
        pixel = self.draw_polygon(side_lengths, num_sides, origin)
        self.pixel = pixel
        return self.pixel

    def create_pixel_from_device(self, parts, pixel_buffer=10, device_buffer=0, technique="bounds", fillet=200, simplify=0.0):
        """
        creates a pixel based on geometry of photonics device, bounds technique results in a rectangular pixel, convex
        hull is like wrapping an elastic band around the object.
        :param parts: list of shapely objects and/or gdshelpers objects.
        :param pixel_buffer: minimum spacing between the edge of the device and the pixel edge
        :param device_buffer: area surrounding the device to be etched to ensure optical confinement
        :param technique: "bounds" or "convex" hull to specify pixel creation method.
        :param simplify: float between 0.0 and 1.0, reduces arcs to corners reducing the num_sides in a convex_hull
        :return: Shapely Polygon Object
        """

        device = unary_union(self.device_shapely(parts))
        if technique == "bounds" or technique == 0:
            pixel = box(*device.bounds).buffer(pixel_buffer, join_style=2, resolution=200)
        elif technique == "convex_hull" or technique == 1:
            pixel = device.convex_hull.buffer(pixel_buffer, join_style=2, resolution=200)
        elif technique == "buffer" or technique == 2:
            pixel = device.buffer(fillet, resolution=200).buffer(pixel_buffer-fillet, join_style=2, resolution=200)
        else:
            raise ValueError("technique should either be bounds or convex hull, or use 0, 1 respectively")

        self._origin = self.find_centre(pixel)
        self.create_devices(parts, device_buffer)

        pixel = pixel.simplify(simplify)
        self.pixel = pixel
        self.interiors = [Polygon(ring) for ring in pixel.interiors]
        return pixel

    @staticmethod
    def draw_polygon(side_lengths, num_sides, origin):
        """
        generates a regular polygon, as in the angle between each vertice is equal.
        :param side_lengths: a list of integers or floats, that represent the length of each vertice.
        :param num_sides:  integer, 4=rectangle, 6=hexagon etc. if side_lengths < num_sides, side_lengths is repeated.
        :param origin: the centre point of the polygon
        :return: Shapely Polygon Object
        """
        angle = 2*pi/num_sides
        if isinstance(side_lengths, list) is False:
            side_lengths = [side_lengths]

        if len(side_lengths) < num_sides:
            temp = side_lengths * num_sides
            side_lengths = temp[0:num_sides]

        corners = [(0, 0)]*(num_sides+1)
        for index, length in enumerate(side_lengths[:-1]):
            x = length * np.sin((index - 1) * angle)
            y = length * np.cos((index - 1) * angle)
            corners[index+1] = tuple(np.add(corners[index], (x, y)))

        polygon = Polygon(corners)
        polygon = aff.translate(polygon, -polygon.centroid.x + origin[0], -polygon.centroid.y + origin[1])
        return polygon

    # border
    def create_border(self, border_gap=10, border_width=10, style=0, side_lengths=150, num_sides=4):
        """
        Creates the boundary region around the pixel that it will attach to
        :param border_gap: integer or float, region between the pixel and the border, this determines the anchor length
        :param border_width: integer or float, the thickness of the bordering region
        :param style: integer, style selector to determine how you want to create the border
        :param side_lengths: a list of integers or floats, that represent the length of each vertice.
        :param num_sides: integer, 4=rectangle, 6=hexagon etc. if side_lengths < num_sides, side_lengths is repeated.
        """
        self.border = self.draw_border(self.geometry["pixel"], self.draw_polygon, self.origin, border_gap,
                                       border_width, style, side_lengths, num_sides)

    # border draw
    @staticmethod
    def draw_border(polygon, shape_generator, origin, border_gap=10, border_width=10, style=0,
                    side_lengths=150, num_sides=4):
        """
        creates a bordering structure around a shapely polygon, default is to create a concentric structure
        :param polygon: shapely polygon object
        :param shape_generator: function that produces polygons with inputs side_lengths and num_sides
        :param origin: the centre point of the border, should be the same as the polygon origin
        :param border_gap: the minimum distance between the edge of the polygon and the inner section of the border
        :param border_width: how thick the border is
        :param style: selector for drawing style of inner and outer borders, 0=(concentric, concentric),
        1=(concentric, custom) 2=(custom, custom)
        :param side_lengths: a list of integers or floats, that represent the length of each vertice.
        :param num_sides: integer, 4=rectangle, 6=hexagon etc. if side_lengths < num_sides, side_lengths is repeated.
        :return: list of shapely polygons representing [inner, outer]
        """

        if style == 0:
            inner = polygon.buffer(distance=border_gap, join_style=2)
            outer = polygon.buffer(distance=border_width+border_gap, join_style=2)
        elif style == 1:
            inner = polygon.buffer(distance=border_gap, join_style=2)
            outer = shape_generator(side_lengths, num_sides, origin)
        elif style == 2:
            inner = shape_generator(side_lengths, num_sides, origin)
            outer = inner.buffer(border_width, join_style=2)
        else:
            raise ValueError("style should be integer from 0 to 2 inclusive")
        return [inner, outer]

    # anchors
    def create_anchors(self, widths, sides, position_list, normalized=True, style="centre", overlap=0.005, e=False,
                       ignore_curves=False):
        """
        creates anchors/tethers that attach the pixel area to the surrounding border,
        these are what keep the membrane suspended.
        :param widths: list of integers or floats, evenly spaced points along the anchor_line that determine the shape
        :param sides: list of integers, the selector for which side to attach anchors to
        :param position_list: list of integers or floats, where along the side the anchors should be positioned
        :param normalized: True or False,
        determines where anchor position is determined by length of the side or fixed value
        :param style: "centre" plot widths on both sides, "left" or "right"  plot widths on either side
        :param overlap: float between -1 and 1 that represent the percentage you want the anchor to overlap the border
        """
        line_list = self.draw_anchor_lines(self.geometry["pixel"], self.geometry["inner"], sides, position_list,
                                           normalized, overlap, e=e, ignore_curves=ignore_curves)
        self.lines["anchors"] = [line_list]
        anchor_list = self.draw_anchors(widths, line_list, style)
        self.geometry["anchors"] = [anchor_list]

    def add_anchors(self, widths, sides, position_list, normalized=True, style="centre", overlap=0.0, e=False,
                    ignore_curves=False):
        """
        creates anchors/tethers that attach the pixel area to the surrounding border,
        these are what keep the membrane suspended
        :param widths: list of integers or floats, evenly spaced points along the anchor that determine the shape
        :param sides: list of integers, the selector for which side to attach anchors to
        :param position_list: list of integers or floats, where along the side the anchors should be positioned
        :param normalized: True or False,
        determines where anchor position is determined by length of the side or fixed value
        :param style: "centre" plot widths on both sides, "left" or "right"  plot widths on either side
        :param overlap: float between -1 and 1 that represent the amount you want the anchor to overlap the border by
        :return: A MultiPolygon of the generated anchors
        """
        line_list = self.draw_anchor_lines(self.geometry["pixel"], self.geometry["inner"], sides, position_list,
                                           normalized, overlap, e=e, ignore_curves=ignore_curves)
        self.lines["anchors"].append(line_list)
        anchor_list = self.draw_anchors(widths, line_list, style)
        self.geometry["anchors"].append(anchor_list)

    def add_internal_anchors(self, widths, side_lists, position_lists, normalized=True, style="centre", overlap=0.0,
                             e=False):
        """
        creates anchors/tethers that attach the pixel area to the surrounding border,
        these are what keep the membrane suspended
        :param widths: list of integers or floats, evenly spaced points along the anchor that determine the shape
        :param side_lists: list of integers, the selector for which side to attach anchors to
        :param position_lists: list of integers or floats, where along the side the anchors should be positioned
        :param normalized: True or False,
        determines where anchor position is determined by length of the side or fixed value
        :param style: "centre" plot widths on both sides, "left" or "right"  plot widths on either side
        :param overlap: float between -1 and 1 that represent the amount you want the anchor to overlap the border by
        :return: A MultiPolygon of the generated anchors
        """

        openings = [orient(Polygon(ring)) for ring in self.pixel.interiors]
        inner_openings = [orient(Polygon(ring)) for ring in self.geometry["inner"].interiors]

        if len(position_lists) < len(inner_openings):
            temp = position_lists*len(inner_openings)
            position_lists = temp[0:len(inner_openings)]

        if len(side_lists) < len(inner_openings):
            temp = side_lists*len(inner_openings)
            side_lists = temp[0:len(inner_openings)]

        line_list = []
        for opening, inner_opening, side_list, position_list in zip(openings, inner_openings, side_lists, position_lists):
            line_list.extend(self.draw_anchor_lines(opening, inner_opening, side_list, position_list, normalized, overlap, e=e))

        self.lines["anchors"].append(line_list)
        anchor_list = self.draw_anchors(widths, line_list, style)
        self.geometry["anchors"].append(anchor_list)

    # anchors draw
    def draw_anchor_lines(self, pixel, internal_border, sides, position_list, normalized=False, overlap=0.005, e=False,
                          ignore_curves=False):
        """
        draws the connecting lines between the vertices of the pixel and the polygon representation of the inner border
        :param pixel: shapely polygon who's sides will be selected for anchor placement.
        :param internal_border: polygon representation of the inner border, to be connected to
        :param sides: list of integers that represent what vertice of the pixel should be connected to
        :param position_list: list of integers or floats, where along the side the anchors should be positioned
        :param normalized: True or False,
        determines where anchor position is determined by length of the side or fixed value
        :param overlap: float between -1 and 1 that represent the amount you want the anchor to overlap the border by
        :return: a list of Shapely LineString objects
        """

        points = []
        if len(position_list) < len(sides):
            temp = position_list*len(sides)
            position_list = temp[0:len(sides)]

        if ignore_curves is True:
            poly_lines = self.extract_flats(pixel, 5)
        else:
            if e is False:
                lst = pixel.exterior.coords
                poly_lines = [LineString(line) for line in zip(lst, lst[1:])]
            else:
                poly_lines = [pixel.exterior]
                sides = [0]
                position_list = [[pos for positions in position_list for pos in positions]]

        anchor_lines = []
        par = internal_border.boundary
        for index, side in enumerate(sides):
            positions = position_list[index]
            for position in positions:
                pnt = poly_lines[side].interpolate(position, normalized=normalized)
                points.append(pnt)

                # par = internal_border.boundary
                pnt2 = par.interpolate(par.project(pnt))

                anchor_line = LineString([pnt, pnt2])
                if overlap != 0:
                    anchor_line = aff.scale(anchor_line, 1 + overlap, 1 + overlap, origin=anchor_line.centroid)
                anchor_lines.append(anchor_line)

        return anchor_lines

    @staticmethod
    def draw_anchors(widths, anchor_lines, style="centre"):
        """
        Draws the shape of the anchors by essentially plotting the widths vector against the anchor line
        :param widths: list of integers or floats, evenly spaced points along the anchor that determine the shape
        :param anchor_lines: list of Shapely LineStrings, to plot the widths vector against
        :param style: "centre" plot widths on both sides, "left" or "right"  plot widths on either side
        :return: A list of Shapely Polygons of the generated anchors
        """
        anchor_list = []
        for line in anchor_lines:
            left_points = []
            right_points = []
            step = 1 / (len(widths) - 1)
            pos = 0
            if style == "centre":
                for width in widths:

                    par = line.parallel_offset(distance=width / 2, side='left')
                    left_points.append(par.interpolate(pos, normalized=True))

                    par = line.parallel_offset(distance=-width / 2, side='left')
                    par = LineString(list(par.coords)[::-1])
                    right_points.append(par.interpolate(pos, normalized=True))
                    pos += step

                right_points = right_points[::-1]

            elif style == "left":
                for width in widths:
                    par = line.parallel_offset(distance=width, side='left')
                    left_points.append(par.interpolate(pos, normalized=True))

                    right_points.append(line.interpolate(pos, normalized=True))
                    pos += step

                right_points = right_points[::-1]

            elif style == "right":
                for width in widths[::-1]:
                    left_points.append(line.interpolate(pos, normalized=True))

                    par = line.parallel_offset(distance=width, side='right')
                    right_points.append(par.interpolate(pos, normalized=True))
                    pos += step

            anchor_list.append(Polygon([*left_points, *right_points]))
        return anchor_list

    # hole creation
    def create_holes(self, radius=1, path_length=10, slot_length=None, slot_angle=None, buffer=0):
        """
        creates arrays of evenly distributed holes across the pixel of the membrane that allow the
        membrane to be under-etched quicker
        :param radius: the radius of the circular or slot shaped hole
        :param path_length: separation between the edge of one hole to the edge of another, supports up to 2 values
        :param slot_length: optional: length of the slot -2*radius
        :param slot_angle: optional: angle in radians that the slot is orientated
        :param buffer: int or float, minimum gap between the edge of pixel and hole
        """
        hole = self.draw_hole(radius, slot_length, slot_angle)
        points = self.draw_layout(self.pixel.buffer(-buffer), hole, path_length)
        holes = self.draw_hole_layout(hole, points)

        self.lines["holes"] = [points]
        self.geometry["holes"] = [holes]

        self.keep_overlap(self.pixel.buffer(-buffer))

        """buff_d, d = self.buffered_parts(self.geometry["devices"], self.device_buffer)
        buff_v, v = self.buffered_parts(self.geometry["voids"], self.void_buffer)
        buffed = unary_union([buff_d, buff_v])"""
        self.remove_overlap(self.device_boundary)

        self.etch_regions = self.create_etch_region(self.hole_lines, self.pixel)

    def add_holes(self, radius=1, path_length=10, slot_length=None, slot_angle=None, buffer=0, remove_old=True):
        """
        creates arrays of evenly distributed holes across the pixel of the membrane that allow the
        membrane to be under-etched quicker
        :param radius: the radius of the circular or slot shaped hole
        :param path_length: separation between the edge of one hole to the edge of another, supports up to 2 values
        :param slot_length: optional: length of the slot -2*radius
        :param slot_angle: optional: angle in radians that the slot is orientated
        :param buffer: int or float, minimum gap between the edge of pixel and hole
        :param remove_old: Boolean, determines whether to remove overlapping holes or not
        """

        hole = self.draw_hole(radius, slot_length, slot_angle)
        points = self.draw_layout(self.pixel.buffer(-buffer), hole, path_length)
        holes = self.draw_hole_layout(hole, points)

        self.remove_overlap(MultiPolygon(holes)) if remove_old else None
        self.lines["holes"].append(points)
        self.geometry["holes"].append(holes)

        buff_d, d = self.buffered_parts(self.geometry["devices"])
        buff_v, v = self.buffered_parts(self.geometry["voids"])
        buffed = unary_union([buff_d, buff_v])
        self.remove_overlap(buffed)

        self.etch_regions = self.create_etch_region(self.hole_lines, self.pixel)

    # hole draw
    @staticmethod
    def draw_hole(radius=1, slot_length=None, slot_angle=None, res=16):
        """
        creates a circular or slot shaped hole as a template to be used for underetch
        :param radius: radius of circle or slot
        :param slot_length: optional: length of the slot -2*radius
        :param slot_angle: optional: angle in radians that the slot is orientated
        :param res: how many lines to approximate curves with
        :return: Shapely Polygon Object
        """
        if slot_length is None or slot_angle is None:
            hole = Point(0, 0).buffer(radius, resolution=res)
        else:
            x = (slot_length - 2 * radius) * np.cos(slot_angle)
            y = (slot_length - 2 * radius) * np.sin(slot_angle)

            line = LineString([(x, y), (-x, -y)])
            hole = line.buffer(radius, resolution=res)
        return hole

    @staticmethod
    def draw_layout(polygon, hole, path_length):
        """
        creates a list of uniformly spaced Shapely Point Objects that represent the layout of the under-etch holes
        :param polygon: the Shapely Polygon Object that the holes will be distributed inside
        :param hole: The Shapely Polygon that represents the hole in the membrane pixel
        :param path_length: separation between the edge of one hole to the edge of another, supports up to 2 values
        :return: a list of Shapely Point Objects
        """
        bounds = polygon.bounds

        xx = bounds[2] - bounds[0]
        yy = bounds[3] - bounds[1]

        shape_bds = hole.bounds
        shape_xx = shape_bds[2] - shape_bds[0]
        shape_yy = shape_bds[3] - shape_bds[1]

        size = np.size(path_length)
        if size == 1:
            path = [path_length]*2
        else:
            path = path_length

        xx_via_num, xx_r = divmod(xx + path[0], path[0] + shape_xx)
        yy_via_num, yy_r = divmod(yy + path[1], path[1] + shape_yy)

        if xx_r <= 1:
            xx_r += path[0] + shape_xx

        if yy_r <= 1:
            yy_r += path[1] + shape_yy

        ends = list(bounds)
        ends[0] += shape_xx / 2 + xx_r / 2
        ends[1] += shape_yy / 2 + yy_r / 2

        xx_points = np.arange(ends[0], ends[2], path[0] + shape_xx)
        yy_points = np.arange(ends[1], ends[3], path[1] + shape_yy)

        points = [Point((x, y)) for x in xx_points for y in yy_points]
        return points

    @staticmethod
    def draw_hole_layout(hole, points):
        """
        creates a list of holes based on a hole template and a list of Shapely Point Objects
        :param hole: Shapely Polygon Object to be used as the template for hole distribution
        :param points: a list of uniformly spaced Shapely Point Objects
        :return: a list of uniformly spaced Shapely Polygon Objects
        """
        holes = [aff.translate(hole, xoff=point.x, yoff=point.y) for point in points]
        return holes

    # hole modification
    def remove_hole(self, lst):
        """
        removes holes from the pixel, useful if position become problematic for another structure
        :param lst: a reverse sorted list of lists that represent the index of which holes to remove from the pixel
        """
        for index1, collection in enumerate(lst):
            for index2 in collection:
                self.lines["holes"][index1].pop(index2)
                self.geometry["holes"][index1].pop(index2)

    def remove_overlap(self, polygon, buffer=0):
        """
        removes holes from the pixel based on if they overlap a Shapely Object
        :param polygon: Shapely Object to use as a tool for hole removal
        :param buffer: minimum space around polygon that should also be cleared
        """
        removed_index = []

        if hasattr(polygon, 'get_shapely_object'):
            polygon = polygon.get_shapely_object()

        polygon = polygon.buffer(buffer)
        for lst in self.geometry["holes"]:
            temp = list(set([index for index, hole in enumerate(lst) if polygon.intersects(hole)]))
            temp.sort(reverse=True)
            removed_index.append(temp)
        self.remove_hole(removed_index)

    def keep_overlap(self, polygon, buffer=0):
        """
        keeps holes from the pixel based on if they overlap a Shapely Object
        :param polygon: Shapely Object to use as a tool for hole removal
        :param buffer: minimum space around polygon that should also be kept
        """
        removed_index = []

        if hasattr(polygon, 'get_shapely_object'):
            polygon = polygon.get_shapely_object()

        polygon = polygon.buffer(buffer)
        for lst in self.geometry["holes"]:
            temp = list(set([index for index, hole in enumerate(lst) if polygon.contains(hole) is False]))
            temp.sort(reverse=True)
            removed_index.append(temp)
        self.remove_hole(removed_index)

    def move_hole(self, lst, offset=(0, 0)):
        """
        moves holes on the pixel by a fixed amount
        :param lst: a reverse sorted list of lists that represent the index of which holes to move on the pixel
        :param offset: tuple representing how far to move the hole
        """
        for index1, collection in enumerate(lst):
            for index2 in collection:
                temp = self.lines["holes"][index1][index2]
                self.lines["holes"][index1][index2] = aff.translate(temp, offset[0], offset[1])
                temp = self.geometry["holes"][index1][index2]
                self.geometry["holes"][index1][index2] = aff.translate(temp, offset[0], offset[1])

    # smart hole creation
    def create_smart_holes(self, radius=1, path_length=10, order=1):
        hole = self.draw_hole(radius, None, None)
        points = self.draw_layout(self.pixel, hole, path_length)
        min_size = 0.001

        device_region = self.device_layer(False, True, False, False, False, False, False)\
            .difference(self.device_boundary)

        regions = self.create_etch_region(MultiPoint(points), device_region)
        points = [polygon.centroid for polygon in regions.geoms]

        for index in range(order):
            regions = self.create_etch_region(MultiPoint(points), device_region)
            points = [polygon.centroid for polygon in regions.geoms]

        smart_holes = self.draw_hole_layout(hole, points)
        self.lines["holes"] = [points]
        self.geometry["holes"] = [smart_holes]

        self.etch_regions = self.create_etch_region(self.hole_lines, self.pixel)
        return

    def create_smart_holes2(self, radius=1, path_length=10, order=1, min_region=2, buffer=0):
        self.create_holes(radius, path_length)

        device_region = self.pixel.buffer(-buffer).difference(self.device_boundary)

        regions = self.create_etch_region(self.hole_lines, device_region).buffer(-min_region)
        points = [polygon.centroid for polygon in regions.geoms]

        """cell = Cell("test")
        cell.add_to_layer(1, self.device_layer())
        cell.add_to_layer(0, self.pixel.difference(self.etch_regions.buffer(-0.2)))

        x_factor = device_region.bounds[2] - device_region.bounds[0] + 50
        y_factor = device_region.bounds[3] - device_region.bounds[1] + 50
        origin = [(x * x_factor, y * y_factor) for y in range(4) for x in range(6)]"""

        for index in range(order):
            device_region = self.pixel.buffer(-buffer).difference(self.device_boundary)
            regions = self.create_etch_region(MultiPoint(points), device_region)
            points = [polygon.centroid for polygon in regions.geoms]

            smart_holes = self.draw_hole_layout(self.draw_hole(radius), points)
            self.lines["holes"] = [points]
            self.geometry["holes"] = [smart_holes]
            self.remove_overlap(self.device_boundary)

            # self.origin = origin[index+1]
        self.etch_regions = self.create_etch_region(self.hole_lines, self.pixel)
        """     
        cell.add_to_layer(1, self.device_layer())
        cell.add_to_layer(0, self.pixel.difference(self.etch_regions.buffer(-0.2)))
        cell.show()
        """
        return

    @staticmethod
    def create_etch_region(multipoint, plane):
        return voronoi_diagram(multipoint).buffer(-0.001).intersection(plane)

    @staticmethod
    def find_mfp(regions):
        mfps = []
        for region in regions.geoms:
            centre = region.centroid
            pnts = region.exterior
            mfps.append(centre.hausdorff_distance(pnts))
        return plt.hist(mfps, bins=100, range=(0, 50))

    # devices

    def create_devices(self, parts, buffer, origin=None, x_off=0.0, y_off=0.0):
        """
        creates an instance of the photonic devices on the membrane, and allows proper placement of device on membrane
        :param parts: list of parts, a mix of shapely/gdshelpers objects are accepted
        :param buffer: surrounding area around the devices to be cleared out
        :param origin: if None, centre of device is used.
        if (x, y) that coordinate on the device is centred on the membrane, can used gdshelper ports to allow placement
        of coupling guides etc.
        :param x_off: the x offset from the given origin value
        :param y_off: the y offset from the given origin value
        :return: coordinate tuple of the device centre
        """
        self.device_buffer = [buffer]
        temp = self.device_shapely(parts)
        devices = unary_union(temp)
        centre = self.find_centre(devices)
        if origin is None:
            x_off, y_off = list(np.subtract(self.origin, (centre[0]-x_off, centre[1]-y_off)))
        else:
            x_off, y_off = list(np.subtract(self.origin, (origin[0]-x_off, origin[1]-y_off)))

        self.geometry["devices"] = [[aff.translate(t, x_off, y_off) for t in temp]]
        return centre

    def add_devices(self, parts, buffer, origin=None, x_off=0.0, y_off=0.0):
        """
        adds an instance of the photonic devices on the membrane, and allows proper placement of device on membrane
        :param parts: list of parts, a mix of shapely/gdshelpers objects are accepted
        :param buffer: surrounding area around the devices to be cleared out
        :param origin: if None, centre of device is used.
        if (x, y) that coordinate on the device is centred on the membrane, can used gdshelper ports to allow placement
        of coupling guides etc.
        :param x_off: the x offset from the given origin value
        :param y_off: the y offset from the given origin value
        :return: coordinate tuple of the device centre
        """
        self.device_buffer.append(buffer)
        temp = self.device_shapely(parts)
        devices = unary_union(temp)
        centre = self.find_centre(devices)
        if origin is None:
            x_off, y_off = list(np.subtract(self.origin, (centre[0] - x_off, centre[1] - y_off)))
        else:
            x_off, y_off = list(np.subtract(self.origin, (origin[0] - x_off, origin[1] - y_off)))

        self.geometry["devices"].append([aff.translate(t, x_off, y_off) for t in temp])
        return centre

    def create_voids(self, parts, buffer, origin=None, x_off=0.0, y_off=0.0):
        """
        creates an instance of void structures on the membrane, and allows proper placement of device on membrane
        :param parts: list of parts, a mix of shapely/gdshelpers objects are accepted
        :param buffer: surrounding area around the devices to be cleared out
        :param origin: if None, centre of device is used.
        if (x, y) that coordinate on the device is centred on the membrane, can used gdshelper ports to allow placement
        of coupling guides etc.
        :param x_off: the x offset from the given origin value
        :param y_off: the y offset from the given origin value
        :return: coordinate tuple of the device centre
        """
        self.void_buffer = [buffer]
        temp = self.device_shapely(parts)
        voids = unary_union(temp)
        centre = self.find_centre(voids)
        if origin is None:
            x_off, y_off = list(np.subtract(self.origin, (centre[0]-x_off, centre[1]-y_off)))
        else:
            x_off, y_off = list(np.subtract(self.origin, (origin[0]-x_off, origin[1]-y_off)))

        self.geometry["voids"] = [[aff.translate(t, x_off, y_off) for t in temp]]
        return centre

    def add_voids(self, parts, buffer, origin=None, x_off=0.0, y_off=0.0):
        """
        adds an instance of the void structures on the membrane, and allows proper placement of device on membrane
        :param parts: list of parts, a mix of shapely/gdshelpers objects are accepted
        :param buffer: surrounding area around the devices to be cleared out
        :param origin: if None, centre of device is used.
        if (x, y) that coordinate on the device is centred on the membrane, can used gdshelper ports to allow placement
        of coupling guides etc.
        :param x_off: the x offset from the given origin value
        :param y_off: the y offset from the given origin value
        :return: coordinate tuple of the device centre
        """
        self.void_buffer.append(buffer)
        temp = self.device_shapely(parts)
        voids = unary_union(temp)
        centre = self.find_centre(voids)
        if origin is None:
            x_off, y_off = list(np.subtract(self.origin, (centre[0] - x_off, centre[1] - y_off)))
        else:
            x_off, y_off = list(np.subtract(self.origin, (origin[0] - x_off, origin[1] - y_off)))

        self.geometry["voids"].append([aff.translate(t, x_off, y_off) for t in temp])
        return centre

    @staticmethod
    def device_shapely(parts):
        """
        takes a list of parts and converts any gdshelpers objects into shapely objects
        :param parts: list of parts, a mix of shapely/gdshelpers objects are accepted
        :return: list of parts, only as Shapely Objects
        """
        if isinstance(parts, list) is False:
            parts = [parts]
        temp = []
        tiny = 0.001  # a tiny buffer can fix invalid geometries
        for part in parts:
            if hasattr(part, "get_shapely_object"):
                temp.append(part.get_shapely_object().buffer(tiny).buffer(-tiny))
            else:
                temp.append(part.buffer(tiny).buffer(-tiny))
        return temp

    # membrane assembly
    def membrane(self, pixel=True, border=True, anchors=True, holes=True, removed=True, holes_on_shallow=False):
        """
        performs all the boolean logic required to properly create a membrane device as well as applying any fillet and
        chamfer on the membrane that help to minimise membrane cracking.
        :param pixel: Boolean, whether or not to include the pixel
        :param border: Boolean, whether or not to include the border
        :param anchors: Boolean, whether or not to include the anchors
        :param holes: Boolean, whether or not to include the holes
        :param removed: Boolean, whether or not to include the removed border sections
        :param holes_on_shallow: determines if holes appear on shallow etched regions
        :return: A Shapely Object that represents the membrane device
        """

        # do any modification operations here to avoid permanently changing parts of the device
        # remove any holes that now don't fit inside the membrane
        p = self.modify_corners(self.pixel, self.chamfer, 2)
        p = self.modify_corners(p, self.fillet, 1)

        self.keep_overlap(p)
        buff_d, d = self.buffered_parts(self.geometry["devices"], self.device_buffer)
        buff_v, v = self.buffered_parts(self.geometry["voids"], self.void_buffer)
        buffed = unary_union([buff_d, buff_v]).buffer(self.fillet).buffer(-self.fillet)

        self.remove_overlap(buffed) if not holes_on_shallow else []

        """
        p = p if pixel else Polygon()
        b = self.border if border else Point(self.origin)
        a = self.anchors if anchors else Polygon()
        h = self.holes if holes else Polygon()
        r = self.removed if removed else Polygon()

        membrane = p.union(b)
        membrane = membrane.union(a)
        membrane = membrane.difference(h)

        """
        membrane = p if pixel else Polygon()
        membrane = membrane.union(self.border) if border else membrane
        membrane = membrane.union(self.anchors) if anchors else membrane
        membrane = membrane.difference(self.holes) if holes else membrane

        if self.resist_tone == "positive":
            cutout = self.geometry["inner"] if border else p
            membrane = cutout.difference(membrane)
            if removed and self.removed:
                membrane = membrane.union(self.removed)
        else:
            if removed and self.removed:
                membrane = membrane.difference(self.removed)
        return membrane

    def device_layer(self, device=True, pixel=True, border=True, anchors=True, holes=True, removed=True,
                     holes_on_shallow=False):
        """
        performs all the boolean logic required to properly the device layer of the membrane
        as well as applying any fillet and chamfer on the membrane that help to minimise membrane cracking.
        :param device: Boolean, whether or not to include the device
        :param pixel: Boolean, whether or not to include the pixel
        :param border: Boolean, whether or not to include the border
        :param anchors: Boolean, whether or not to include the anchors
        :param holes: Boolean, whether or not to include the holes
        :param removed: Boolean, whether or not to include the removed border sections
        :param holes_on_shallow: determines if holes appear on shallow etched regions
        :return: A Shapely Object that represents the membrane device layer
        """

        buff_d, d = self.buffered_parts(self.geometry["devices"], self.device_buffer)
        buff_d = buff_d.intersection(self.pixel)
        buff_v, v = self.buffered_parts(self.geometry["voids"], self.void_buffer)
        buff_v = buff_v.intersection(self.pixel)
        buffed = unary_union([buff_d, buff_v]).buffer(self.fillet).buffer(-self.fillet)

        d = d.intersection(self.pixel) if device else Polygon()

        if self.resist_tone == "negative":
            # temp = self.devices.buffer(self.device_buffer).difference(self.devices)
            device_layer = self.membrane(pixel, border, anchors, holes, removed, holes_on_shallow)\
                .difference(buffed).union(d)
            return device_layer
        else:
            # temp = self.devices.buffer(self.device_buffer).difference(self.devices)
            device_layer = self.membrane(pixel, border, anchors, holes, removed, holes_on_shallow)\
                .union(buffed).difference(d)
            return device_layer

    # useful methods
    def buffered_parts(self, parts, buffers):
        """
        generates the union of devices and the surrounding etch area
        :return: tuple of Shapely Objects that represent the joined devices and the surrounding etch area.
        """
        d = []
        d_buffered = []
        tiny = 0.0001
        for part, buffer in zip(parts, buffers):
            t = unary_union(part).buffer(tiny).buffer(-tiny)
            d.append(t)
            d_buffered.append(t.buffer(buffer + self.fillet).buffer(-self.fillet))

        return unary_union(d_buffered), unary_union(d)

    @staticmethod
    def create_sides(polygon):
        """
        generates a list of LineStrings of that represent the vertices of the Shapely Polygon object.
        Useful when you want to open ub bordering regions.
        :param polygon: polygon with exterior points used to generate LineStrings
        :return: List of Shapely LineString Objects representing border of polygon
        """
        if type(polygon) is Polygon:
            corners = polygon.exterior.coords
            sides = [LineString(line) for line in zip(corners, corners[1:])]
        elif type(polygon) is MultiPolygon:
            corner_list = [p.exterior.coords for p in polygon.geoms]
            sides = [LineString(line) for corners in corner_list
                     for line in zip(corners, corners[1:])]
        elif type(polygon) is LinearRing:
            corners = polygon.coords
            sides = [LineString(line) for line in zip(corners, corners[1:])]
        else:
            raise TypeError("Object type not supported")

        return sides

    def nested_op(self, nested_list, function):
        """
        recursive method, that allows functions to be applied to any Shapely objects inside nested lists
        :param nested_list: list in a list of any level, that eventually contains shapely objects
        :param function: the function you want to apply to the shapely objects
        :return: the nested list after operations are performed
        """
        for index, entry in enumerate(nested_list):
            if type(entry) is list:
                self.nested_op(entry, function)
            else:
                nested_list[index] = function(entry)
        return nested_list

    def flatten_list(self, nested_list):
        """
        recursive method, that takes shapely objects in nested list and appends them to 1D list
        :param nested_list: list in a list of any level, that eventually contains shapely objects
        :return: 1D list that contains all Shapely Objects in nested list
        """
        if not nested_list:
            return nested_list
        if isinstance(nested_list[0], list):
            return self.flatten_list(nested_list[0]) + self.flatten_list(nested_list[1:])
        return nested_list[:1] + self.flatten_list(nested_list[1:])

    def remove_border_section(self, sides, method="inner-outer"):
        """
        Allows you to remove a side of the border, useful in some cases for paneled membranes
        :param sides: list of integers, the selector for which side to remove
        :param method: String, whether it projects the outer region to the inner, or inner to outer
        :return: A MultiPolygon object of the removed regions
        """
        if isinstance(sides, list) is False:
            sides = [sides]
        if method == "outer-inner":
            for side in sides:
                pnt1, pnt2 = self.geometry["outer"].exterior.coords[side:side+2]
                pnt3 = self.lines["inner"][side].interpolate(self.lines["inner"][side].project(Point(pnt2)))
                pnt4 = self.lines["inner"][side].interpolate(self.lines["inner"][side].project(Point(pnt1)))
                self.geometry["removed"].append(Polygon([pnt1, pnt2, pnt3, pnt4]))
        elif method == "inner-outer":
            for side in sides:
                pnt1, pnt2 = self.geometry["inner"].exterior.coords[side:side+2]
                pnt3 = self.lines["outer"][side].interpolate(self.lines["outer"][side].project(Point(pnt2)))
                pnt4 = self.lines["outer"][side].interpolate(self.lines["outer"][side].project(Point(pnt1)))
                self.geometry["removed"].append(Polygon([pnt1, pnt2, pnt3, pnt4]))

    def assemble_border(self):
        """
        simple method that assembles the border from the two representative polygons
        :return: Shapely MultiPolygon Object
        """
        outer = unary_union(self.geometry["outer"])
        inner = unary_union(self.geometry["inner"])
        return outer.difference(inner)

    def hole_layout(self, text_size=1):
        """
        show the layout of holes in membrane with the text specifying it's index entry
        in the line and geometry dictionaries
        :param text_size: height of the text in microns
        :return:gdshelpers cell that containes the text annotated hole layout
        """
        layout_cell = Cell("Layout")
        layout_cell.add_to_layer(0, self.pixel.boundary.buffer(1))
        for layer, lst in enumerate(self.lines["holes"]):
            for index, point in enumerate(lst):
                text = Text((point.x, point.y), text_size,
                            f"[{layer}][{index}]", 'center-center')
                layout_cell.add_to_layer(layer+1, text)

        return layout_cell

    @staticmethod
    def find_centre(polygon):
        """
        quick method to find the centre of a polygon by creating a bounding box, and calculating that centroid
        :param polygon: Shapely Object we are trying to find the centre of
        :return: coordinate of the polygon centre
        """
        return box(*polygon.bounds).centroid.x, box(*polygon.bounds).centroid.y

    @staticmethod
    def modify_corners(polygon, radius, join_style=1):
        """
        Allows you to round or mitre the corners of the polygon,
        this can be useful to help minimise cracking during fabrication
        :param polygon: Polygon object to be modified
        :param radius: integer or float, the radius of the modify effect
        :param join_style: integer 0 - 2, selector for corner treatment
        :return: A Polygon of the modified pixel
        """
        polygon = polygon.buffer(radius).buffer(-2*radius).buffer(radius, join_style=join_style)
        return polygon

    def fillet_corners(self, radius):
        """
        simple method that allows fillet to be part of a parameter sweep
        :param radius: the size of the round over
        """
        self.fillet = radius

    def chamfer_corners(self, radius):
        """
        simple method that allows fillet to be part of a parameter sweep
        :param radius: the size of the chamfer
        """
        self.chamfer = radius

    @staticmethod
    def remove_curves(polygon, min_distance=2):
        points = list(polygon.exterior.coords)
        s = [Point(p1).distance(Point(p2)) for p1, p2 in zip(points, points[1:])]
        indices = []
        [indices.extend([i, i+1]) for i, v in enumerate(s) if v > min_distance]
        new_points = [points[i] for i in indices]
        return Polygon(new_points)

    @staticmethod
    def extract_flats(polygon, min_distance=5):
        points = list(polygon.exterior.coords)
        s = [Point(p1).distance(Point(p2)) for p1, p2 in zip(points, points[1:])]
        indices = []
        [indices.append([i, i+1]) for i, v in enumerate(s) if v > min_distance]
        flats = [LineString([points[i], points[j]]) for i, j in indices]
        return flats


class Panel:
    def __init__(self):
        self.membranes = []
        self.locations = None
        self.outer = Polygon()
        self.inner = Polygon()
        self.borders = Polygon()
        self.removed = []
        self.resist_tone = "positive"
        self.max_y = 0
        self.max_x = 0
        self.start = (0, 0)
        self._columns = []

    @property
    def rows(self):
        return list(range(np.size(self.membranes, 0)))

    @property
    def columns(self):
        return list(range(np.size(self.membranes, 1)))

    def create_by_spacing(self, membrane, num=(2, 2), x_spacing=50, y_spacing=50, origin=(0, 0)):
        """
        creates 2d array of devices with separation defined as the gap between sides of the membrane devices.
        the panel is constructed using duplicates of the template membrane, allowing the user to create the majority
        of the design from the template.
        :param membrane: Membrane object to be used as the template
        :param num: tuple, number of rows and columns in the panel
        :param x_spacing: the horizontal spacing between inner borders of membrane devices
        :param y_spacing: the vertical spacing between inner borders of the membrane devices
        :param origin: coordinates of the bottom left membrane in the panel
        :return: 2d list of membrane objects
        """
        inner = membrane.geometry["inner"]
        x_min, x_max = inner.bounds[0::2]
        y_min, y_max = inner.bounds[1::2]
        width = x_max - x_min
        height = y_max - y_min
        self.create_by_pitch(membrane, num, width + x_spacing, height + y_spacing, origin)
        return self

    def add_by_spacing(self, membrane, num=(2, 2), x_spacing=50, y_spacing=50, origin=(0, 0)):
        """
        creates 2d array of devices with separation defined as the gap between sides of the membrane devices.
        the panel is constructed using duplicates of the template membrane, allowing the user to create the majority
        of the design from the template.
        :param membrane: Membrane object to be used as the template
        :param num: tuple, number of rows and columns in the panel
        :param x_spacing: the horizontal spacing between inner borders of membrane devices
        :param y_spacing: the vertical spacing between inner borders of the membrane devices
        :param origin: coordinates of the bottom left membrane in the panel
        :return: 2d list of membrane objects
        """
        inner = membrane.geometry["inner"]
        x_min, x_max = inner.bounds[0::2]
        y_min, y_max = inner.bounds[1::2]
        width = x_max - x_min
        height = y_max - y_min
        self.add_by_pitch(membrane, num, width + x_spacing, height + y_spacing, origin)
        return self

    def create_by_pitch(self, membrane, num=(2, 2), x_pitch=200, y_pitch=200, origin=(0, 0)):
        """
        creates 2d array of devices with separation defined as the gap between centres of membrane devices.
        the panel is constructed using duplicates of the template membrane, allowing the user to create the majority
        of the design from the template.
        :param membrane: Membrane object to be used as the template
        :param num: tuple, number of rows and columns in the panel
        :param x_pitch: the horizontal pitch between centres of membrane devices
        :param y_pitch: the vertical pitch between centres of membrane devices
        :param origin: coordinates of the bottom left membrane in the panel
        :return: 2d list of membrane objects
        """
        membrane = copy.deepcopy(membrane)
        membrane.origin = (origin[0], origin[1])
        p = []
        cutouts = []
        borders = []
        """
        self.rows = list(range(num[0]))
        self.columns = list(range(num[1]))
        """
        for c in range(num[0]):
            column = []
            membrane.origin = (origin[0], (c * y_pitch) + origin[1])
            for r in range(num[1]):
                m = copy.deepcopy(membrane)
                column.append(m)
                cutouts.append(m.geometry["inner"])
                borders.append(m.geometry["outer"])
                membrane.move(x_off=x_pitch)

            p.append(column)

        self.max_x = membrane.origin[0]
        self.max_y = membrane.origin[1]
        self.membranes = p
        self.inner = unary_union(cutouts)
        self.borders = unary_union(borders)
        self.outer = self.borders.envelope.difference(self.inner)
        return self

    def add_by_pitch(self, membrane, num=(2, 2), x_pitch=200, y_pitch=200, origin=(0, 0)):
        """
        creates 2d array of devices with separation defined as the gap between centres of membrane devices.
        the panel is constructed using duplicates of the template membrane, allowing the user to create the majority
        of the design from the template.
        :param membrane: Membrane object to be used as the template
        :param num: tuple, number of rows and columns in the panel
        :param x_pitch: the horizontal pitch between centres of membrane devices
        :param y_pitch: the vertical pitch between centres of membrane devices
        :param origin: coordinates of the bottom left membrane in the panel
        :return: 2d list of membrane objects
        """
        membrane = copy.deepcopy(membrane)
        membrane.origin = (origin[0] + self.start[0], origin[1] + self.start[1])
        p = []
        cutouts = []
        borders = []
        """
        self.rows = list(range(num[0]))
        self.columns = list(range(num[1]))
        """
        for c in range(num[0]):
            column = []
            membrane.origin = (origin[0] + self.start[0], origin[1] + self.start[1] + (c * y_pitch))
            for r in range(num[1]):
                m = copy.deepcopy(membrane)
                column.append(m)
                cutouts.append(m.geometry["inner"])
                borders.append(m.geometry["outer"])
                membrane.move(x_off=x_pitch)

            self.membranes.append(column)

        # self.membranes.append(*p)
        self.max_x = membrane.origin[0] - x_pitch
        self.max_y = membrane.origin[1]
        self.inner = unary_union([*cutouts, self.inner])
        self.borders = unary_union([*borders, self.borders])
        self.outer = self.borders.envelope.difference(self.inner)
        return self

    def create_new_column_by_pitch(self, x_pitch=0):
        self.start = (self.max_x + x_pitch, self.start[1])
        return self

    def create_new_row_by_pitch(self, y_pitch=0):
        self.start = (0, self.max_y + y_pitch)
        self._columns = np.size(self.membranes)
        return self

    def function_by_column(self, column, cls_method, args):
        """
        allows a column of devices to be modified all at once, in combination with a loop allows
        for parameter sweeps along the panel
        :param column: index value of the column to be swept
        :param cls_method: method from the membrane class that will be applied to membrane instances in column
        :param args: arguments for class method
        """
        results = []
        for row in self.membranes:
            if isinstance(args, dict):
                results.append(cls_method(row[column], **args))
            else:
                results.append(cls_method(row[column], *args))
        return results

    def function_by_row(self, row, cls_method, args):
        """
        allows a row of devices to be modified all at once, in combination with a loop allows
        for parameter sweeps along the panel
        :param row: index value of the row to be swept
        :param cls_method: method from the membrane class that will be applied to membrane instances in column
        :param args: arguments for class method
        """
        results = []
        for column in self.membranes[row]:
            if isinstance(args, dict):
                results.append(cls_method(column, **args))
            else:
                results.append(cls_method(column, *args))
        return results

    def function_by_position(self, row, column, cls_method, args):
        """
        for individual devices to be modified, in combination with nested loops allows
        for parameter sweeps along the panel
        :param row: index value of the column to be modified
        :param column: index value of the column to be modified
        :param cls_method: method from the membrane class that will be applied to membrane instances in column
        :param args: arguments for class method
        """
        mem = self.membranes[row][column]
        if isinstance(args, dict):
            result = cls_method(mem, **args)
        else:
            result = cls_method(mem, *args)
        return result

    def open_rows(self, left_side=1, right_side=3, end_cap=25):
        """
        creates openings between membrane devices that allow for less restricted flow of wet etch,
        as a trade off for less anchor contact points
        :param left_side: the index value of the left side of inner border
        :param right_side: the index value of the right side of inner border
        """
        rights = []
        lefts = []
        for column in self.membranes:
            r = []
            l = []
            for row in column:
                temp = self.grab_sides(row, right_side, left_side)
                r.append(temp[0])
                l.append(temp[1])
                self.removed.append(LineString(temp[0]).buffer(end_cap, single_sided=True, cap_style=CAP_STYLE.square))
                self.removed.append(LineString(temp[1]).buffer(end_cap, single_sided=True, cap_style=CAP_STYLE.square))
            rights.append(r)
            lefts.append(l)

        """openings = [LinearRing([*r_p, *l_p]) for r, l in zip(rights, lefts) for r_p, l_p in zip(r[:-1], l[1:])]
        ends = [
            LineString(r[-1]).buffer(end_cap, single_sided=True, cap_style=CAP_STYLE.square),
            LineString(l[0]).buffer(end_cap, single_sided=True, cap_style=CAP_STYLE.square)
        ]
        self.removed.extend([Polygon(o) for o in openings if o.is_ccw])
        self.removed.extend(ends)"""

        """openings = []
        for r, l in zip(lefts, rights):
            openings.extend(
                [LineString(l[-1]).buffer(end_cap, single_sided=True, cap_style=CAP_STYLE.square),
                 LineString(r[0]).buffer(end_cap, single_sided=True, cap_style=CAP_STYLE.square)]
            )
            for l_p, r_p in zip(l[:-1], r[1:]):
                openings.append(Polygon([*l_p, *r_p]))

        self.removed.extend(openings)"""

    def open_columns(self, top_side=2, bottom_side=0, end_cap=25):
        """
        creates openings between membrane devices that allow for less restricted flow of wet etch,
        as a trade off for less anchor contact points
        :param top_side: the index value of the top side of inner border
        :param bottom_side: the index value of the bottom side of inner border
        """
        temp = np.array(self.membranes)
        temp = temp.transpose().tolist()
        tops = []
        bottoms = []
        for column in temp:
            t = []
            b = []
            for row in column:
                temp = self.grab_sides(row, top_side, bottom_side)
                t.append(temp[0])
                b.append(temp[1])
                self.removed.append(LineString(temp[0]).buffer(end_cap, single_sided=True, cap_style=CAP_STYLE.square))
                self.removed.append(LineString(temp[1]).buffer(end_cap, single_sided=True, cap_style=CAP_STYLE.square))
            tops.append(t)
            bottoms.append(b)

        """openings = [LinearRing([*t_p, *b_p]) for t, b in zip(tops, bottoms) for t_p, b_p in zip(t[:-1], b[1:])]
        ends = [
            LineString(t[-1]).buffer(end_cap, single_sided=True, cap_style=CAP_STYLE.square),
            LineString(b[0]).buffer(end_cap, single_sided=True, cap_style=CAP_STYLE.square)
        ]"""
        """openings = []
        for t, b in zip(tops, bottoms):
            openings.extend(
                [LineString(t[-1]).buffer(end_cap, single_sided=True, cap_style=CAP_STYLE.square),
                 LineString(b[0]).buffer(end_cap, single_sided=True, cap_style=CAP_STYLE.square)]
            )
            for t_p, b_p in zip(t[:-1], b[1:]):
                openings.append(Polygon([*t_p, *b_p]))

        self.removed.extend(openings)
        # self.removed.extend(ends)"""
        return

    @staticmethod
    def grab_sides(membrane, side_1, side_2, key="inner"):
        """
        grabs the coordinates of a vertices stored in the geometry dictionary in the membrane class
        :param membrane: membrane instance to access
        :param side_1: index of first vertice to access
        :param side_2: index of second vertice to access
        :param key: key for geometry dictionary to be accessed
        :return: tuple of side 1 and 2 coordinates
        """
        sides_1 = membrane.geometry[key].exterior.coords[side_1:side_1 + 2]
        sides_2 = membrane.geometry[key].exterior.coords[side_2:side_2 + 2]
        return sides_1, sides_2

    def device_layer(self, device=True, pixel=True, border=True, anchors=True, holes=True, removed=True,
                     holes_on_shallow=False):
        """
        creates the layer of the membrane that contains the photonics device,
        allows users to toggle what geometries are include in this layer to customise design process
        :param device: True or False
        :param pixel: True or False
        :param border: True or False
        :param anchors: True or False
        :param holes: True or False
        :param removed: True or False
        :return: a Shapely construct of the membrane panel
        """
        if self.resist_tone == "positive":
            lst = [unary_union(self.removed)] if removed and self.removed else []
        else:
            if border is False:
                lst = []
            else:
                lst = [self.outer.difference(unary_union(self.removed))] if removed and self.removed else [self.outer]

        for row in self.membranes:
            for pos in row:
                pos.resist_tone = self.resist_tone
                lst.append(pos.device_layer(device, pixel, border, anchors, holes, False, holes_on_shallow))

        return unary_union(lst)

    def get_device_layer(self, position):
        position.resist_tone = self.resist_tone
        return position.device_layer(True, True, False, True, True, False)

    def membrane(self, pixel=True, border=True, anchors=True, holes=True, removed=True, holes_on_shallow=False):
        """
        caller function, simplifies add-ons by allowing membrane and panel non-device layers to be called the same thing
        """
        return self.panel(pixel=pixel, border=border, anchors=anchors, holes=holes, removed=removed,
                          holes_on_shallow=holes_on_shallow)

    def panel(self, pixel=True, border=True, anchors=True, holes=True, removed=True, holes_on_shallow=False):
        """
        creates the layer of the membrane that does not contain the photonics device,
        allows users to toggle what geometries are included in this layer to customise design process
        :param pixel: True or False
        :param border: True or False
        :param anchors: True or False
        :param holes: True or False
        :param removed: True or False
        :return: a Shapely construct of the membrane panel
        """
        if self.resist_tone == "positive":
            lst = [unary_union(self.removed)] if removed and self.removed else []
        else:
            if border is False:
                lst = []
            else:
                lst = [self.outer.difference(unary_union(self.removed))] if removed and self.removed else [self.outer]

        for row in self.membranes:
            for pos in row:
                pos.resist_tone = self.resist_tone
                lst.append(pos.membrane(pixel, border, anchors, holes, False, holes_on_shallow))
        return unary_union(lst)


class MembraneRing:
    def __init__(self):
        self.origin = (0, 0)
        self.angle = 0
        self.ring_width = 1
        self.guide_width = 1

        self.ring = Waveguide
        self.input_guide = Waveguide
        self.add_guide = Waveguide

        self.in_port = Port
        self.out_port = Port
        self.add_port = Port
        self.drop_port = Port
        self.all_ports = []

    def create_ring(self, radius, race_lengths, gaps):
        ring_guide = Waveguide(origin=self.origin, angle=self.angle, width=self.ring_width)

        ports = []
        lengths = self.right_length(race_lengths, 2, 0)*2
        gaps = self.right_length(gaps, 2, gaps)*2

        for length, gap in zip(lengths, gaps):
            off = gap + (self.ring_width + self.guide_width) / 2
            ports.append(ring_guide.port.parallel_offset(offset=-off))
            if length:
                ring_guide.add_straight_segment(length)
            ports.append(ring_guide.port.parallel_offset(offset=-off))
            ring_guide.add_bend(pi*0.5, radius)

        self.ring = ring_guide
        self.origin = (ring_guide.get_shapely_object().centroid.x, ring_guide.get_shapely_object().centroid.y)

        self.all_ports = ports
        self.in_port = ports[0]
        self.out_port = ports[1]
        self.add_port = ports[4]
        self.drop_port = ports[5]

    def create_input_guide(self, lengths, separation, offset=0.0, taper_length=None, final_width=None):
        wg = self._create_coupled_guide_2(self.in_port, lengths, separation, offset, taper_length, final_width)
        self.input_guide = wg

    def create_add_guide(self, lengths, separation, offset=0.0, taper_length=None, final_width=None):
        wg = self._create_coupled_guide_2(self.add_port, lengths, separation, offset, taper_length, final_width)
        self.add_guide = wg

    def _create_coupled_guide(self, port, coupling_length, bend_angle, bend_radius, total_length,
                              offset=0.0, taper_length=None, final_width=None):
        if taper_length and final_width:
            off_port = port.longitudinal_offset(-taper_length + offset)
            wg = Waveguide(off_port.origin, off_port.angle, final_width)
            wg.ring_width = final_width
            wg.add_straight_segment(taper_length, self.guide_width)
            tl = total_length - taper_length
        else:
            off_port = port.longitudinal_offset(offset)
            wg = Waveguide(off_port.origin, off_port.angle, self.guide_width)
            tl = total_length

        wg.add_straight_segment(coupling_length)
        wg.add_bend(-bend_angle, bend_radius)
        wg.add_bend(bend_angle, bend_radius)
        wg.add_straight_segment(tl-wg.length)

        if taper_length and final_width:
            wg.add_straight_segment(taper_length, final_width)
        return wg

    def _create_coupled_guide_2(self, port, lengths, separation, offset=0.0, taper_length=None, final_width=None):

        if isinstance(separation, list) is False:
            separation = [separation] * 2
        if len(separation) < 2:
            separation = separation * 2
        if isinstance(lengths, list) is False:
            lengths = [lengths] * 2
        if len(lengths) < 2:
            lengths = lengths * 2

        if taper_length and final_width:
            off_port = port.longitudinal_offset(-taper_length - lengths[0] - separation[0] + offset)\
                .parallel_offset(-separation[0])
            wg = Waveguide(off_port.origin, off_port.angle, final_width)
            wg.ring_width = final_width
            wg.add_straight_segment(taper_length, self.guide_width)
        else:
            off_port = port.longitudinal_offset(- lengths[0] - separation[0] + offset)\
                .parallel_offset(-separation[0])
            wg = Waveguide(off_port.origin, off_port.angle, self.guide_width)

        radius = min(separation)/2

        wg.add_straight_segment(lengths[0])
        wg.add_bend(pi / 2, radius)
        wg.add_straight_segment(separation[0] - 2 * radius)
        wg.add_bend(-pi / 2, radius)
        wg.add_straight_segment(lengths[1])
        wg.add_bend(-pi / 2, radius)
        wg.add_straight_segment(separation[1] - 2 * radius)
        wg.add_bend(pi / 2, radius)
        wg.add_straight_segment(lengths[2])

        if taper_length and final_width:
            wg.add_straight_segment(taper_length, final_width)
        return wg

    def move(self, x_off, y_off):
        self.origin = (self.origin[0] + x_off, self.origin[1] + y_off)

    def right_length(self, lst, length, filler):
        """ makes sure list has the correct number of entries"""
        if isinstance(lst, list) is False:
            lst = [lst]
        if len(lst) < length:
            lst.append(filler)
            return self.right_length(lst, length, filler)
        elif len(lst) > length:
            lst = lst[:2]
            return self.right_length(lst, length, filler)
        else:
            return lst
