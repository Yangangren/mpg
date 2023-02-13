#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: endtoend_env_utils.py
# =====================================

import math
import os
import numpy as np
from collections import OrderedDict
from shapely import geometry


class Para:
    # MAP
    L, W = 4.8, 2.0
    LANE_WIDTH_1 = 3.75
    LANE_WIDTH_2 = 3.25
    LANE_WIDTH_3 = 4.00
    WALK_WIDTH = 6.00
    GREEN_BELT_LAT = 10
    GREEN_BELT_LON = 2
    BIKE_LANE_WIDTH = 1.0
    PERSON_LANE_WIDTH = 2.0

    OFFSET_L = -3
    OFFSET_R = -7
    OFFSET_U = 1
    OFFSET_D = -0.38

    LANE_NUMBER_LON_IN = 3
    LANE_NUMBER_LON_OUT = 2
    LANE_NUMBER_LAT_IN = 4
    LANE_NUMBER_LAT_OUT = 3

    CROSSROAD_SIZE_LAT = 64
    CROSSROAD_SIZE_LON = 76
    CROSSROAD_INTER = [(-13.38, -38.00), (-32.00, -21.00), (-32.00, 21.25), (-13.30, 38.00),
                       (14.10, 38.00), (32.00, 21.05), (32.00, -21.25), (14.00, -38.00)]

    # DIM
    EGO_ENCODING_DIM = 8
    TRACK_ENCODING_DIM = 4
    ROAD_ENCODING_DIM = 2
    LIGHT_ENCODING_DIM = 2
    TASK_ENCODING_DIM = 3
    PER_OTHER_INFO_DIM = 10

    # MAX NUM
    MAX_VEH_NUM = 6  # to be align with VEHICLE_MODE_DICT
    MAX_BIKE_NUM = 4  # to be align with BIKE_MODE_DICT
    MAX_PERSON_NUM = 4  # to be align with PERSON_MODE_DICT
    MAX_TRAFFIC = 400   # to be align with sensor_module and Sensors.so
    # TRAFFIC LIGHT
    YELLOW_TIME = 3.  # unit: s

    # NOISE
    # (v_x, v_y, r, x, y, phi) for ego
    # (x, y, v, phi, l, w; type encoding (d=3), turn rad) for other
    EGO_MEAN = np.array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
    EGO_VAR = np.diag([0.0418, 0.0418, 0., 0.0245, 0.0227, 0.0029*(180./np.pi)**2, 0., 0.]).astype(np.float32)

    VEH_MEAN = np.tile(np.zeros((PER_OTHER_INFO_DIM,), dtype=np.float32), MAX_VEH_NUM)
    VEH_VAR = np.tile(np.array([0.0245, 0.0227, 0.0418, 0.0029*(180./np.pi)**2, 0.0902, 0.0202, 0., 0., 0., 0.,], dtype=np.float32), MAX_VEH_NUM)

    BIKE_MEAN = np.tile(np.zeros((PER_OTHER_INFO_DIM,), dtype=np.float32), MAX_BIKE_NUM)
    BIKE_VAR = np.tile(np.array([0.172**2, 0.1583**2, 0.1763**2, (0.1707*180./np.pi)**2, 0.1649**2, 0.1091**2, 0., 0., 0., 0.,], dtype=np.float32), MAX_BIKE_NUM)

    PERSON_MEAN = np.tile(np.zeros((PER_OTHER_INFO_DIM,), dtype=np.float32), MAX_PERSON_NUM)
    PERSON_VAR = np.tile(np.array([0.1102**2, 0.1108**2, 0.1189**2, (0.2289*180./np.pi)**2, 0.1468**2, 0.1405**2, 0., 0., 0., 0.,], dtype=np.float32), MAX_PERSON_NUM)

    OTHERS_MEAN = np.concatenate([BIKE_MEAN, PERSON_MEAN, VEH_MEAN], axis=-1) # order determined in line 735 in e2e.py
    OTHERS_VAR = np.diag(np.concatenate([BIKE_VAR, PERSON_VAR, VEH_VAR], axis=-1)).astype(np.float32)

    adv_act_bound = dict(high=[0.0052, 0.0059, 0.0805, 0.0064] * MAX_BIKE_NUM +
                              [0.0136, 0.0143, 0.1815, 0.1092] * MAX_PERSON_NUM +
                              [0.0125, 0.0182, 0.2253, 0.0160] * MAX_VEH_NUM,
                         low=[-0.0052, -0.0058, -0.0779, -0.0066] * MAX_BIKE_NUM +
                             [-0.0136, -0.0144, -0.2135, -0.1096] * MAX_PERSON_NUM +
                             [-0.0134, -0.0179, -0.2254, -0.0159] * MAX_VEH_NUM)  # x, y, v, phi (in rad)


LIGHT_PHASE_TO_GREEN_OR_RED = {0: 'green', 1: 'green', 2: 'red', 3: 'red', 4: 'red', 5: 'red', 6: 'red', 7: 'red',
                               8: 'red', 9: 'red'}  # 0: green, 1: red
TASK_ENCODING = dict(left=[1.0, 0.0, 0.0], straight=[0.0, 1.0, 0.0], right=[0.0, 0.0, 1.0])
LIGHT_ENCODING = {0: [1.0, 0.0], 1: [1.0, 0.0], 2: [1.0, 1.0], 3: [0.0, 1.0], 4: [0.0, 1.0], 5: [0.0, 1.0],
                  6: [0.0, 1.0], 7: [0.0, 1.0], 8: [0.0, 1.0], 9: [0.0, 1.0]}


ROUTE2MODE = {('1o', '2i'): 'dr', ('1o', '3i'): 'du', ('1o', '4i'): 'dl',
              ('2o', '1i'): 'rd', ('2o', '3i'): 'ru', ('2o', '4i'): 'rl',
              ('3o', '1i'): 'ud', ('3o', '2i'): 'ur', ('3o', '4i'): 'ul',
              ('4o', '1i'): 'ld', ('4o', '2i'): 'lr', ('4o', '3i'): 'lu'}

MODE2TASK = {'dr': 'right', 'du': 'straight', 'dl': 'left',
             'rd': 'left', 'ru': 'right', 'rl': ' straight',
             'ud': 'straight', 'ur': 'left', 'ul': 'right',
             'ld': 'right', 'lr': 'straight', 'lu': 'left',
             'ud_b': 'straight', 'du_b': 'straight', 'lr_b': 'straight',
             'c1': 'straight', 'c2': 'straight', 'c3': 'straight'}

TASK2ROUTEID = {'left': 'dl', 'straight': 'du', 'right': 'dr'}

MODE2ROUTE = {'dr': ('1o', '2i'), 'du': ('1o', '3i'), 'dl': ('1o', '4i'),
              'rd': ('2o', '1i'), 'ru': ('2o', '3i'), 'rl': ('2o', '4i'),
              'ud': ('3o', '1i'), 'ur': ('3o', '2i'), 'ul': ('3o', '4i'),
              'ld': ('4o', '1i'), 'lr': ('4o', '2i'), 'lu': ('4o', '3i')}


def judge_feasible(orig_x, orig_y, task):  # map dependant
    def is_in_straight_before1(orig_x, orig_y):
        return Para.OFFSET_D < orig_x < Para.LANE_WIDTH_2 + Para.OFFSET_D and orig_y <= -Para.CROSSROAD_SIZE_LON / 2

    def is_in_straight_before2(orig_x, orig_y):
        return Para.LANE_WIDTH_2 + Para.OFFSET_D < orig_x < Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 + Para.OFFSET_D and orig_y <= -Para.CROSSROAD_SIZE_LON / 2

    def is_in_straight_before3(orig_x, orig_y):
        return Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 + Para.OFFSET_D < orig_x < Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 * 2 + Para.OFFSET_D and orig_y <= -Para.CROSSROAD_SIZE_LON / 2

    def is_in_straight_after(orig_x, orig_y):
        return Para.OFFSET_U + Para.GREEN_BELT_LON < orig_x < Para.OFFSET_U + Para.GREEN_BELT_LON + Para.LANE_WIDTH_3 * 2 and orig_y >= Para.CROSSROAD_SIZE_LON / 2

    def is_in_left(orig_x, orig_y):
        return Para.OFFSET_L + Para.GREEN_BELT_LAT < orig_y < Para.OFFSET_L + Para.GREEN_BELT_LAT + Para.LANE_WIDTH_1 * 3 and orig_x < -Para.CROSSROAD_SIZE_LAT / 2

    def is_in_right(orig_x, orig_y):
        return Para.OFFSET_R - Para.LANE_WIDTH_1 * 3 < orig_y < Para.OFFSET_R and orig_x > Para.CROSSROAD_SIZE_LAT / 2

    def is_in_middle(orig_x, orig_y):
        return True if -Para.CROSSROAD_SIZE_LON / 2 < orig_y < Para.CROSSROAD_SIZE_LON / 2 and -Para.CROSSROAD_SIZE_LAT / 2 < orig_x < Para.CROSSROAD_SIZE_LAT / 2 else False

    if task == 'left':
        return True if is_in_straight_before1(orig_x, orig_y) or is_in_left(orig_x, orig_y) \
                       or is_in_middle(orig_x, orig_y) else False
    elif task == 'straight':
        return True if is_in_straight_before2(orig_x, orig_y) or is_in_straight_after(
            orig_x, orig_y) or is_in_middle(orig_x, orig_y) else False
    else:
        assert task == 'right'
        return True if is_in_straight_before3(orig_x, orig_y) or is_in_right(orig_x, orig_y) \
                       or is_in_middle(orig_x, orig_y) else False


def shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param coordi_shift_x: coordi_shift_x along x axis
    :param coordi_shift_y: coordi_shift_y along y axis
    :return: shifted_x, shifted_y
    """
    shifted_x = orig_x - coordi_shift_x
    shifted_y = orig_y - coordi_shift_y
    return shifted_x, shifted_y


def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """

    coordi_rotate_d_in_rad = coordi_rotate_d * math.pi / 180
    transformed_x = orig_x * math.cos(coordi_rotate_d_in_rad) + orig_y * math.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * math.sin(coordi_rotate_d_in_rad) + orig_y * math.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    if transformed_d > 180:
        while transformed_d > 180:
            transformed_d = transformed_d - 360
    elif transformed_d <= -180:
        while transformed_d <= -180:
            transformed_d = transformed_d + 360
    else:
        transformed_d = transformed_d
    return transformed_x, transformed_y, transformed_d

def rotate_coordination_vec(orig_x, orig_y, orig_d, coordi_rotate_d):
    coordi_rotate_d_in_rad = coordi_rotate_d * np.pi / 180
    transformed_x = orig_x * np.cos(coordi_rotate_d_in_rad) + orig_y * np.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * np.sin(coordi_rotate_d_in_rad) + orig_y * np.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    transformed_d = np.where(transformed_d>180, transformed_d - 360, transformed_d)
    transformed_d = np.where(transformed_d<=-180, transformed_d + 360, transformed_d)
    return transformed_x, transformed_y, transformed_d


def shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


def rotate_and_shift_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y, transformed_d \
        = rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d)
    transformed_x, transformed_y = shift_coordination(shift_x, shift_y, coordi_shift_x, coordi_shift_y)

    return transformed_x, transformed_y, transformed_d


def cal_info_in_transform_coordination(filtered_objects, x, y, rotate_d):  # rotate_d is positive if anti
    results = []
    for obj in filtered_objects:
        orig_x = obj['x']
        orig_y = obj['y']
        orig_v = obj['v']
        orig_heading = obj['phi']
        width = obj['w']
        length = obj['l']
        route = obj['route']
        shifted_x, shifted_y = shift_coordination(orig_x, orig_y, x, y)
        trans_x, trans_y, trans_heading = rotate_coordination(shifted_x, shifted_y, orig_heading, rotate_d)
        trans_v = orig_v
        results.append({'x': trans_x,
                        'y': trans_y,
                        'v': trans_v,
                        'phi': trans_heading,
                        'w': width,
                        'l': length,
                        'route': route, })
    return results


def cal_ego_info_in_transform_coordination(ego_dynamics, x, y, rotate_d):
    orig_x, orig_y, orig_a, corner_points = ego_dynamics['x'], ego_dynamics['y'], ego_dynamics['phi'], ego_dynamics[
        'Corner_point']
    shifted_x, shifted_y = shift_coordination(orig_x, orig_y, x, y)
    trans_x, trans_y, trans_a = rotate_coordination(shifted_x, shifted_y, orig_a, rotate_d)
    trans_corner_points = []
    for corner_x, corner_y in corner_points:
        shifted_x, shifted_y = shift_coordination(corner_x, corner_y, x, y)
        trans_corner_x, trans_corner_y, _ = rotate_coordination(shifted_x, shifted_y, orig_a, rotate_d)
        trans_corner_points.append((trans_corner_x, trans_corner_y))
    ego_dynamics.update(dict(x=trans_x,
                             y=trans_y,
                             phi=trans_a,
                             Corner_point=trans_corner_points))
    return ego_dynamics


def xy2_edgeID_lane(x, y):
    if y < -Para.CROSSROAD_SIZE_LON / 2:
        edgeID = '1o'
        if x <= Para.OFFSET_D + Para.LANE_WIDTH_2:
            lane = 4
        else:
            lane = int(Para.LANE_NUMBER_LON_IN - int((x - Para.OFFSET_D - Para.LANE_WIDTH_2) / Para.LANE_WIDTH_3))
    elif x < -Para.CROSSROAD_SIZE_LAT / 2:
        edgeID = '4i'
        lane = int((Para.LANE_NUMBER_LAT_OUT + 1) - int((y - Para.OFFSET_L) / Para.LANE_WIDTH_1))
    elif y > Para.CROSSROAD_SIZE_LON / 2:
        edgeID = '3i'
        lane = 3 if x <= Para.OFFSET_U + Para.LANE_WIDTH_2 else 2
    elif x > Para.CROSSROAD_SIZE_LAT / 2:
        edgeID = '2i'
        lane = int((Para.LANE_NUMBER_LAT_OUT + 1) - int(-(y - Para.OFFSET_R) / Para.LANE_WIDTH_1))
    else:
        edgeID = '0'
        lane = 0
    return edgeID, lane


def _convert_car_coord_to_sumo_coord(x_in_car_coord, y_in_car_coord, a_in_car_coord, car_length):  # a in deg
    x_in_sumo_coord = x_in_car_coord + car_length / 2 * math.cos(math.radians(a_in_car_coord))
    y_in_sumo_coord = y_in_car_coord + car_length / 2 * math.sin(math.radians(a_in_car_coord))
    a_in_sumo_coord = -a_in_car_coord + 90.
    return x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord


def _convert_sumo_coord_to_car_coord(x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord, car_length):
    a_in_car_coord = -a_in_sumo_coord + 90.
    x_in_car_coord = x_in_sumo_coord - (math.cos(a_in_car_coord / 180. * math.pi) * car_length / 2)
    y_in_car_coord = y_in_sumo_coord - (math.sin(a_in_car_coord / 180. * math.pi) * car_length / 2)
    return x_in_car_coord, y_in_car_coord, deal_with_phi(a_in_car_coord)


def deal_with_phi(phi):
    while phi > 180:
        phi -= 360
    while phi <= -180:
        phi += 360
    return phi


if __name__ == '__main__':
    from matplotlib import pyplot
    from shapely.geometry import Polygon, Point, LineString
    from descartes.patch import PolygonPatch
    from shapely.figures import BLUE, SIZE, set_limits, plot_coords, color_isvalid
    import figures

    fig = pyplot.figure(1, figsize=SIZE, dpi=90)

    # 1: valid polygon
    ax = fig.add_subplot(121)

    ext = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
    int = [(1, 0), (0.5, 0.5), (1, 1), (1.5, 0.5), (1, 0)][::-1]
    polygon = Polygon(ext, [int])

    plot_coords(ax, polygon.interiors[0])
    plot_coords(ax, polygon.exterior)

    patch = PolygonPatch(polygon, facecolor=color_isvalid(polygon), edgecolor=color_isvalid(polygon, valid=BLUE),
                         alpha=0.5, zorder=2)
    ax.add_patch(patch)

    ax.set_title('a) valid')

    set_limits(ax, -1, 3, -1, 3)

    # 2: invalid self-touching ring
    ax = fig.add_subplot(122)
    ext = [(0, 0), (0, 2), (2, 2), (2, 0), (0, 0)]
    int = [(1, 0), (0, 1), (0.5, 1.5), (1.5, 0.5), (1, 0)][::-1]
    polygon = Polygon(ext, [int])

    plot_coords(ax, polygon.interiors[0])
    plot_coords(ax, polygon.exterior)

    patch = PolygonPatch(polygon, facecolor=color_isvalid(polygon), edgecolor=color_isvalid(polygon, valid=BLUE),
                         alpha=0.5, zorder=2)
    ax.add_patch(patch)

    ax.set_title('b) invalid')

    set_limits(ax, -1, 3, -1, 3)

    pyplot.pause(5.0)

    # solve the intersections of line and polygon
    p1 = Point(5.414213562373095, 2.585786437626905)
    p2 = Point(15.17279752753168, -7.172797527531679)
    p3 = Point(2.2, 5.9999999)

    l = LineString([p1, p2])

    l1 = LineString([(2, 2), (2, 6)])
    l2 = LineString([(2, 6), (6, 6)])
    l3 = LineString([(6, 6), (6, 2)])
    l4 = LineString([(6, 2), (2, 2)])
    l5 = LineString([(3, 3), (4, 4)])
    l6 = LineString([(1, 1), (7, 7)])

    sp = Polygon([(2, 2), (2, 6), (6, 6), (6, 2)])

    print("Polygon contains p1:", sp.contains(p1))
    print("Polygon contains p2:", sp.contains(p2))
    print("Polygon contains p3:", sp.contains(p3))

    for i, line in enumerate((l1, l2, l3, l4, l5, l6)):
        res1 = line.intersects(sp)
        res2 = line.intersection(sp)
        temp = np.asarray(res2.coords.xy)
        print("Line {} intersects l1: {} {}".format(i, res1, temp))