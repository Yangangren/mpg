#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: traffic.py
# =====================================

import copy
import math
import os
import random
import sys
from collections import defaultdict
from math import fabs, cos, sin, pi
from shapely.geometry import Polygon, Point, LineString

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from sumolib import checkBinary
import traci
from traci.exceptions import FatalTraCIError, TraCIException
from env_build.endtoend_env_utils import shift_and_rotate_coordination, _convert_car_coord_to_sumo_coord, \
    _convert_sumo_coord_to_car_coord, xy2_edgeID_lane, TASK2ROUTEID

SUMO_BINARY = checkBinary('sumo')
SIM_PERIOD = 1.0 / 10


class Traffic(object):
    def __init__(self, step_length, mode, init_n_ego_dict):  # mode 'display' or 'training'
        self.random_traffic = None
        self.sim_time = 0
        self.n_ego_vehicles = defaultdict(list)
        self.step_length = step_length
        self.step_time_str = str(float(step_length) / 1000)
        self.collision_flag = False
        self.n_ego_collision_flag = {}
        self.collision_ego_id = None
        self.v_light = None
        self.n_ego_dict = init_n_ego_dict
        self.light_phase = None
        self.mode = mode
        self.cross_num = None
        self.net = None

    def reset(self, cross_num=1):
        self.net = sumolib.net.readNet(os.path.dirname(__file__) + "/sumo_files/" + "cross_" + str(cross_num) + '/' + 'a.net.xml')
        try:
            traci.start([SUMO_BINARY, "-c",
                         os.path.dirname(__file__) + "/sumo_files/" + "cross_" + str(cross_num) + "/cross.sumocfg",
                         "--step-length", self.step_time_str,
                         # "--lateral-resolution", "3.5",
                         "--random",
                         # "--start",
                         # "--quit-on-end",
                         "--no-warnings",
                         "--no-step-log",
                         # '--seed', str(int(seed))
                         ], numRetries=5)  # '--seed', str(int(seed))

        # Traci connection is already active
        except TraCIException:
            traci.close()
            traci.start([SUMO_BINARY, "-c",
                         os.path.dirname(__file__) + "/sumo_files/" + "cross_" + str(cross_num) + "/cross.sumocfg",
                         "--step-length", self.step_time_str,
                         # "--lateral-resolution", "3.5",
                         "--random",
                         # "--start",
                         # "--quit-on-end",
                         "--no-warnings",
                         "--no-step-log",
                         # '--seed', str(int(seed))
                         ], numRetries=5)  # '--seed', str(int(seed))

        except FatalTraCIError:
            print('Retry by other port')
            port = sumolib.miscutils.getFreeSocketPort()
            traci.start(
                [SUMO_BINARY, "-c",
                 os.path.dirname(__file__) + "/sumo_files/" + "cross_" + str(cross_num) + "/cross.sumocfg",
                 "--step-length", self.step_time_str,
                 "--lateral-resolution", "3.5",
                 "--random",
                 # "--start",
                 # "--quit-on-end",
                 "--no-warnings",
                 "--no-step-log",
                 # '--seed', str(int(seed))
                 ], port=port, numRetries=5)  # '--seed', str(int(seed))

        traci.junction.subscribeContext(objectID='0', domain=traci.constants.CMD_GET_VEHICLE_VARIABLE, dist=10000.0,
                                        varIDs=[traci.constants.VAR_POSITION,
                                                traci.constants.VAR_LENGTH,
                                                traci.constants.VAR_WIDTH,
                                                traci.constants.VAR_ANGLE,
                                                traci.constants.VAR_SIGNALS,
                                                traci.constants.VAR_SPEED,
                                                traci.constants.VAR_SPEED_LAT,
                                                traci.constants.VAR_TYPE,
                                                # traci.constants.VAR_EMERGENCY_DECEL,
                                                # traci.constants.VAR_LANE_INDEX,
                                                # traci.constants.VAR_LANEPOSITION,
                                                # traci.constants.VAR_EDGES,
                                                # traci.constants.VAR_ROAD_ID,
                                                traci.constants.VAR_EDGES,
                                                # traci.constants.VAR_NEXT_EDGE,
                                                # traci.constants.VAR_ROUTE_INDEX
                                                ], begin=0.0, end=2147483647.0)

        traci.junction.subscribeContext(objectID='0', domain=traci.constants.CMD_GET_PERSON_VARIABLE, dist=10000.0,
                                        varIDs=[traci.constants.VAR_POSITION,
                                                traci.constants.VAR_LENGTH,
                                                traci.constants.VAR_WIDTH,
                                                traci.constants.VAR_ANGLE,
                                                # traci.constants.VAR_SIGNALS,
                                                traci.constants.VAR_SPEED,
                                                traci.constants.VAR_TYPE,
                                                # traci.constants.VAR_EMERGENCY_DECEL,
                                                # traci.constants.VAR_LANE_INDEX,
                                                # traci.constants.VAR_LANEPOSITION,
                                                # traci.constants.VAR_EDGES,
                                                traci.constants.VAR_ROAD_ID,
                                                # traci.constants.VAR_NEXT_EDGE,
                                                # traci.constants.VAR_ROUTE_ID,
                                                # traci.constants.VAR_ROUTE_INDEX
                                                ], begin=0.0, end=2147483647.0)

        # This step aims to create different initial conditions by setting traffic lights
        total_step = random.randint(50, 350)
        while traci.simulation.getTime() < total_step:
            if traci.simulation.getTime() < total_step - 1:
                traci.trafficlight.setPhase('0', 2)     # todo
            else:
                traci.trafficlight.setPhase('0', 0)     # todo
            traci.simulationStep()

    def close(self):
        traci.close()

    def add_self_car(self, n_ego_dict, with_delete=True):
        for egoID, ego_dict in n_ego_dict.items():
            ego_v_x = ego_dict['v_x']
            ego_v_y = ego_dict['v_y']
            ego_l = ego_dict['l']
            ego_x = ego_dict['x']
            ego_y = ego_dict['y']
            ego_phi = ego_dict['phi']
            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi, ego_l)
            keeproute = 0
            if with_delete:
                try:
                    traci.vehicle.remove(egoID)
                except traci.exceptions.TraCIException:
                    # print('Don\'t worry, it\'s been handled well')
                    pass
                traci.simulationStep()
                traci.vehicle.addLegacy(vehID=egoID, routeID=ego_dict['routeID'], typeID='self_car')
            traci.vehicle.moveToXY(egoID, -1, -1, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keepRoute=keeproute)
            traci.vehicle.setLength(egoID, ego_dict['l'])
            traci.vehicle.setWidth(egoID, ego_dict['w'])
            traci.vehicle.setSpeed(egoID, math.sqrt(ego_v_x ** 2 + ego_v_y ** 2))

    def generate_random_traffic(self):
        random_traffic = traci.junction.getContextSubscriptionResults('0')
        random_traffic = copy.deepcopy(random_traffic)

        for ego_id in self.n_ego_dict.keys():
            if ego_id in random_traffic:
                del random_traffic[ego_id]
        return random_traffic

    def init_light(self):
        if self.mode == 'training' or self.mode == 'evaluating':
            if random.random() > 0.7:
                self.light_phase = 4
            else:
                self.light_phase = 0
            traci.trafficlight.setPhase('0', self.light_phase)
            # traci.trafficlight.setPhaseDuration('0', 10000)
            traci.simulationStep()
            self._get_traffic_light()
        else:
            # traci.trafficlight.setPhase('0', 1)
            # traci.trafficlight.setPhaseDuration('0', 2)
            # traci.simulationStep()
            self._get_traffic_light()
        return self.v_light

    def get_light_duration(self):
        return traci.trafficlight.getNextSwitch('0')

    def init_traffic(self, init_n_ego_dict, training_task):
        self.training_task = training_task
        self.sim_time = 0
        self.n_ego_vehicles = defaultdict(list)
        self.collision_flag = False
        self.n_ego_collision_flag = {}
        self.collision_ego_id = None
        self.n_ego_dict = init_n_ego_dict
        self.add_self_car(init_n_ego_dict)
        traci.simulationStep()
        random_traffic = self.generate_random_traffic()
        self.add_self_car(init_n_ego_dict, with_delete=False)

        # move ego to the given position and remove conflict cars
        for egoID, ego_dict in self.n_ego_dict.items():
            ego_x, ego_y, ego_v_x, ego_v_y, ego_phi, ego_l, ego_w = ego_dict['x'], ego_dict['y'], ego_dict['v_x'], \
                                                                    ego_dict['v_y'], ego_dict['phi'], ego_dict['l'], \
                                                                    ego_dict['w']
            for veh in random_traffic:
                x_in_sumo, y_in_sumo = random_traffic[veh][traci.constants.VAR_POSITION]
                a_in_sumo = random_traffic[veh][traci.constants.VAR_ANGLE]
                veh_l = random_traffic[veh][traci.constants.VAR_LENGTH]
                veh_v = random_traffic[veh][traci.constants.VAR_SPEED]
                veh_type = random_traffic[veh][traci.constants.VAR_TYPE]
                # veh_sig = random_traffic[veh][traci.constants.VAR_SIGNALS]

                x, y, a = _convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, veh_l)
                x_in_ego_coord, y_in_ego_coord, a_in_ego_coord = shift_and_rotate_coordination(x, y, a, ego_x,
                                                                                               ego_y, ego_phi)
                ego_x_in_veh_coord, ego_y_in_veh_coord, ego_a_in_veh_coord = shift_and_rotate_coordination(0, 0, 0,
                                                                                                           x_in_ego_coord,
                                                                                                           y_in_ego_coord,
                                                                                                           a_in_ego_coord)
                if (-5 < x_in_ego_coord < 1 * (ego_v_x) + ego_l / 2. + veh_l / 2. + 2 and abs(y_in_ego_coord) < 3) or \
                   (-5 < ego_x_in_veh_coord < 1 * (veh_v) + ego_l / 2. + veh_l / 2. + 2 and abs(ego_y_in_veh_coord) < 3):
                    if veh_type == 'DEFAULT_PEDTYPE':
                        traci.person.removeStages(veh)
                    else:
                        traci.vehicle.remove(veh)

                # if 0<x_in_sumo<3.5 and -22<y_in_sumo<-15:# and veh_sig!=1 and veh_sig!=9:
                #     traci.vehicle.moveToXY(veh, '4o', 1, -80, 1.85, 180,2)
                #     traci.vehicle.remove(vehID=veh)

    def _get_vehicles(self):
        self.n_ego_vehicles = defaultdict(list)
        veh_infos = traci.junction.getContextSubscriptionResults('0')
        for egoID in self.n_ego_dict.keys():
            veh_info_dict = copy.deepcopy(veh_infos)
            for i, veh in enumerate(veh_info_dict):
                if veh != egoID:
                    length = veh_info_dict[veh][traci.constants.VAR_LENGTH]
                    width = veh_info_dict[veh][traci.constants.VAR_WIDTH]
                    type = veh_info_dict[veh][traci.constants.VAR_TYPE]
                    if type == 'DEFAULT_PEDTYPE':
                        route = '0 0'
                    else:
                        route = veh_info_dict[veh][traci.constants.VAR_EDGES]
                    if type == 'DEFAULT_PEDTYPE':
                        road = veh_info_dict[veh][traci.constants.VAR_ROAD_ID]
                    else:
                        road = '0'
                    if route[0] == '4i':
                        continue
                    x_in_sumo, y_in_sumo = veh_info_dict[veh][traci.constants.VAR_POSITION]
                    a_in_sumo = veh_info_dict[veh][traci.constants.VAR_ANGLE]
                    # transfer x,y,a in car coord
                    x, y, a = _convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, length)
                    v = veh_info_dict[veh][traci.constants.VAR_SPEED]
                    self.n_ego_vehicles[egoID].append(dict(type=type, x=x, y=y, v=v, phi=a, l=length,
                                                           w=width, route=route, road=road))

    def _get_traffic_light(self):
        self.v_light = traci.trafficlight.getPhase('0')

    def get_road(self, ref_x, ref_y, task='right'):
        # Reference: https://sumo.dlr.de/docs/Tools/Sumolib.html#locate_nearby_edges_based_on_the_geo-coordinate
        edge_list = self.net.getNeighboringEdges(ref_x, ref_y, r=10.)          # todo: check the coordinates
        # pick the closest edge
        if len(edge_list) > 0:
            dist2edge, edge = sorted([(dist, edge) for edge, dist in edge_list])[0]
            edge_name = edge.getID()
        else:
            dist2edge, edge, edge_name = None, None, None

        # pick the closest lane
        lane_list = self.net.getNeighboringLanes(ref_x, ref_y, r=10.)          # todo: check the coordinates
        if len(lane_list) > 0:
            dist2lane, lane = sorted([(dist, lane) for lane, dist in lane_list])[0]
            lane_name = lane.getID()
        else:
            dist2lane, lane, lane_name = None, None, None

        if edge_name is None and lane_name is None:
            assert 0, 'No lane and edge found for ({} {})!'.format(ref_x, ref_y)
        elif edge_name is None and lane_name is not None:
            edge_name = lane_name.split('_')[0]
        elif edge_name is not None and lane_name is not None:
            assert edge_name == lane_name.split('_')[0], 'Unmatched edge and lane name for ({} {})!'.format(ref_x, ref_y)
        else:
            assert 0, 'No lane and edge found for ({} {})!'.format(ref_x, ref_y)

        point_list = []
        # judge the outgoing edge for different tasks
        if 'i' in lane_name:
            if task == 'left':
                target_lane_list = ['4i_4', '4i_3', '4i_2']
            elif task == 'straight':
                target_lane_list = ['3i_3', '3i_2']
            else:
                target_lane_list = ['2i_4', '2i_3', '2i_2']
            lane_center_l = list(traci.lane.getShape(target_lane_list[0]))
            lane_center_r = list(traci.lane.getShape(target_lane_list[-1]))
            width_l = traci.lane.getWidth(target_lane_list[0])
            width_r = traci.lane.getWidth(target_lane_list[-1])
            center_line_l = LineString(lane_center_l)
            center_line_r = LineString(lane_center_r)
            road_edge_l = center_line_l.parallel_offset(distance=width_l / 2, side='left', join_style=2, mitre_limit=50.)
            road_edge_r = center_line_r.parallel_offset(distance=width_r / 2, side='right', join_style=2, mitre_limit=50.)
            point_list.extend(list(road_edge_l.coords))
            point_list.extend(list(road_edge_r.coords))
        else:
            width = traci.lane.getWidth(lane_name)
            lane_center = list(traci.lane.getShape(lane_name))
            center_line = LineString(lane_center)
            road_edge_l = center_line.parallel_offset(distance=width / 2, side='left', join_style=2, mitre_limit=50.)
            road_edge_r = center_line.parallel_offset(distance=width / 2, side='right', join_style=2, mitre_limit=50.)
            point_list.extend(list(road_edge_l.coords))
            point_list.extend(list(road_edge_r.coords))
        return edge_name, lane_name, point_list

    def sim_step(self):
        self.sim_time += SIM_PERIOD
        # keep the signal lights unchanged during training
        if self.mode == "training":
            traci.trafficlight.setPhase('0', self.light_phase)
        traci.simulationStep()
        self._get_vehicles()
        self._get_traffic_light()
        self.collision_check()
        for egoID, collision_flag in self.n_ego_collision_flag.items():
            if collision_flag:
                self.collision_flag = True
                self.collision_ego_id = egoID

    def set_own_car(self, n_ego_dict_):
        assert len(self.n_ego_dict) == len(n_ego_dict_)
        for egoID in self.n_ego_dict.keys():
            self.n_ego_dict[egoID]['v_x'] = ego_v_x = n_ego_dict_[egoID]['v_x']
            self.n_ego_dict[egoID]['v_y'] = ego_v_y = n_ego_dict_[egoID]['v_y']
            self.n_ego_dict[egoID]['r'] = ego_r = n_ego_dict_[egoID]['r']
            self.n_ego_dict[egoID]['x'] = ego_x = n_ego_dict_[egoID]['x']
            self.n_ego_dict[egoID]['y'] = ego_y = n_ego_dict_[egoID]['y']
            self.n_ego_dict[egoID]['phi'] = ego_phi = n_ego_dict_[egoID]['phi']

            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi,
                                                                                           self.n_ego_dict[egoID]['l'])
            egdeID, lane = xy2_edgeID_lane(ego_x, ego_y)
            keeproute = 0
            try:
                traci.vehicle.moveToXY(egoID, -1, -1, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
            except traci.exceptions.TraCIException:
                print(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
                traci.vehicle.moveToXY(egoID, -1, -1, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
            traci.vehicle.setSpeed(egoID, math.sqrt(ego_v_x ** 2 + ego_v_y ** 2))

    def collision_check(self):  # True: collision
        flag_dict = dict()
        for egoID, list_of_veh_dict in self.n_ego_vehicles.items():
            ego_x = self.n_ego_dict[egoID]['x']
            ego_y = self.n_ego_dict[egoID]['y']
            ego_phi = self.n_ego_dict[egoID]['phi']
            ego_l = self.n_ego_dict[egoID]['l']
            ego_w = self.n_ego_dict[egoID]['w']
            ego_lw = (ego_l - ego_w) / 2
            ego_x0 = (ego_x + cos(ego_phi / 180 * pi) * ego_lw)
            ego_y0 = (ego_y + sin(ego_phi / 180 * pi) * ego_lw)
            ego_x1 = (ego_x - cos(ego_phi / 180 * pi) * ego_lw)
            ego_y1 = (ego_y - sin(ego_phi / 180 * pi) * ego_lw)
            flag_dict[egoID] = False

            for veh in list_of_veh_dict:
                if fabs(veh['x'] - ego_x) < 10 and fabs(veh['y'] - ego_y) < 10:
                    surrounding_lw = (veh['l'] - veh['w']) / 2
                    surrounding_x0 = (veh['x'] + cos(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_y0 = (veh['y'] + sin(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_x1 = (veh['x'] - cos(veh['phi'] / 180 * pi) * surrounding_lw)
                    surrounding_y1 = (veh['y'] - sin(veh['phi'] / 180 * pi) * surrounding_lw)
                    collision_check_dis = ((veh['w'] + ego_w) / 2 + 0.5) ** 2
                    if (ego_x0 - surrounding_x0) ** 2 + (ego_y0 - surrounding_y0) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x0 - surrounding_x1) ** 2 + (ego_y0 - surrounding_y1) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x1 - surrounding_x1) ** 2 + (ego_y1 - surrounding_y1) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x1 - surrounding_x0) ** 2 + (ego_y1 - surrounding_y0) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True

        self.n_ego_collision_flag = flag_dict


def test_traffic():
    import numpy as np
    from env_build.dynamics_and_models import ReferencePath

    def _reset_init_state():
        ref_path = ReferencePath('straight')
        random_index = int(np.random.random() * (900 + 500)) + 700
        x, y, phi = ref_path.indexs2points(random_index)
        v = 8 * np.random.random()
        return dict(ego=dict(v_x=v,
                             v_y=0,
                             r=0,
                             x=x.numpy(),
                             y=y.numpy(),
                             phi=phi.numpy(),
                             l=4.8,
                             w=2.2,
                             routeID='du',
                             ))

    # init_state = _reset_init_state()
    init_state = dict(ego=dict(v_x=8., v_y=0, r=0, x=-1.875, y=-30, phi=180, l=4.8, w=2.2, routeID='dl', ))
    traffic = Traffic(100., mode='training', init_n_ego_dict=init_state)
    traffic.init_light()
    traffic.init_traffic(init_state, training_task='left')
    traffic.sim_step()
    for i in range(100000000):
        for j in range(50):
            traffic.set_own_car(init_state)
            traffic.sim_step()
        # init_state = _reset_init_state()
        # traffic.init_traffic(init_state)
        traffic.sim_step()


def test_get_road_info(ref_x, ref_y, task):
    init_state = dict(ego=dict(v_x=8., v_y=0, r=0, x=-1.875, y=-30, phi=180, front_wheel=0, acc=0, l=4.8, w=2.2, routeID='dl'))
    traffic = Traffic(100., mode='training', init_n_ego_dict=init_state)
    traffic.reset(cross_num=1)
    from endtoend_env_utils import Para
    if Point(ref_x, ref_y).within(Polygon(Para.CROSSROAD_INTER)):
        print('inside the intersection')
    edge_name, lane_name, lane_shape = traffic.get_road(ref_x, ref_y, task)


if __name__ == "__main__":
    # test_traffic()
    test_get_road_info(1.63, -19.87, 'left')
