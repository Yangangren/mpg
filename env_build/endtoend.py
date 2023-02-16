#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/04/12
# @Author  : Yangang Ren (Tsinghua Univ.)
# @FileName: endtoend.py
# =====================================

import warnings
from collections import OrderedDict
from math import cos, sin, pi, sqrt
import random
from random import choice

import gym
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib.patches as mpatch
from shapely.geometry import Polygon, Point, LineString, mapping
from gym.utils import seeding

from env_build.dynamics_and_models import VehicleDynamics, ReferencePath, EnvironmentModel
from env_build.endtoend_env_utils import *
from env_build.traffic import Traffic
from LasVSim.sensor_module import *
from LasVSim.simulator import Settings

warnings.filterwarnings("ignore")


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = gym.spaces.Box(low, high, dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class CrossroadEnd2endMix(gym.Env):
    def __init__(self,
                 mode='training',
                 multi_display=False,
                 future_point_num=25,
                 **kwargs):
        self.mode = mode
        self.dynamics = VehicleDynamics()
        self.interested_other = None
        self.detected_other = None
        self.future_n_edge = None
        self.all_other = None
        self.ego_dynamics = None
        self.init_state = {}
        self.front_wheel_bound = [-25, 25]  # deg
        self.ego_l, self.ego_w = Para.L, Para.W
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.adv_action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.adv_act_bound = Para.adv_act_bound
        self.seed()
        self.light_phase = None
        self.light_encoding = None
        self.task_encoding = None
        self.step_length = 100  # ms

        self.step_time = self.step_length / 1000.0
        self.obs = None
        self.action = None

        self.done_type = 'not_done_yet'
        self.reward_info = None
        self.ego_info_dim = Para.EGO_ENCODING_DIM
        self.track_info_dim = Para.TRACK_ENCODING_DIM
        self.road_info_dim = Para.ROAD_ENCODING_DIM
        self.light_info_dim = Para.LIGHT_ENCODING_DIM
        self.task_info_dim = Para.TASK_ENCODING_DIM
        self.per_other_info_dim = Para.PER_OTHER_INFO_DIM
        self.other_start_dim = sum([self.ego_info_dim, self.track_info_dim, self.road_info_dim, self.light_info_dim, self.task_info_dim])
        self.veh_num = Para.MAX_VEH_NUM
        self.bike_num = Para.MAX_BIKE_NUM
        self.person_num = Para.MAX_PERSON_NUM
        self.other_number = sum([self.veh_num, self.bike_num, self.person_num])

        self.bicycle_mode_dict = None
        self.person_mode_dict = None
        self.training_task = None
        self.env_model = None
        self.ref_path = None
        self.ref_point = None
        self.future_n_point = None
        self.future_point_num = future_point_num

        """Load sensor module."""
        if self.mode == 'testing':
            self.settings = Settings('../../LasVSim/Library/default_simulation_setting.xml')
        else:
            self.settings = Settings('../LasVSim/Library/default_simulation_setting.xml')
        step_length = (self.settings.step_length *
                       self.settings.sensor_frequency)
        self.sensors = Sensors(step_length=step_length, sensor_info=self.settings.sensors)

        if not multi_display:
            self.traffic = Traffic(self.step_length,
                                   mode=self.mode,
                                   init_n_ego_dict=self.init_state)
            self.traffic.reset()
            self.reset()
            action = self.action_space.sample()
            observation, _, _, _ = self.step(action)
            self._set_observation_space(observation)
            plt.ion()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):  # kwargs include three keys
        if random.random() > 0.95:
            self.traffic.reset(cross_num=1)
        self.light_phase = self.traffic.init_light()
        self.training_task = choice(['left', 'straight', 'right'])
        self.task_encoding = TASK_ENCODING[self.training_task]
        self.light_encoding = LIGHT_ENCODING[self.light_phase]
        self.ref_path = ReferencePath(self.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.light_phase])
        self.env_model = EnvironmentModel()
        self.init_state = self._reset_init_state(LIGHT_PHASE_TO_GREEN_OR_RED[self.light_phase])
        self.traffic.init_traffic(self.init_state, self.training_task)
        self.traffic.sim_step()
        ego_dynamics = self._get_ego_dynamics([self.init_state['ego']['v_x'],
                                               self.init_state['ego']['v_y'],
                                               self.init_state['ego']['r'],
                                               self.init_state['ego']['x'],
                                               self.init_state['ego']['y'],
                                               self.init_state['ego']['phi'],
                                               self.init_state['ego']['front_wheel'],
                                               self.init_state['ego']['acc']],
                                              [0,
                                               0,
                                               self.dynamics.vehicle_params['miu'],
                                               self.dynamics.vehicle_params['miu']]
                                              )
        self._get_all_info(ego_dynamics)
        self.obs, other_mask_vector, self.future_n_point, self.future_n_edge = self.get_obs()
        self.action = None
        self.reward_info = None
        self.done_type = 'not_done_yet'
        all_info = dict(future_n_point=self.future_n_point, mask=other_mask_vector, future_n_edge=self.future_n_edge)
        return self.obs, all_info

    def close(self):
        self.traffic.close()

    def step(self, action):
        self.action = self._action_transformation_for_end2end(action)
        reward, self.reward_info = self._compute_reward(self.obs, self.action)
        self.done_type, done = self._judge_done()
        next_ego_state, next_ego_params = self._get_next_ego_state(self.action)
        ego_dynamics = self._get_ego_dynamics(next_ego_state, next_ego_params)
        self.traffic.set_own_car(dict(ego=ego_dynamics))
        self.traffic.sim_step()
        all_info = self._get_all_info(ego_dynamics)
        self.obs, other_mask_vector, self.future_n_point, self.future_n_edge = self.get_obs()
        self.reward_info.update({'final_rew': reward})
        all_info.update({'reward_info': self.reward_info, 'future_n_point': self.future_n_point,
                         'mask': other_mask_vector, 'future_n_edge': self.future_n_edge})
        return self.obs, reward, done, all_info

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _get_ego_dynamics(self, next_ego_state, next_ego_params):
        out = dict(v_x=next_ego_state[0],
                   v_y=next_ego_state[1],
                   r=next_ego_state[2],
                   x=next_ego_state[3],
                   y=next_ego_state[4],
                   phi=next_ego_state[5],
                   front_wheel=next_ego_state[6],
                   acc=next_ego_state[7],
                   l=self.ego_l,
                   w=self.ego_w,
                   alpha_f=next_ego_params[0],
                   alpha_r=next_ego_params[1],
                   miu_f=next_ego_params[2],
                   miu_r=next_ego_params[3])
        miu_f, miu_r = out['miu_f'], out['miu_r']
        F_zf, F_zr = self.dynamics.vehicle_params['F_zf'], self.dynamics.vehicle_params['F_zr']
        C_f, C_r = self.dynamics.vehicle_params['C_f'], self.dynamics.vehicle_params['C_r']
        alpha_f_bound, alpha_r_bound = 3 * miu_f * F_zf / C_f, 3 * miu_r * F_zr / C_r
        r_bound = miu_r * self.dynamics.vehicle_params['g'] / (abs(out['v_x']) + 1e-8)

        l, w, x, y, phi = out['l'], out['w'], out['x'], out['y'], out['phi']

        def cal_corner_point_of_ego_car():
            x0, y0, a0 = rotate_and_shift_coordination(l / 2, w / 2, 0, -x, -y, -phi)
            x1, y1, a1 = rotate_and_shift_coordination(l / 2, -w / 2, 0, -x, -y, -phi)
            x2, y2, a2 = rotate_and_shift_coordination(-l / 2, w / 2, 0, -x, -y, -phi)
            x3, y3, a3 = rotate_and_shift_coordination(-l / 2, -w / 2, 0, -x, -y, -phi)
            return (x0, y0), (x1, y1), (x2, y2), (x3, y3)

        corner_point = cal_corner_point_of_ego_car()
        out.update(dict(alpha_f_bound=alpha_f_bound,
                        alpha_r_bound=alpha_r_bound,
                        r_bound=r_bound,
                        corner_point=corner_point))

        return out

    def _get_all_info(self, ego_dynamics):  # used to update info, must be called every timestep before get_obs
        # to fetch info
        self.all_other = self.traffic.n_ego_vehicles['ego']  # coordination 2
        self.ego_dynamics = ego_dynamics  # coordination 2
        self.light_phase = self.traffic.v_light

        all_info = dict(all_other=self.all_other,
                        ego_dynamics=self.ego_dynamics,
                        v_light=self.light_phase)
        return all_info

    def _judge_done(self):
        """
        :return:
         1: bad done: collision
         2: bad done: break_road_constrain
         3: good done: task succeed
         4: not done
        """
        if self.traffic.collision_flag:
            return 'collision', 1
        if self._break_road_constrain():
            return 'break_road_constrain', 1
        elif self._deviate_too_much():
            return 'deviate_too_much', 1
        elif self._break_stability():
            return 'break_stability', 1
        elif self._break_red_light():
            return 'break_red_light', 1
        elif self._is_achieve_goal():
            return 'good_done', 1
        else:
            return 'not_done_yet', 0

    def _deviate_too_much(self):
        delta_longi, delta_lateral, delta_phi, delta_v = self.obs[self.ego_info_dim:self.ego_info_dim + self.track_info_dim]
        return True if abs(delta_lateral) > 15 else False

    def _break_road_constrain(self):
        results = list(map(lambda x: judge_feasible(*x, self.training_task), self.ego_dynamics['corner_point']))
        return not all(results)

    def _break_stability(self):
        alpha_f, alpha_r, miu_f, miu_r = self.ego_dynamics['alpha_f'], self.ego_dynamics['alpha_r'], \
                                         self.ego_dynamics['miu_f'], self.ego_dynamics['miu_r']
        alpha_f_bound, alpha_r_bound = self.ego_dynamics['alpha_f_bound'], self.ego_dynamics['alpha_r_bound']
        r_bound = self.ego_dynamics['r_bound']
        if -r_bound < self.ego_dynamics['r'] < r_bound:
            return False
        else:
            return True

    def _break_red_light(self):
        return True if self.light_phase > 2 and self.ego_dynamics['y'] > -Para.CROSSROAD_SIZE_LON / 2 and self.training_task != 'right' else False

    def _is_achieve_goal(self):
        x = self.ego_dynamics['x']
        y = self.ego_dynamics['y']
        if self.training_task == 'left':
            return True if x < -Para.CROSSROAD_SIZE_LAT / 2 - 10 and Para.OFFSET_L + Para.GREEN_BELT_LAT < y < Para.OFFSET_L + Para.GREEN_BELT_LAT + Para.LANE_WIDTH_1 * 3 else False
        elif self.training_task == 'right':
            return True if x > Para.CROSSROAD_SIZE_LAT / 2 + 10 and Para.OFFSET_R - Para.LANE_WIDTH_1 * 3 < y < Para.OFFSET_R else False
        else:
            assert self.training_task == 'straight'
            return True if y > Para.CROSSROAD_SIZE_LON / 2 + 10 and Para.OFFSET_U + Para.GREEN_BELT_LON < x < Para.OFFSET_U + Para.GREEN_BELT_LON + Para.LANE_WIDTH_3 * 2 else False

    def _action_transformation_for_end2end(self, action):  # [-1, 1]
        action = np.clip(action, -1.05, 1.05)
        delta_wheel_norm, delta_a_x_norm = action[0], action[1]
        delta_wheel = 0.4 * delta_wheel_norm
        delta_a_x = 4.5 * delta_a_x_norm
        scaled_action = np.array([delta_wheel, delta_a_x], dtype=np.float32)
        return scaled_action

    def _get_next_ego_state(self, trans_action):
        current_v_x = self.ego_dynamics['v_x']
        current_v_y = self.ego_dynamics['v_y']
        current_r = self.ego_dynamics['r']
        current_x = self.ego_dynamics['x']
        current_y = self.ego_dynamics['y']
        current_phi = self.ego_dynamics['phi']
        current_front_wheel = self.ego_dynamics['front_wheel']
        current_acc = self.ego_dynamics['acc']

        delta_front_wheel, delta_acc = trans_action
        state = np.array([[current_v_x, current_v_y, current_r, current_x, current_y, current_phi, current_front_wheel, current_acc]], dtype=np.float32)
        action = np.array([[delta_front_wheel, delta_acc]], dtype=np.float32)
        next_ego_state, next_ego_params = self.dynamics.prediction(state, action, 10)
        next_ego_state, next_ego_params = next_ego_state.numpy()[0], next_ego_params.numpy()[0]
        next_ego_state[0] = next_ego_state[0] if next_ego_state[0] >= 0 else 0.
        next_ego_state[5] = deal_with_phi(next_ego_state[5])
        next_ego_state[6] = np.clip(next_ego_state[6], self.front_wheel_bound[0], self.front_wheel_bound[1])  # todo
        next_ego_state[7] = np.clip(next_ego_state[7], -3.0, 1.5)
        return next_ego_state, next_ego_params

    def get_obs(self, exit_='D'):
        other_vector, other_mask_vector = self._construct_other_vector(exit_)
        ego_vector = self._construct_ego_vector()
        track_vector, self.ref_point, road_ref_point = self.ref_path.tracking_error_vector_vectorized(ego_vector[3], ego_vector[4], ego_vector[5], ego_vector[0]) # 3 for x; 4 foy y
        road_vector = self._construct_road_vector(np.array([road_ref_point]).T, ego_vector)
        future_n_point = self.ref_path.get_future_n_point(ego_vector[3], ego_vector[4], self.future_point_num * 2)
        future_n_edge = self._get_future_n_road_edge(future_n_point)
        self.light_encoding = LIGHT_ENCODING[self.light_phase]
        vector = np.concatenate((ego_vector, track_vector, road_vector, self.light_encoding, self.task_encoding, other_vector), axis=0)
        vector = vector.astype(np.float32)
        vector = self._convert_to_rela(vector)

        return vector, other_mask_vector, future_n_point, future_n_edge

    def _add_noise_to_vector(self, vector, vec_type=None):
        """
        Enabled by the 'vector_noise' variable in this class
        Add noise to the vector of objects, whose order is (x, y, v, phi, l, w) for other and (v_x, v_y, r, x, y, phi) for ego

        Noise is i.i.d for each element in the vector, i.e. the covariance matrix is diagonal
        Different types of objs lead to different mean and var, which are defined in the 'Para' class in e2e_utils.py

        :params
            vector: np.array(6,)
            vec_type: str in ['ego', 'other']
        :return
            noise_vec: np.array(6,)
        """
        assert vec_type in ['ego', 'other']
        if vec_type == 'ego':
            return vector + self.rng.multivariate_normal(Para.EGO_MEAN, Para.EGO_VAR)
        elif vec_type == 'other':
            return vector + self.rng.multivariate_normal(Para.OTHERS_MEAN, Para.OTHERS_VAR)

    def _convert_to_rela(self, obs_abso):
        obs_ego, obs_track, obs_road, obs_light, obs_task, obs_other = self._split_all(obs_abso)
        obs_other_reshape = self._reshape_other(obs_other)
        ego_x, ego_y = obs_ego[3], obs_ego[4]
        ego = np.array(([ego_x, ego_y] + [0.] * (self.per_other_info_dim - 2)), dtype=np.float32)
        ego = ego[np.newaxis, :]
        rela = obs_other_reshape - ego
        rela_obs_other = self._reshape_other(rela, reverse=True)
        return np.concatenate([obs_ego, obs_track, obs_road, obs_light, obs_task, rela_obs_other], axis=0)

    def _convert_to_abso(self, obs_rela):
        obs_ego, obs_track, obs_road, obs_light, obs_task, obs_other = self._split_all(obs_rela)
        obs_other_reshape = self._reshape_other(obs_other)
        ego_x, ego_y = obs_ego[3], obs_ego[4]
        ego = np.array(([ego_x, ego_y] + [0.] * (self.per_other_info_dim - 2)), dtype=np.float32)
        ego = ego[np.newaxis, :]
        abso = obs_other_reshape + ego
        abso_obs_other = self._reshape_other(abso, reverse=True)
        return np.concatenate([obs_ego, obs_track, obs_road, obs_light, obs_task, abso_obs_other])

    def _split_all(self, obs):
        obs_ego = obs[:self.ego_info_dim]
        obs_track = obs[self.ego_info_dim:
                        self.ego_info_dim + self.track_info_dim]
        obs_road = obs[self.ego_info_dim + self.track_info_dim:
                       self.ego_info_dim + self.track_info_dim + self.road_info_dim]
        obs_light = obs[self.ego_info_dim + self.track_info_dim + self.road_info_dim:
                        self.ego_info_dim + self.track_info_dim + self.road_info_dim + self.light_info_dim]
        obs_task = obs[self.ego_info_dim + self.track_info_dim + self.road_info_dim + self.light_info_dim:
                       self.ego_info_dim + self.track_info_dim + self.road_info_dim + self.light_info_dim + self.task_info_dim]
        obs_other = obs[self.other_start_dim:]

        return obs_ego, obs_track, obs_road, obs_light, obs_task, obs_other

    def _split_other(self, obs_other):
        obs_bike = obs_other[:self.bike_num * self.per_other_info_dim]
        obs_person = obs_other[self.bike_num * self.per_other_info_dim:
                               (self.bike_num + self.person_num) * self.per_other_info_dim]
        obs_veh = obs_other[(self.bike_num + self.person_num) * self.per_other_info_dim:]
        return obs_bike, obs_person, obs_veh

    def _reshape_other(self, obs_other, reverse=False):
        if reverse:
            return np.reshape(obs_other, (self.other_number * self.per_other_info_dim,))
        else:
            return np.reshape(obs_other, (self.other_number, self.per_other_info_dim))

    def _construct_ego_vector(self):
        ego_v_x = self.ego_dynamics['v_x']
        ego_v_y = self.ego_dynamics['v_y']
        ego_r = self.ego_dynamics['r']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_front_wheel = self.ego_dynamics['front_wheel']
        ego_acc = self.ego_dynamics['acc']
        ego_feature = [ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi, ego_front_wheel, ego_acc]
        return np.array(ego_feature, dtype=np.float32)

    def _construct_other_vector(self, exit_='D'):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi'] * pi / 180
        other_vector = []
        other_mask_vector = []

        ego_state = [self.ego_dynamics['x'], self.ego_dynamics['y'], self.ego_dynamics['v_x'], self.ego_dynamics['phi']]
        all_others = list(filter(lambda v: Para.CROSSROAD_SIZE_LAT/2 + 40 > v['x'] > -Para.CROSSROAD_SIZE_LAT/2 - 40 and
                                             Para.CROSSROAD_SIZE_LON/2 + 40 > v['y'] > -Para.CROSSROAD_SIZE_LON/2 - 40, self.all_other))
        all_others = list(filter(lambda v: ((v['x'] - ego_x) * np.cos(ego_phi) + (v['y'] - ego_y) * np.sin(ego_phi)) > -1.5 * Para.L, all_others))
        if len(all_others) > Para.MAX_TRAFFIC:
            all_others = sorted(all_others, key=lambda v: (sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2)))
            all_others = all_others[0:Para.MAX_TRAFFIC]

        self.sensors.update(pos=ego_state, vehicles=all_others)
        self.detected_other = self.sensors.getVisibleVehicles()

        name_settings = dict(D=dict(do='1o', di='1i', ro='2o', ri='2i', uo='3o', ui='3i', lo='4o', li='4i'),
                             R=dict(do='2o', di='2i', ro='3o', ri='3i', uo='4o', ui='4i', lo='1o', li='1i'),
                             U=dict(do='3o', di='3i', ro='4o', ri='4i', uo='1o', ui='1i', lo='2o', li='2i'),
                             L=dict(do='4o', di='4i', ro='1o', ri='1i', uo='2o', ui='2i', lo='3o', li='3i'))

        name_setting = name_settings[exit_]

        def filter_interested_other(vs):
            veh_list, ped_list, bike_list = [], [], []

            def cal_turn_rad(v):
                if not(-Para.CROSSROAD_SIZE_LAT/2 < v['x'] < Para.CROSSROAD_SIZE_LAT/2 and -Para.CROSSROAD_SIZE_LON/2 < v['y'] < Para.CROSSROAD_SIZE_LON/2):
                    turn_rad = 0.
                else:
                    start = v['route'][0]
                    end = v['route'][1]
                    if (start == name_setting['do'] and end == name_setting['ui']) or (start == name_setting['ro'] and end == name_setting['li'])\
                        or (start == name_setting['uo'] and end == name_setting['di']) or (start == name_setting['lo'] and end == name_setting['ri']):
                        turn_rad = 0.
                    elif (start == name_setting['do'] and end == name_setting['ri']) or (start == name_setting['ro'] and end == name_setting['ui'])\
                        or (start == name_setting['uo'] and end == name_setting['li']) or (start == name_setting['lo'] and end == name_setting['di']):
                        turn_rad = -1/(Para.CROSSROAD_SIZE_LON / 2 - 3.9 * Para.LANE_WIDTH_1)
                    elif start == name_setting['do'] and end == name_setting['li']:   # 'dl'
                        turn_rad = 1/sqrt((v['x']-(-Para.CROSSROAD_SIZE_LAT/2))**2 + (v['y']-(-Para.CROSSROAD_SIZE_LON/2))**2)
                    elif start == name_setting['ro'] and end == name_setting['di']:   # 'rd'
                        turn_rad = 1/sqrt((v['x']-(Para.CROSSROAD_SIZE_LAT/2))**2 + (v['y']-(-Para.CROSSROAD_SIZE_LON/2))**2)
                    elif start == name_setting['uo'] and end == name_setting['ri']:   # 'ur'
                        turn_rad = 1/sqrt((v['x']-(Para.CROSSROAD_SIZE_LAT/2))**2 + (v['y']-(Para.CROSSROAD_SIZE_LON/2))**2)
                    elif start == name_setting['lo'] and end == name_setting['ui']:   # 'lu'
                        turn_rad = 1/sqrt((v['x']-(-Para.CROSSROAD_SIZE_LAT/2))**2 + (v['y']-(Para.CROSSROAD_SIZE_LON/2))**2)
                    else:
                        turn_rad = 0.
                return turn_rad

            for v in vs:
                if v['type'].startswith('bicycle'):
                    v.update(partici_type=[1., 0., 0.], turn_rad=0.0, exist=True)
                    bike_list.append(v)

                elif v['type'].startswith('DEFAULT_PEDTYPE'):
                    v.update(partici_type=[0., 1., 0.], turn_rad=0.0, exist=True)
                    ped_list.append(v)

                elif v['type'].startswith('car'):
                    v.update(partici_type=[0., 0., 1.], turn_rad=cal_turn_rad(v), exist=True)
                    veh_list.append(v)

                else:
                    assert False, 'no defined type!'

            vir_bike = dict(type="bicycle_1",
                          x=Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 + Para.LANE_WIDTH_3 + Para.BIKE_LANE_WIDTH / 2,
                          y=-(Para.CROSSROAD_SIZE_LON / 2 + 30), v=0,
                          phi=90, w=0.48, l=2, route=('1o', '3i'), partici_type=[1., 0., 0.], turn_rad=0., exist=False)

            while len(bike_list) < self.bike_num:
                bike_list.append(vir_bike)
            if len(bike_list) > self.bike_num:
                bike_list = sorted(bike_list, key=lambda v: (sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2)))
                bike_list = bike_list[:self.bike_num]

            vir_person = dict(type='DEFAULT_PEDTYPE',
                        x=Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 + Para.LANE_WIDTH_3 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH / 2,
                        y=-(Para.CROSSROAD_SIZE_LON / 2 + 30),
                        v=0, phi=90, w=0.525, l=0.75, road="0_c1", partici_type=[0., 1., 0.], turn_rad=0., exist=False)

            while len(ped_list) < self.person_num:
                ped_list.append(vir_person)
            if len(ped_list) > self.person_num:
                ped_list = sorted(ped_list, key=lambda v: (sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2)))
                ped_list = ped_list[:self.person_num]

            vir_veh = dict(type="car_1", x=Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 * 1.5,
                        y=-(Para.CROSSROAD_SIZE_LON / 2 + 30), v=0, phi=90, w=2.5, l=5, route=('1o', '2i'),
                        partici_type=[0., 0., 1.], turn_rad=0., exist=False)

            while len(veh_list) < self.veh_num:
                veh_list.append(vir_veh)
            if len(veh_list) > self.veh_num:
                veh_list = sorted(veh_list, key=lambda v: (sqrt((v['y'] - ego_y) ** 2 + (v['x'] - ego_x) ** 2), -v['x']))
                veh_list = veh_list[:self.veh_num]

            traffic_actor = bike_list + ped_list + veh_list
            return traffic_actor

        self.interested_other = filter_interested_other(self.detected_other)

        for other in self.interested_other:
            other_x, other_y, other_v, other_phi, other_l, other_w, other_type, other_turn_rad, other_mask = \
                other['x'], other['y'], other['v'], other['phi'], other['l'], other['w'], other['partici_type'], other[
                    'turn_rad'], other['exist']
            other_vector.extend(
                [other_x, other_y, other_v, other_phi, other_l, other_w] + other_type + [other_turn_rad])
            other_mask_vector.append(other_mask)

        return np.array(other_vector, dtype=np.float32), np.array(other_mask_vector, dtype=np.float32)

    def _construct_road_vector(self, ref_point, ego_vector):
        road_edge_list = []
        for col in range(ref_point.shape[1]):
            road_edge_list.append(self._find_road_edge(ref_point[:, col]))
        road_edges = np.concatenate(road_edge_list, axis=-1).squeeze(axis=1)
        road_edges = np.reshape(road_edges, newshape=(-1, 2))
        self.road_edges = road_edges
        shifted_x = road_edges[:, 0] - ego_vector[3]
        shifted_y = road_edges[:, 1] - ego_vector[4]
        coordi_rotate_d_in_rad = ego_vector[5] * np.pi / 180
        delta_y = -shifted_x * np.sin(coordi_rotate_d_in_rad) + shifted_y * np.cos(coordi_rotate_d_in_rad)
        return delta_y

    def _get_future_n_road_edge(self, ref_point):
        road_edge_list = []
        for col in range(ref_point.shape[1]):
            road_edge_list.append(self._find_road_edge(ref_point[:, col]))
        return np.concatenate(road_edge_list, axis=-1)

    def _find_road_edge(self, ref_point):      # depend on road map
        ref_x, ref_y, ref_phi, ref_v = ref_point
        ref_phi = ref_phi * pi / 180

        if Point(ref_x, ref_y).within(Polygon(Para.CROSSROAD_INTER)):  # inside intersection (not include edge points)
            inter_polygon = Polygon(Para.CROSSROAD_INTER)
        else:                                                          # outside intersection
            edge_name, lane_name, rect_shape = self.traffic.get_road(ref_x, ref_y, self.training_task)
            inter_polygon = Polygon(rect_shape)

        slope_vert_line = np.arctan(-1 / np.tan(ref_phi))
        delta_d = 100
        right_point_x, right_point_y = ref_x + delta_d * cos(slope_vert_line), ref_y + delta_d * sin(slope_vert_line)
        left_point_x, left_point_y = ref_x - delta_d * cos(slope_vert_line), ref_y - delta_d * sin(slope_vert_line)
        while Point(right_point_x, right_point_y).within(inter_polygon) or Point(left_point_x, left_point_y).within(inter_polygon):
            delta_d = 2.0 * delta_d
            right_point_x, right_point_y = ref_x + delta_d * cos(slope_vert_line), ref_y + delta_d * sin(slope_vert_line)
            left_point_x, left_point_y = ref_x - delta_d * cos(slope_vert_line), ref_y - delta_d * sin(slope_vert_line)

        vert_line = LineString([(left_point_x, left_point_y), (right_point_x, right_point_y)])
        intersets = np.array(vert_line.intersection(inter_polygon).coords, dtype=np.float32).reshape((-1, 1))
        return intersets

    def _reset_init_state(self, light_phase):
        if light_phase == 'green':
            if self.training_task == 'left':
                random_index = int(np.random.random() * (1200 + 1400)) + 600
            elif self.training_task == 'straight':
                random_index = int(np.random.random() * (1400 + 1400)) + 600
            else:
                random_index = int(np.random.random() * (700 + 900)) + 600
        else:
            random_index = int(np.random.random() * 200) + 500

        if self.mode == 'testing':
            random_index = int(np.random.random() * 200) + 600
            init_ref_path = ReferencePath(self.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.light_phase])
        elif self.mode == 'evaluating':
            random_index = int(np.random.random() * 200) + 600
            init_ref_path = self.ref_path
        else:
            init_ref_path = self.ref_path

        x, y, phi, exp_v = init_ref_path.idx2point(random_index)
        v = exp_v * np.random.random()
        routeID = TASK2ROUTEID[self.training_task]
        return dict(ego=dict(v_x=v,
                             v_y=0,
                             r=0,
                             x=x,
                             y=y,
                             phi=phi,
                             front_wheel=0,
                             acc=0,
                             l=self.ego_l,
                             w=self.ego_w,
                             routeID=routeID,
                             ))

    def _compute_reward(self, obs, action):
        obses, actions = obs[np.newaxis, :], action[np.newaxis, :]
        reward, punish_term_for_training, real_punish_term, _, _, _, _, _, reward_dict = self.env_model.compute_rewards(obses, actions)
        reward = np.clip(reward.numpy()[0], a_min=-25., a_max=0., dtype=np.float32)
        if self.traffic.collision_flag:
            reward += -30
        if self._break_road_constrain():
            reward += -10
        elif self._deviate_too_much():
            reward += -1
        elif self._break_red_light():
            reward += -10
        elif self._is_achieve_goal():
            reward += 100
        else:
            reward += 5

        for k, v in reward_dict.items():
            reward_dict[k] = v.numpy()[0]
        reward_dict.update(punish_term_for_training=punish_term_for_training.numpy()[0],
                           real_punish_term=real_punish_term.numpy()[0])
        return reward, reward_dict

    def render(self, mode='human', weights=None):
        if mode == 'human':
            # plot basic map
            extension = 40
            dotted_line_style = '--'
            solid_line_style = '-'

            plt.cla()
            ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
            ax.axis("equal")
            patches = []

            # ----------arrow--------------
            # plt.arrow(lane_width / 2, -square_length / 2 - 10, 0, 3, color='darkviolet')
            # plt.arrow(lane_width / 2, -square_length / 2 - 10 + 3, -0.5, 1.0, color='darkviolet', head_width=0.7)
            # plt.arrow(lane_width * 1.5, -square_length / 2 - 10, 0, 4, color='darkviolet', head_width=0.7)
            # plt.arrow(lane_width * 2.5, -square_length / 2 - 10, 0, 3, color='darkviolet')
            # plt.arrow(lane_width * 2.5, -square_length / 2 - 10 + 3, 0.5, 1.0, color='darkviolet', head_width=0.7)

            # ----------green belt--------------
            patches.append(plt.Rectangle((-Para.CROSSROAD_SIZE_LAT / 2 - extension, Para.OFFSET_L),
                                       extension, Para.GREEN_BELT_LAT, edgecolor='black', facecolor='lightgrey',
                                       linewidth=1))
            patches.append(plt.Rectangle((-Para.CROSSROAD_SIZE_LAT / 2 - extension, Para.OFFSET_L),
                                       extension, Para.GREEN_BELT_LAT, edgecolor='black', facecolor='lightgrey',
                                       linewidth=1))
            patches.append(plt.Rectangle((Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_R),
                                       extension, Para.GREEN_BELT_LAT, edgecolor='black', facecolor='lightgrey',
                                       linewidth=1))
            patches.append(plt.Rectangle((Para.OFFSET_U, Para.CROSSROAD_SIZE_LON / 2),
                                       Para.GREEN_BELT_LON, extension, edgecolor='black', facecolor='lightgrey',
                                       linewidth=1))
            patches.append(plt.Rectangle((Para.OFFSET_D - Para.GREEN_BELT_LON, -Para.CROSSROAD_SIZE_LON / 2 - extension),
                                       Para.GREEN_BELT_LON, extension, edgecolor='black', facecolor='lightgrey',
                                       linewidth=1))

            # Left out lane
            for i in range(1, Para.LANE_NUMBER_LAT_OUT + 2):
                lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                                   Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_L + Para.GREEN_BELT_LAT
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_OUT else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LAT_OUT else 1
                plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension, -Para.CROSSROAD_SIZE_LAT / 2],
                         [base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                         linestyle=linestyle, color='black', linewidth=linewidth)
            # Left in lane
            for i in range(1, Para.LANE_NUMBER_LAT_IN + 2):
                lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                                   Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_L
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_IN else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LAT_IN else 1
                plt.plot([-Para.CROSSROAD_SIZE_LAT / 2 - extension, -Para.CROSSROAD_SIZE_LAT / 2],
                         [base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Right out lane
            for i in range(1, Para.LANE_NUMBER_LAT_OUT + 2):
                lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                                   Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_R
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_OUT else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LAT_OUT else 1
                plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2 + extension],
                         [base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Right in lane
            for i in range(1, Para.LANE_NUMBER_LAT_IN + 2):
                lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                                   Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_R + Para.GREEN_BELT_LAT
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LAT_IN else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LAT_IN else 1
                plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2 + extension],
                         [base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Up in lane
            for i in range(1, Para.LANE_NUMBER_LON_IN + 2):
                lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                                   Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_U
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_IN else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LON_IN else 1
                plt.plot([base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                         [Para.CROSSROAD_SIZE_LON / 2 + extension, Para.CROSSROAD_SIZE_LON / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Up out lane
            for i in range(1, Para.LANE_NUMBER_LON_OUT + 2):
                lane_width_flag = [Para.LANE_WIDTH_3, Para.LANE_WIDTH_3, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_U + Para.GREEN_BELT_LON
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_OUT else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LON_OUT else 1
                plt.plot([base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                         [Para.CROSSROAD_SIZE_LON / 2 + extension, Para.CROSSROAD_SIZE_LON / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Down in lane
            for i in range(1, Para.LANE_NUMBER_LON_IN + 2):
                lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                                   Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_D
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_IN else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LON_IN else 1
                plt.plot([base + sum(lane_width_flag[:i]), base + sum(lane_width_flag[:i])],
                         [-Para.CROSSROAD_SIZE_LON / 2 - extension, -Para.CROSSROAD_SIZE_LON / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Down out lane
            for i in range(1, Para.LANE_NUMBER_LON_OUT + 2):
                lane_width_flag = [Para.LANE_WIDTH_3, Para.LANE_WIDTH_3, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
                base = Para.OFFSET_D - Para.GREEN_BELT_LON
                linestyle = dotted_line_style if i < Para.LANE_NUMBER_LON_OUT else solid_line_style
                linewidth = 1 if i < Para.LANE_NUMBER_LON_OUT else 1
                plt.plot([base - sum(lane_width_flag[:i]), base - sum(lane_width_flag[:i])],
                         [-Para.CROSSROAD_SIZE_LON / 2 - extension, -Para.CROSSROAD_SIZE_LON / 2],
                         linestyle=linestyle, color='black', linewidth=linewidth)

            # Oblique
            inter_polygon = Polygon(Para.CROSSROAD_INTER)
            ax.plot(inter_polygon.exterior.xy[0], inter_polygon.exterior.xy[1], '-', color='black', linewidth=1.5, zorder=0, alpha=1.0)

            # stop line
            lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Down
            plt.plot([Para.OFFSET_D, Para.OFFSET_D + sum(lane_width_flag[:Para.LANE_NUMBER_LON_IN])],
                     [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2], color='black')
            lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Up
            plt.plot([-sum(lane_width_flag[:Para.LANE_NUMBER_LON_IN]) + Para.OFFSET_U, Para.OFFSET_U],
                     [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2], color='black')
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_L, Para.OFFSET_L - sum(lane_width_flag[:Para.LANE_NUMBER_LAT_IN])],
                     color='black')  # left
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2], [Para.OFFSET_R + Para.GREEN_BELT_LAT,
                                                                                  Para.OFFSET_R + Para.GREEN_BELT_LAT + sum(
                                                                                      lane_width_flag[
                                                                                      :Para.LANE_NUMBER_LAT_IN])],
                     color='black')

            # traffic light
            v_light = self.light_phase
            light_line_width = 2
            if v_light == 0 or v_light == 1:
                v_color_1, v_color_2, h_color_1, h_color_2 = 'green', 'green', 'red', 'red'
            elif v_light == 2:
                v_color_1, v_color_2, h_color_1, h_color_2 = 'orange', 'orange', 'red', 'red'
            elif v_light == 3:
                v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'green'
            elif v_light == 4:
                v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'red', 'orange'
            elif v_light == 5:
                v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'green', 'red'
            else:
                v_color_1, v_color_2, h_color_1, h_color_2 = 'red', 'red', 'orange', 'red'

            lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Down
            plt.plot([Para.OFFSET_D, Para.OFFSET_D + sum(lane_width_flag[:1])],
                     [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2],
                     color=v_color_1, linewidth=light_line_width)
            plt.plot([Para.OFFSET_D + sum(lane_width_flag[:1]), Para.OFFSET_D + sum(lane_width_flag[:2])],
                     [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2],
                     color=v_color_2, linewidth=light_line_width)
            plt.plot([Para.OFFSET_D + sum(lane_width_flag[:2]), Para.OFFSET_D + sum(lane_width_flag[:3])],
                     [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2],
                     color='green', linewidth=light_line_width)

            lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Up
            plt.plot([-sum(lane_width_flag[:1]) + Para.OFFSET_U, Para.OFFSET_U],
                     [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2],
                     color=v_color_1, linewidth=light_line_width)
            plt.plot([-sum(lane_width_flag[:2]) + Para.OFFSET_U, -sum(lane_width_flag[:1]) + Para.OFFSET_U],
                     [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2],
                     color=v_color_2, linewidth=light_line_width)
            plt.plot([-sum(lane_width_flag[:3]) + Para.OFFSET_U, -sum(lane_width_flag[:2]) + Para.OFFSET_U],
                     [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2],
                     color='green', linewidth=light_line_width)

            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # left
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_L, Para.OFFSET_L - sum(lane_width_flag[:1])],
                     color=h_color_1, linewidth=light_line_width)
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_L - sum(lane_width_flag[:1]), Para.OFFSET_L - sum(lane_width_flag[:3])],
                     color=h_color_2, linewidth=light_line_width)
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_L - sum(lane_width_flag[:3]), Para.OFFSET_L - sum(lane_width_flag[:4])],
                     color='green', linewidth=light_line_width)

            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # right
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_R + Para.GREEN_BELT_LAT,
                      Para.OFFSET_R + Para.GREEN_BELT_LAT + sum(lane_width_flag[:1])],
                     color=h_color_1, linewidth=light_line_width)
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_R + Para.GREEN_BELT_LAT + sum(lane_width_flag[:1]),
                      Para.OFFSET_R + Para.GREEN_BELT_LAT + sum(lane_width_flag[:3])],
                     color=h_color_2, linewidth=light_line_width)
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2],
                     [Para.OFFSET_R + Para.GREEN_BELT_LAT + sum(lane_width_flag[:3]),
                      Para.OFFSET_R + Para.GREEN_BELT_LAT + sum(lane_width_flag[:4])],
                     color='green', linewidth=light_line_width)

            def is_in_plot_area(x, y, tolerance=5):
                if -Para.CROSSROAD_SIZE_LAT / 2 - extension + tolerance < x < Para.CROSSROAD_SIZE_LAT / 2 + extension - tolerance and \
                        -Para.CROSSROAD_SIZE_LON / 2 - extension + tolerance < y < Para.CROSSROAD_SIZE_LON / 2 + extension - tolerance:
                    return True
                else:
                    return False

            def draw_rotate_rec(type, x, y, a, l, w, color, linestyle='-', patch=False):
                RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
                RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
                LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
                LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
                if patch:
                    if type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                        item_color = 'purple'
                    elif type == 'DEFAULT_PEDTYPE':
                        item_color = 'lime'
                    else:
                        item_color = 'lightgray'
                    patches.append(plt.Rectangle((x + LU_x, y + LU_y), w, l, edgecolor=item_color,facecolor=item_color,
                                               angle=-(90 - a), zorder=30))
                else:
                    patches.append(matplotlib.patches.Rectangle(np.array([-l/2+x, -w/2+y]),
                                                                width=l, height=w,
                                                                fill=False,
                                                                facecolor=None,
                                                                edgecolor=color,
                                                                linestyle=linestyle,
                                                                linewidth=1.0,
                                                                transform=Affine2D().rotate_deg_around(*(x, y),
                                                                                                       a)))

            def draw_rotate_batch_rec(x, y, a, l, w):
                for i in range(len(x)):
                    patches.append(matplotlib.patches.Rectangle(np.array([-l[i]/2+x[i], -w[i]/2+y[i]]),
                                                                width=l[i], height=w[i],
                                                                fill=False,
                                                                facecolor=None,
                                                                edgecolor='k',
                                                                linewidth=1.0,
                                                                transform=Affine2D().rotate_deg_around(*(x[i], y[i]),
                                                                                                       a[i])))

            def plot_phi_line(type, x, y, phi, color):
                if type.startswith('bicycle'):
                    line_length = 2
                elif type.startswith('car'):
                    line_length = 5
                else:
                    line_length = 1
                x_forw, y_forw = x + line_length * cos(phi * pi / 180.), \
                                 y + line_length * sin(phi * pi / 180.)
                plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

            def draw_sensor_range(x_ego, y_ego, a_ego, l_bias, w_bias, angle_bias, angle_range, dist_range, color):
                x_sensor = x_ego + l_bias * cos(a_ego) - w_bias * sin(a_ego)
                y_sensor = y_ego + l_bias * sin(a_ego) + w_bias * cos(a_ego)
                theta1 = a_ego + angle_bias - angle_range / 2
                theta2 = a_ego + angle_bias + angle_range / 2
                sensor = mpatch.Wedge([x_sensor, y_sensor], dist_range, theta1=theta1 * 180 / pi,
                                      theta2=theta2 * 180 / pi, fc=color, alpha=0.2, zorder=1)
                ax.add_patch(sensor)

            # plot others
            filted_all_other = [item for item in self.all_other if is_in_plot_area(item['x'], item['y'])]
            other_xs = np.array([item['x'] for item in filted_all_other], np.float32)
            other_ys = np.array([item['y'] for item in filted_all_other], np.float32)
            other_as = np.array([item['phi'] for item in filted_all_other], np.float32)
            other_ls = np.array([item['l'] for item in filted_all_other], np.float32)
            other_ws = np.array([item['w'] for item in filted_all_other], np.float32)

            draw_rotate_batch_rec(other_xs, other_ys, other_as, other_ls, other_ws)

            # plot interested others
            if weights is not None:
                assert weights.shape == (self.other_number,), print(weights.shape)
            index_top_k_in_weights = weights.argsort()[-4:][::-1]
            for i in range(len(self.interested_other)):
                item = self.interested_other[i]
                item_mask = item['exist']
                item_x = item['x']
                item_y = item['y']
                item_phi = item['phi']
                item_l = item['l']
                item_w = item['w']
                item_type = item['type']
                if is_in_plot_area(item_x, item_y):
                    draw_rotate_rec(item_type, item_x, item_y, item_phi, item_l, item_w, color='pink', linestyle=':', patch=True)
                    plt.text(item_x, item_y, str(item_mask)[0])

            # detected traffic by sensors
            if self.detected_other is not None:
                for item in self.detected_other:
                    item_x = item['x']
                    item_y = item['y']
                    item_phi = item['phi']
                    item_l = item['l']
                    item_w = item['w']
                    item_type = item['type']

                    draw_rotate_rec(str(item_type), item_x, item_y, item_phi, item_l, item_w, color='g', linestyle='-', patch=False)

            # plot own car
            abso_obs = self._convert_to_abso(self.obs)
            obs_ego, obs_track, obs_road, obs_light, obs_task, obs_other = self._split_all(abso_obs)
            ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi, ego_wheel, ego_ax = obs_ego
            devi_longi, devi_lateral, devi_phi, devi_v = obs_track

            # plot_phi_line('self_car', ego_x, ego_y, ego_phi, 'fuchsia')
            draw_rotate_rec('self_car', ego_x, ego_y, ego_phi, self.ego_l, self.ego_w, 'fuchsia')

            plt.plot(self.future_n_point[0], self.future_n_point[1], 'g.')
            plt.plot(self.future_n_edge[0], self.future_n_edge[1], '*', color='darkred')
            plt.plot(self.future_n_edge[2], self.future_n_edge[3], 'x', color='slateblue')
            path_x, path_y, path_phi, path_v = self.ref_point[0], self.ref_point[1], self.ref_point[2], self.ref_point[3]
            ax.plot(path_x, path_y, '.', color='fuchsia', markersize=14)

            plt.scatter(self.road_edges[:, 0], self.road_edges[:, 1], marker='o', color='red', zorder=3)

            # plot sensors
            draw_sensor_range(ego_x, ego_y, ego_phi * pi / 180, l_bias=self.ego_l / 2, w_bias=0, angle_bias=0,
                              angle_range=2 * pi, dist_range=70, color='thistle')
            draw_sensor_range(ego_x, ego_y, ego_phi * pi / 180, l_bias=self.ego_l / 2, w_bias=0, angle_bias=0,
                              angle_range=70 * pi / 180, dist_range=80, color="slategray")
            draw_sensor_range(ego_x, ego_y, ego_phi * pi / 180, l_bias=self.ego_l / 2, w_bias=0, angle_bias=0,
                              angle_range=90 * pi / 180, dist_range=60, color="slategray")

            # from matplotlib.patches import Circle, Ellipse
            # cir_right = Circle(xy=(Para.CROSSROAD_SIZE_LAT/2, -Para.CROSSROAD_SIZE_LON/2), radius=Para.CROSSROAD_SIZE_LON / 2 - 3.9 * Para.LANE_WIDTH_1, alpha=0.5)
            # ax.add_patch(cir_right)
            # cir_right = Circle(xy=(-Para.CROSSROAD_SIZE_LAT/2, -Para.CROSSROAD_SIZE_LON/2), radius=Para.CROSSROAD_SIZE_LAT / 2 + Para.OFFSET_D + 0.5 * Para.LANE_WIDTH_2, alpha=0.5)
            # ax.add_patch(cir_right)
            # cir_right = Circle(xy=(-Para.CROSSROAD_SIZE_LAT/2, -Para.CROSSROAD_SIZE_LON/2), radius=Para.CROSSROAD_SIZE_LON / 2 + Para.OFFSET_L + Para.GREEN_BELT_LAT + 0.5 * Para.LANE_WIDTH_1, alpha=0.5)
            # ax.add_patch(cir_right)
            # cir_right = Circle(xy=(-Para.CROSSROAD_SIZE_LAT/2, -Para.CROSSROAD_SIZE_LON/2), radius=0.5*(Para.CROSSROAD_SIZE_LON / 2 + Para.OFFSET_L + Para.GREEN_BELT_LAT + 0.5 * Para.LANE_WIDTH_1)+0.5*(Para.CROSSROAD_SIZE_LAT / 2 + Para.OFFSET_D + 0.5 * Para.LANE_WIDTH_2), alpha=0.5)
            # ax.add_patch(cir_right)
            # e = Ellipse(xy=(-Para.CROSSROAD_SIZE_LAT/2, -Para.CROSSROAD_SIZE_LON/2), width=(Para.CROSSROAD_SIZE_LAT / 2 + Para.OFFSET_D + 0.5 * Para.LANE_WIDTH_2) * 2, height=(Para.CROSSROAD_SIZE_LON / 2 + Para.OFFSET_L + Para.GREEN_BELT_LAT + 0.5 * Para.LANE_WIDTH_1) * 2, angle=0.)
            # ax.add_patch(e)
            # e = Ellipse(xy=(Para.CROSSROAD_SIZE_LAT/2, -Para.CROSSROAD_SIZE_LON/2), width=(Para.CROSSROAD_SIZE_LAT / 2 + Para.OFFSET_D + 0.5 * Para.LANE_WIDTH_2) * 2, height=(Para.CROSSROAD_SIZE_LON / 2 + Para.OFFSET_L + Para.GREEN_BELT_LAT + 0.5 * Para.LANE_WIDTH_1) * 2, angle=0.)
            # ax.add_patch(e)

            # plot real-time trajectory
            color = ['blue', 'coral', 'darkcyan', 'pink']
            for i, item in enumerate(self.ref_path.path_list['green']):
                if self.ref_path.path_index == i:                   # todo
                    plt.plot(item[0], item[1], color=color[0], alpha=0.7)
                else:
                    plt.plot(item[0], item[1], color=color[1], alpha=0.2)
                    # indexs, points = item.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
                    # path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
                    # plt.plot(path_x, path_y,  color=color[i])

            # text
            text_x, text_y_start = -110, 60
            ge = iter(range(0, 1000, 4))
            plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
            plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
            plt.text(text_x, text_y_start - next(ge), 'path_x: {:.2f}m'.format(path_x))
            plt.text(text_x, text_y_start - next(ge), 'path_y: {:.2f}m'.format(path_y))
            plt.text(text_x, text_y_start - next(ge), 'devi_longi: {:.2f}m'.format(devi_longi))
            plt.text(text_x, text_y_start - next(ge), 'devi_lateral: {:.2f}m'.format(devi_lateral))
            plt.text(text_x, text_y_start - next(ge), 'devi_v: {:.2f}m/s'.format(devi_v))
            plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
            plt.text(text_x, text_y_start - next(ge), r'path_phi: ${:.2f}\degree$'.format(path_phi))
            plt.text(text_x, text_y_start - next(ge), r'devi_phi: ${:.2f}\degree$'.format(devi_phi))
            plt.text(text_x, text_y_start - next(ge), ' ')

            plt.text(text_x, text_y_start - next(ge), 'road_dist_l: {:.2f}m'.format(obs_road[0]))
            plt.text(text_x, text_y_start - next(ge), 'road_dist_r: {:.2f}m'.format(obs_road[1]))
            plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
            plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
            plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(path_v))
            plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))
            plt.text(text_x, text_y_start - next(ge), ' ')

            plt.text(text_x, text_y_start - next(ge), r'front_wheel: ${:.2f}\degree$'.format(self.ego_dynamics['front_wheel']))
            plt.text(text_x, text_y_start - next(ge), r'steer_wheel: ${:.2f}\degree$'.format(15 * self.ego_dynamics['front_wheel']))

            if self.action is not None:
                steer, a_x = self.action[0], self.action[1]
                plt.text(text_x, text_y_start - next(ge),
                         r'delta_wheel: {:.3f}rad (${:.3f}\degree$)'.format(steer, steer * 180 / np.pi))
                plt.text(text_x, text_y_start - next(ge), ' ')
                plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(self.ego_dynamics['acc']))
                plt.text(text_x, text_y_start - next(ge), 'delta_a_x: {:.2f}m/s^2'.format(a_x))

            text_x, text_y_start = 80, 60
            ge = iter(range(0, 1000, 4))

            # done info
            plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.done_type))

            # reward info
            if self.reward_info is not None:
                for key, val in self.reward_info.items():
                    print(key, val)
                    plt.text(text_x, text_y_start - next(ge), 'rew_{}: {:.4f}'.format(key, val))

            ax.add_collection(PatchCollection(patches, match_original=True))
            plt.xlim(-(Para.CROSSROAD_SIZE_LAT / 2 + extension), Para.CROSSROAD_SIZE_LAT / 2 + extension)
            plt.ylim(-(Para.CROSSROAD_SIZE_LON / 2 + extension), Para.CROSSROAD_SIZE_LON / 2 + extension)
            plt.pause(0.001)

    def set_traj(self, trajectory):
        """set the real trajectory to reconstruct observation"""
        self.ref_path = trajectory


def test_end2end():
    import tensorflow as tf
    env = CrossroadEnd2endMix()
    env_model = EnvironmentModel()
    obs, all_info = env.reset()
    i = 0
    batch_size = 1
    while i < 100000:
        for j in range(50):
            i += 1
            action = np.array([0.01, 0.3 + np.random.rand(1)*0.8], dtype=np.float32) # np.random.rand(1)*0.1 - 0.05
            obs, reward, done, info = env.step(action)
            env.render(weights=np.zeros(env.other_number,))
            # obses, actions = obs[np.newaxis, :], action[np.newaxis, :]
            # obses = tf.convert_to_tensor(np.tile(obs, (batch_size, 1)), dtype=tf.float32)
            # ref_points = tf.convert_to_tensor(np.tile(info['future_n_point'], (batch_size, 1, 1)), dtype=tf.float32)
            # road_edges = tf.convert_to_tensor(np.tile(info['future_n_edge'], (batch_size, 1, 1)), dtype=tf.float32)
            # actions = tf.convert_to_tensor(np.tile(actions, (batch_size, 1)), dtype=tf.float32)
            # random noise
            # import tensorflow_probability as tfp
            # tfd = tfp.distributions
            # tfb = tfp.bijectors
            # mean = np.zeros(shape=(obses.shape[0], env.adv_action_space.shape[0] * env.other_number), dtype=np.float32)
            # std1 = np.zeros(shape=(obses.shape[0], env.adv_action_space.shape[0] * (env.other_number - 1)), dtype=np.float32)
            # std2 = np.zeros(shape=(obses.shape[0], env.adv_action_space.shape[0] * 1), dtype=np.float32)
            # std = np.concatenate([std1, std2], axis=1)
            # act_dist = tfp.distributions.MultivariateNormalDiag(mean, std)
            # act_dist = tfp.distributions.TransformedDistribution(distribution=act_dist, bijector=tfb.Tanh())
            # adv_actions = act_dist.sample()
            # env_model.reset(obses, ref_points, road_edges)
            # for i in range(25):
            #     obses, rewards, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, \
            #         veh2bike4real, veh2person4real, _ = env_model.rollout_out(actions, adv_actions)
            #     if batch_size == 1:
            #         env_model.render()
            if done:
                break
        obs, _ = env.reset()


if __name__ == '__main__':
    test_end2end()