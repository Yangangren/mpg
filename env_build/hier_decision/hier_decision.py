#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/04/12
# @Author  : Yangang Ren (Tsinghua Univ.)
# @FileName: hier_decision.py
# =====================================

import datetime
import shutil
import time
import json
import os
import heapq
from math import cos, sin, pi

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Wedge
import matplotlib.patches as mpatch
from matplotlib.collections import PatchCollection
import numpy as np
import tensorflow as tf
from shapely.geometry import Polygon, Point
from shapely.figures import plot_coords

from env_build.dynamics_and_models import EnvironmentModel, ReferencePath
from env_build.endtoend import CrossroadEnd2endMix
from env_build.endtoend_env_utils import *
from multi_path_generator import MultiPathGenerator
from env_build.utils.load_policy import LoadPolicy
from env_build.utils.misc import TimerStat
from env_build.utils.recorder import Recorder


class HierarchicalDecision(object):
    def __init__(self, train_exp_dir, ite, logdir=None):
        self.policy = LoadPolicy('../utils/models/{}'.format(train_exp_dir), ite)
        self.args = self.policy.args
        self.env = CrossroadEnd2endMix(mode='testing', future_point_num=self.args.num_rollout_list_for_policy_update[0])
        self.model = EnvironmentModel(mode='testing')
        self.recorder = Recorder()
        self.episode_counter = -1
        self.step_counter = -1
        self.obs = None
        self.stg = MultiPathGenerator()
        self.step_timer = TimerStat()
        self.ss_timer = TimerStat()
        self.logdir = logdir
        if self.logdir is not None:
            config = dict(train_exp_dir=train_exp_dir, ite=ite)
            with open(self.logdir + '/config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        self.fig_plot = 0
        self.safe_flag = False
        self.green_belt = plt.imread('green_belt.png')
        self.hist_posi = []
        self.old_index = 0
        self.surp_time_yellow = 0
        self.path_list = self.stg.generate_path(self.env.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        # ------------------build graph for tf.function in advance-----------------------
        obs, all_info = self.env.reset()
        mask, future_n_point, future_n_edge = all_info['mask'], all_info['future_n_point'], all_info['future_n_edge']
        obs = tf.convert_to_tensor(obs[np.newaxis, :], dtype=tf.float32)
        mask = tf.convert_to_tensor(mask[np.newaxis, :], dtype=tf.float32)
        future_n_point = tf.convert_to_tensor(future_n_point[np.newaxis, :], dtype=tf.float32)
        future_n_edge = tf.convert_to_tensor(future_n_edge[np.newaxis, :], dtype=tf.float32)
        self.is_safe(obs, mask, future_n_point, future_n_edge)
        self.policy.run_batch(obs, mask)
        self.policy.obj_value_batch(obs, mask)
        # ------------------build graph for tf.function in advance-----------------------
        self.reset()
        plt.ion()

    def reset(self,):
        self.recorder.reset()
        self.old_index = 0
        self.surp_time_yellow = 0
        self.hist_posi = []
        if self.logdir is not None:
            self.episode_counter += 1
            os.makedirs(self.logdir + '/episode{}/figs'.format(self.episode_counter))
            self.step_counter = -1
            self.recorder.save(self.logdir)
            if self.episode_counter >= 1:
                select_and_rename_snapshots_of_an_episode(self.logdir, self.episode_counter-1, 12)
                self.recorder.plot_and_save_ith_episode_curves(self.episode_counter-1,
                                                               self.logdir + '/episode{}/figs'.format(self.episode_counter-1),
                                                               isshow=False)
                # if self.env.done_type == 'good_done':
                #     select_and_rename_snapshots_of_an_episode(self.logdir, self.episode_counter-1, 12)
                #     self.recorder.plot_and_save_ith_episode_curves(self.episode_counter-1,
                #                                                    self.logdir + '/episode{}/figs'.format(self.episode_counter-1),
                #                                                    isshow=False)
                # else:
                #     shutil.rmtree(self.logdir + '/episode{}'.format(self.episode_counter-1))

        self.obs, _ = self.env.reset()
        self.path_list = self.stg.generate_path(self.env.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        return self.obs

    @tf.function
    def is_safe(self, obses, masks, future_n_point, future_n_edge):
        self.model.reset(obses, future_n_point, future_n_edge)
        punish = 0.
        for step in range(5):
            action, _, adv_action = self.policy.run_batch(obses, masks)
            obses, _, _, _, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, _ \
                = self.model.rollout_out(action, adv_action)
            punish += veh2veh4real[0] + veh2bike4real[0] + veh2person4real[0]
        return False if punish > 0 else True

    def safe_shield(self, real_obs, real_mask, real_future_n_point, real_future_n_edge):
        action_safe_set = [[[0., -1.]]]
        real_obs = tf.convert_to_tensor(real_obs[np.newaxis, :], dtype=tf.float32)
        real_mask = tf.convert_to_tensor(real_mask[np.newaxis, :], dtype=tf.float32)
        real_future_n_point = tf.convert_to_tensor(real_future_n_point[np.newaxis, :], dtype=tf.float32)
        real_future_n_edge = tf.convert_to_tensor(real_future_n_edge[np.newaxis, :], dtype=tf.float32)
        if self.safe_flag and (not self.is_safe(real_obs, real_mask, real_future_n_point, real_future_n_edge)):
            print('SAFETY SHIELD STARTED!')
            _, weight, adv_actions = self.policy.run_batch(real_obs, real_mask)
            return np.array(action_safe_set[0], dtype=np.float32).squeeze(0), weight.numpy()[0], adv_actions[0], True
        else:
            action, weight, adv_actions = self.policy.run_batch(real_obs, real_mask)
            return action.numpy()[0], weight.numpy()[0], adv_actions[0], False

    def _rule4light(self, real_light, obs_mask, exit='D'):
        obs, mask = self.env._convert_to_abso(obs_mask[0]), obs_mask[1]
        ego_x, ego_y = self.env.ego_dynamics['x'], self.env.ego_dynamics['y']
        ego_v_x = self.env.ego_dynamics['v_x']
        obs_other = np.reshape(obs[self.env.other_start_dim:], (-1, self.env.per_other_info_dim))

        if self.env.training_task == 'right':
            return 1
        if real_light == 0 or real_light == 1:
            # rule4green
            for index, item in enumerate(obs_other):
                if mask[index]:
                    other_x, other_y, other_l = item[0], item[1], item[4]
                    if ego_y - 30. < -Para.CROSSROAD_SIZE_LON / 2 and abs(ego_x - other_x) < Para.LANE_WIDTH_2 and 0 < other_y - ego_y < (Para.L + other_l):
                        return 3
            return 1

        elif real_light == 2:
            # rule4yellow
            if ego_y < -Para.CROSSROAD_SIZE_LON / 2 - 30 or ego_y - Para.L > -Para.CROSSROAD_SIZE_LON / 2:
                if ego_y - Para.L > -Para.CROSSROAD_SIZE_LON / 2:
                    self.surp_time_yellow = 0
                return 1

            else:
                self.surp_time_yellow += 1
                surp_dist = abs(ego_y - (-Para.CROSSROAD_SIZE_LON / 2))
                decel_time = abs(ego_v_x) / 2.0
                decel_dist = ego_v_x ** 2 / (2 * 2.0)

                if surp_dist >= decel_dist and decel_time <= (Para.YELLOW_TIME - self.surp_time_yellow / 10.) :
                    return 3
                elif surp_dist > decel_dist and decel_time > (Para.YELLOW_TIME - self.surp_time_yellow / 10.):
                    return 3
                elif surp_dist <= decel_dist and decel_time >= (Para.YELLOW_TIME - self.surp_time_yellow / 10.):
                    return 3
                elif surp_dist < decel_dist and decel_time < (Para.YELLOW_TIME - self.surp_time_yellow / 10.):
                    return 1
                else:
                    return 3

        else:
            # rule4red
            return 1 if ego_y - Para.L / 2 > -Para.CROSSROAD_SIZE_LON / 2 else 3

    def step(self):
        self.step_counter += 1
        # transform the traffic light
        trans_light = self._rule4light(self.env.light_phase, self.env.get_obs()[:2])
        self.path_list = self.stg.generate_path(self.env.training_task, LIGHT_PHASE_TO_GREEN_OR_RED[self.env.light_phase])
        with self.step_timer:
            obs_list, mask_list, future_n_point_list, future_n_edge_list = [], [], [], []
            # select optimal path
            for path in self.path_list:
                self.env.set_traj(path)
                vector, mask_vector, future_n_point, future_n_edge = self.env.get_obs()
                obs_list.append(vector)
                mask_list.append(mask_vector)
                future_n_point_list.append(future_n_point)
                future_n_edge_list.append(future_n_edge)
            all_obs = tf.stack(obs_list, axis=0).numpy()
            all_mask = tf.stack(mask_list, axis=0).numpy()

            path_values = self.policy.obj_value_batch(all_obs, all_mask).numpy()
            old_value = path_values[self.old_index]
            # value is to approximate (- sum of reward)
            new_index, new_value = int(np.argmin(path_values)), min(path_values)
            # rule for equal traj value
            path_index_error = []
            if self.step_counter % 5 == 0:
                if heapq.nsmallest(2, path_values)[0] == heapq.nsmallest(2, path_values)[1]:
                    for i in range(len(path_values)):
                        if path_values[i] == min(path_values):
                            index_error = abs(self.old_index - i)
                            path_index_error.append(index_error)
                    # new_index_new = min(path_index_error) + self.old_index if min(path_index_error) + self.old_index < 4 else self.old_index - min(path_index_error)
                    new_index_new = self.old_index - min(path_index_error) if self.old_index - min(path_index_error) > -1 else self.old_index + min(path_index_error)
                    new_value_new = path_values[new_index_new]
                    path_index = self.old_index if old_value - new_value_new < 0.1 else new_index_new
                else:
                    path_index = self.old_index if old_value - new_value < 0.1 else new_index
                self.old_index = path_index
            else:
                path_index = self.old_index
            self.env.set_traj(self.path_list[path_index])
            obs_real, mask_real, future_n_point_real, future_n_edge_real = obs_list[path_index], mask_list[path_index], future_n_point_list[path_index], future_n_edge_list[path_index]

            # obtain safe action
            with self.ss_timer:
                safe_action, weights, adv_actions, is_ss = self.safe_shield(obs_real, mask_real, future_n_point_real, future_n_edge_real)
            # print('ALL TIME:', self.step_timer.mean, 'ss', self.ss_timer.mean)

        self.recorder.record(obs_real, safe_action, self.step_timer.mean, path_index, path_values, self.ss_timer.mean, is_ss, np.array([self.env.light_phase, trans_light]))
        self.obs, r, done, info = self.env.step(safe_action)
        self.render(path_values, path_index,  weights)
        return done

    def render(self, path_values, path_index, weights):
        extension = 40
        dotted_line_style = '--'
        solid_line_style = '-'

        if not self.fig_plot:
            self.fig = plt.figure(figsize=(8, 8))
            self.fig_plot = 1

        plt.cla()
        ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
        ax.imshow(self.green_belt, extent=(-Para.CROSSROAD_SIZE_LAT / 2 - extension, -Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_L, Para.OFFSET_L + Para.GREEN_BELT_LAT))
        ax.imshow(self.green_belt, extent=(Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2 + extension, Para.OFFSET_R, Para.OFFSET_R + Para.GREEN_BELT_LAT))
        for ax in self.fig.get_axes():
            ax.axis('off')
        patches = []
        ax.axis("equal")

        # ----------arrow--------------
        # plt.arrow(lane_width / 2, -square_length / 2 - 10, 0, 5, color='orange')
        # plt.arrow(lane_width / 2, -square_length / 2 - 10 + 5, -0.5, 0, color='orange', head_width=1)
        # plt.arrow(lane_width * 1.5, -square_length / 2 - 10, 0, 4, color='orange', head_width=1)
        # plt.arrow(lane_width * 2.5, -square_length / 2 - 10, 0, 5, color='orange')
        # plt.arrow(lane_width * 2.5, -square_length / 2 - 10 + 5, 0.5, 0, color='orange', head_width=1)
        # ----------green belt--------------
        ax.add_patch(plt.Rectangle((Para.OFFSET_U, Para.CROSSROAD_SIZE_LON / 2),
                                   Para.GREEN_BELT_LON, extension, edgecolor='white', facecolor='darkgray',
                                   linewidth=1))
        ax.add_patch(plt.Rectangle((Para.OFFSET_D - Para.GREEN_BELT_LON, -Para.CROSSROAD_SIZE_LON / 2 - extension),
                                   Para.GREEN_BELT_LON, extension, edgecolor='white', facecolor='darkgray',
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
                                                                                  :Para.LANE_NUMBER_LAT_IN])], color='black')

        # traffic light
        v_light = self.env.light_phase
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

        # zebra crossing
        # j1, j2 = 20.5, 6.75
        # for ii in range(19):
        #     if ii <= 3:
        #         continue
        #     ax.add_patch(
        #         plt.Rectangle((-Para.CROSSROAD_SIZE_LON / 2 + j1 + ii * 1.6, -Para.CROSSROAD_SIZE_LON / 2 + 0.5),
        #                       0.8, 4,
        #                       color='lightgray', alpha=0.5))
        #     ii += 1
        # for ii in range(19):
        #     if ii <= 3:
        #         continue
        #     ax.add_patch(plt.Rectangle((-Para.CROSSROAD_SIZE_LON / 2 + j1 + ii * 1.6, Para.CROSSROAD_SIZE_LON / 2 - 0.5 - 4),
        #                       0.8, 4, color='lightgray', alpha=0.5))
        #     ii += 1
        # for ii in range(28):
        #     if ii <= 3:
        #         continue
        #     ax.add_patch(
        #         plt.Rectangle(
        #             (-Para.CROSSROAD_SIZE_LAT / 2 + 0.5, Para.CROSSROAD_SIZE_LAT / 2 - j2 - 0.8 - ii * 1.6), 4, 0.8,
        #             color='lightgray',
        #             alpha=0.5))
        #     ii += 1
        # for ii in range(28):
        #     if ii <= 3:
        #         continue
        #     ax.add_patch(
        #         plt.Rectangle(
        #             (Para.CROSSROAD_SIZE_LAT / 2 - 0.5 - 4, Para.CROSSROAD_SIZE_LAT / 2 - j2 - 0.8 - ii * 1.6), 4,
        #             0.8,
        #             color='lightgray',
        #             alpha=0.5))
        #     ii += 1

        def is_in_plot_area(x, y, tolerance=5):
            if -Para.CROSSROAD_SIZE_LAT / 2 - extension + tolerance < x < Para.CROSSROAD_SIZE_LAT / 2 + extension - tolerance and \
                    -Para.CROSSROAD_SIZE_LON / 2 - extension + tolerance < y < Para.CROSSROAD_SIZE_LON / 2 + extension - tolerance:
                return True
            else:
                return False

        def draw_sensor_range(x_ego, y_ego, a_ego, l_bias, w_bias, angle_bias, angle_range, dist_range, color):
            x_sensor = x_ego + l_bias * cos(a_ego) - w_bias * sin(a_ego)
            y_sensor = y_ego + l_bias * sin(a_ego) + w_bias * cos(a_ego)
            theta1 = a_ego + angle_bias - angle_range / 2
            theta2 = a_ego + angle_bias + angle_range / 2
            sensor = mpatch.Wedge([x_sensor, y_sensor], dist_range, theta1=theta1 * 180 / pi,
                                   theta2=theta2 * 180 / pi, fc=color, alpha=0.2, zorder=1)
            ax.add_patch(sensor)

        def draw_rotate_rec(type, x, y, a, l, w, color, linestyle='-', patch=False):
            RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
            RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
            LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
            if patch:
                if type.startswith('bicycle'):
                    item_color = 'purple'
                elif type.startswith('DEFAULT_PEDTYPE'):
                    item_color = 'lime'
                else:
                    item_color = 'gold'
                ax.add_patch(plt.Rectangle((x + LU_x, y + LU_y), w, l, edgecolor=item_color, facecolor=item_color, angle=-(90 - a), zorder=30))
            else:
                ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color, linestyle=linestyle, zorder=40)
                ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color, linestyle=linestyle, zorder=40)
                ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color, linestyle=linestyle, zorder=40)
                ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color, linestyle=linestyle, zorder=40)

        def draw_rotate_batch_rec(type, x, y, a, l, w, patch=False):
            for i in range(len(x)):
                patches.append(matplotlib.patches.Rectangle(np.array([-l[i] / 2 + x[i], -w[i] / 2 + y[i]]),
                                                            width=l[i], height=w[i],
                                                            fill=True,
                                                            facecolor='white',
                                                            edgecolor='k',
                                                            linestyle='-',
                                                            linewidth=1.0,
                                                            transform=Affine2D().rotate_deg_around(
                                                                *(x[i], y[i]), a[i])))

        def plot_phi_line(type, x, y, phi, color):
            if type.startswith('bicycle'):
                line_length = 2
                color = 'steelblue'
            elif type.startswith('DEFAULT_PEDTYPE'):
                line_length = 1
                color = 'purple'
            else:
                line_length = 3
                color = color
            x_forw, y_forw = x + line_length * cos(phi * pi / 180.), y + line_length * sin(phi * pi / 180.)
            plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

        # plot others
        filted_all_other = [item for item in self.env.all_other if is_in_plot_area(item['x'], item['y'])]
        other_xs = np.array([item['x'] for item in filted_all_other], np.float32)
        other_ys = np.array([item['y'] for item in filted_all_other], np.float32)
        other_as = np.array([item['phi'] for item in filted_all_other], np.float32)
        other_ls = np.array([item['l'] for item in filted_all_other], np.float32)
        other_ws = np.array([item['w'] for item in filted_all_other], np.float32)
        other_types = [item['type'] for item in filted_all_other]

        draw_rotate_batch_rec(other_types, other_xs, other_ys, other_as, other_ls, other_ws, patch=True)
        for index, item in enumerate(filted_all_other):
            x = item['x']
            y = item['y']
            phi = item['phi']
            type = item['type']
            plot_phi_line(type, x, y, phi, 'black')

        # detected traffic by sensors
        if self.env.detected_other is not None:
            for item in self.env.detected_other:
                item_x = item['x']
                item_y = item['y']
                item_phi = item['phi']
                item_l = item['l']
                item_w = item['w']
                item_type = item['type']
                draw_rotate_rec(item_type, item_x, item_y, item_phi, item_l, item_w, color='g', linestyle='-', patch=False)

        # plot interested others
        for i in range(len(self.env.interested_other)):
            item = self.env.interested_other[i]
            item_mask = item['exist']
            item_x = item['x']
            item_y = item['y']
            item_phi = item['phi']
            item_type = item['type']
            item_l, item_w = item['l'], item['w']
            if is_in_plot_area(item_x, item_y) and (item_mask == 1.0):
                draw_rotate_rec(item_type, item_x, item_y, item_phi, item_l, item_w, color='m', linestyle=':', patch=True)
                # if (weights is not None) and (weights[i] > 0.00):
                #     plt.text(item_x + 0.05, item_y + 0.15, "{:.2f}".format(weights[i]), color='purple', fontsize=12)

        # plot own car
        abso_obs = self.env._convert_to_abso(self.obs)
        obs_ego, obs_track, obs_road, obs_light, obs_task, obs_other = self.env._split_all(abso_obs)
        ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi, ego_front_wheel, ego_a_x = obs_ego
        self.hist_posi.append((ego_x, ego_y))
        devi_longi, devi_lateral, devi_phi, devi_v = obs_track

        # plot sensors
        draw_sensor_range(ego_x, ego_y, ego_phi * pi / 180, l_bias=self.env.ego_l / 2, w_bias=0, angle_bias=0,
                          angle_range=2 * pi, dist_range=70, color='thistle')
        draw_sensor_range(ego_x, ego_y, ego_phi * pi / 180, l_bias=self.env.ego_l / 2, w_bias=0, angle_bias=0,
                          angle_range=70 * pi / 180, dist_range=80, color="slategray")
        draw_sensor_range(ego_x, ego_y, ego_phi * pi / 180, l_bias=self.env.ego_l / 2, w_bias=0, angle_bias=0,
                          angle_range=90 * pi / 180, dist_range=60, color="slategray")

        plot_phi_line('self_car', ego_x, ego_y, ego_phi, 'fuchsia')
        draw_rotate_rec('self_car', ego_x, ego_y, ego_phi, self.env.ego_l, self.env.ego_w, 'fuchsia')

        ax.plot(self.env.road_edges[0, 0], self.env.road_edges[0, 1], '+', color='darkred')
        ax.plot(self.env.road_edges[1, 0], self.env.road_edges[1, 1], '.', color='slateblue')

        # plot history
        xs = [pos[0] for pos in self.hist_posi]
        ys = [pos[1] for pos in self.hist_posi]
        plt.scatter(np.array(xs), np.array(ys), color='fuchsia', alpha=0.1)

        # plot real time traj
        color = ['blue', 'coral', 'darkcyan', 'pink']
        for i, item in enumerate(self.path_list):
            if i == path_index:
                plt.plot(item.path[0], item.path[1], color=color[0], alpha=1.0, zorder=0)
            else:
                plt.plot(item.path[0], item.path[1], color=color[1], alpha=0.3, zorder=0)

        path_x, path_y, path_phi, path_v = self.env.ref_point[0], self.env.ref_point[1], self.env.ref_point[2], self.env.ref_point[3]

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
        plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(path_v))
        plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
        plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))
        plt.text(text_x, text_y_start - next(ge), ' ')

        plt.text(text_x, text_y_start - next(ge), r'front_wheel: ${:.2f}\degree$'.format(ego_front_wheel))
        plt.text(text_x, text_y_start - next(ge), 'steer_wheel: ${:.2f}\degree$'.format(15 * ego_front_wheel))
        if self.env.action is not None:
            delta_wheel, delta_a_x = self.env.action[0], self.env.action[1]
            plt.text(text_x, text_y_start - next(ge), r'delta_wheel: {:.2f}rad'.format(delta_wheel))
            plt.text(text_x, text_y_start - next(ge), r'delta_wheel: ${:.2f}\degree$'.format(delta_wheel * 180 / np.pi))
            plt.text(text_x, text_y_start - next(ge), ' ')
            plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(ego_a_x))
            plt.text(text_x, text_y_start - next(ge), 'delta_a_x: {:.2f}m/s^2'.format(delta_a_x))

        text_x, text_y_start = 80, 60
        ge = iter(range(0, 1000, 4))

        # done info
        plt.text(text_x, text_y_start - next(ge), 'done info: {}'.format(self.env.done_type))

        # reward info
        if self.env.reward_info is not None:
            for key, val in self.env.reward_info.items():
                plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))

        # indicator for trajectory selection
        text_x, text_y_start = 25, -30
        ge = iter(range(0, 1000, 6))
        if path_values is not None:
            for i, value in enumerate(path_values):
                if i == path_index:
                    plt.text(text_x, text_y_start - next(ge), 'Path cost={:.4f}'.format(value), fontsize=14,
                             color=color[0], fontstyle='italic')
                else:
                    plt.text(text_x, text_y_start - next(ge), 'Path cost={:.4f}'.format(value), fontsize=12,
                             color=color[1], fontstyle='italic')
        ax.add_collection(PatchCollection(patches, match_original=True))
        plt.xlim(-(Para.CROSSROAD_SIZE_LAT / 2 + extension), Para.CROSSROAD_SIZE_LAT / 2 + extension)
        plt.ylim(-(Para.CROSSROAD_SIZE_LON / 2 + extension), Para.CROSSROAD_SIZE_LON / 2 + extension)
        plt.pause(0.001)
        if self.logdir is not None:
            plt.savefig(self.logdir + '/episode{}'.format(self.episode_counter) + '/step{}.png'.format(self.step_counter))


def plot_and_save_ith_episode_data(logdir, i):
    recorder = Recorder()
    recorder.load(logdir)
    save_dir = logdir + '/episode{}/figs'.format(i)
    os.makedirs(save_dir, exist_ok=True)
    recorder.plot_and_save_ith_episode_curves(i, save_dir, True)


def main():
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = './results/{time}'.format(time=time_now)
    os.makedirs(logdir)
    hier_decision = HierarchicalDecision('experiment-2022-06-12-23-29-19', 300000, logdir)

    for i in range(300):
        for _ in range(350):
            done = hier_decision.step()
            if done:
                print('Episode {}, done type {}'.format(i, hier_decision.env.done_type))
                break
        hier_decision.reset()


def plot_static_path():
    extension = 40
    dotted_line_style = '--'
    solid_line_style = '-'
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes([0, 0, 1, 1])
    green_belt = plt.imread('green_belt.png')
    ax.imshow(green_belt, extent=(-Para.CROSSROAD_SIZE_LAT / 2 - extension, -Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_L, Para.OFFSET_L + Para.GREEN_BELT_LAT))
    ax.imshow(green_belt, extent=(Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2 + extension, Para.OFFSET_R, Para.OFFSET_R + Para.GREEN_BELT_LAT))
    for ax in fig.get_axes():
        ax.axis('off')
    ax.axis("equal")

    # ----------arrow--------------
    plt.arrow(Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON / 2 - 10, 0, 5, color='b')
    plt.arrow(Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON / 2 - 10 + 5, -0.5, 0, color='b', head_width=1)
    plt.arrow(Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON / 2 - 10 + 2., 0, 4, color='b', head_width=1)
    plt.arrow(Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_2 * 1.5, -Para.CROSSROAD_SIZE_LON / 2 - 10, 0, 5, color='b')
    plt.arrow(Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_2 * 1.5, -Para.CROSSROAD_SIZE_LON / 2 - 10 + 5, 0.5, 0, color='b', head_width=1)

    # ----------green belt--------------
    ax.add_patch(plt.Rectangle((Para.OFFSET_U, Para.CROSSROAD_SIZE_LON / 2),
                               Para.GREEN_BELT_LON, extension, edgecolor='white', facecolor='darkgray',
                               linewidth=1))
    ax.add_patch(plt.Rectangle((Para.OFFSET_D - Para.GREEN_BELT_LON, -Para.CROSSROAD_SIZE_LON / 2 - extension),
                               Para.GREEN_BELT_LON, extension, edgecolor='white', facecolor='darkgray',
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
    polygon = Polygon(Para.CROSSROAD_INTER)
    ax.plot(polygon.exterior.xy[0], polygon.exterior.xy[1], '-', color='black', linewidth=1.5, zorder=0, alpha=1.0)

    # stop line
    lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                       Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Down
    plt.plot([Para.OFFSET_D, Para.OFFSET_D + sum(lane_width_flag[:Para.LANE_NUMBER_LON_IN])],
             [-Para.CROSSROAD_SIZE_LON / 2, -Para.CROSSROAD_SIZE_LON / 2], color='gray')
    lane_width_flag = [Para.LANE_WIDTH_2, Para.LANE_WIDTH_3, Para.LANE_WIDTH_3,
                       Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]  # Up
    plt.plot([-sum(lane_width_flag[:Para.LANE_NUMBER_LON_IN]) + Para.OFFSET_U, Para.OFFSET_U],
             [Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2], color='gray')
    lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                       Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
    plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2],
             [Para.OFFSET_L, Para.OFFSET_L - sum(lane_width_flag[:Para.LANE_NUMBER_LAT_IN])],
             color='gray')  # left
    lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                       Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
    plt.plot([Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2], [Para.OFFSET_R + Para.GREEN_BELT_LAT,
                                                                          Para.OFFSET_R + Para.GREEN_BELT_LAT + sum(
                                                                              lane_width_flag[
                                                                              :Para.LANE_NUMBER_LAT_IN])], color='gray')
    color = ['steelblue', 'coral', 'cyan', 'm']
    for id, task in enumerate(['left', 'straight', 'right']):
        path = ReferencePath(task)
        path_list = path.path_list['green']
        control_points = path.control_points
        for i, (path_x, path_y, _, _) in enumerate(path_list):
            plt.plot(path_x[600:-600], path_y[600:-600], color=color[id])
    #     if task == 'straight':
    #         for i, four_points in enumerate(control_points):
    #             for point in four_points:
    #                 plt.scatter(point[0], point[1], color=color[i])
    #             plt.plot([four_points[0][0], four_points[1][0]], [four_points[0][1], four_points[1][1]], linestyle='--', color=color[i], alpha=0.3)
    #             plt.plot([four_points[1][0], four_points[2][0]], [four_points[1][1], four_points[2][1]], linestyle='--', color=color[i], alpha=0.3)
    #             plt.plot([four_points[2][0], four_points[3][0]], [four_points[2][1], four_points[3][1]], linestyle='--', color=color[i], alpha=0.3)
    #     plt.xlim(-(Para.CROSSROAD_SIZE_LAT / 2 + 20), Para.CROSSROAD_SIZE_LAT / 2 + 20)
    # plt.ylim(-(Para.CROSSROAD_SIZE_LON / 2 + 20), Para.CROSSROAD_SIZE_LON / 2 + 20)
    plt.savefig('./intersection.png')
    plt.show()


def select_and_rename_snapshots_of_an_episode(logdir, epinum, num):
    file_list = os.listdir(logdir + '/episode{}'.format(epinum))
    file_num = len(file_list) - 1
    interval = file_num // (num-1)
    start = file_num % (num-1)
    selected = [start//2] + [start//2+interval*i for i in range(1, num-1)]
    if file_num > 0:
        for i, j in enumerate(selected):
            shutil.copyfile(logdir + '/episode{}/step{}.png'.format(epinum, j),
                            logdir + '/episode{}/figs/{}.png'.format(epinum, i))


if __name__ == '__main__':
    main()
    # plot_static_path()
    # plot_and_save_ith_episode_data('./results/2022-03-02-16-09-58', 27)
    # select_and_rename_snapshots_of_an_episode('./results/good/2021-03-15-23-56-21', 0, 12)


