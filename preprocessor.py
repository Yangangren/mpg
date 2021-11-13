#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: preprocessor.py
# =====================================

import numpy as np
import tensorflow as tf
import math


def shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y):
    '''
    :param orig_x: original x
    :param orig_y: original y
    :param coordi_shift_x: coordi_shift_x along x axis
    :param coordi_shift_y: coordi_shift_y along y axis
    :return: shifted_x, shifted_y
    '''
    shifted_x = orig_x - coordi_shift_x
    shifted_y = orig_y - coordi_shift_y
    return shifted_x, shifted_y


def np_rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """

    coordi_rotate_d_in_rad = coordi_rotate_d * np.pi / 180
    transformed_x = orig_x * np.cos(coordi_rotate_d_in_rad) + orig_y * np.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * np.sin(coordi_rotate_d_in_rad) + orig_y * np.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    while np.any(transformed_d>180):
        transformed_d = np.where(transformed_d>180, transformed_d - 360, transformed_d)
    while np.any(transformed_d <= -180):
        transformed_d = np.where(transformed_d <= -180, transformed_d + 360, transformed_d)
    return transformed_x, transformed_y, transformed_d


def tf_rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """

    coordi_rotate_d_in_rad = coordi_rotate_d * np.pi / 180
    transformed_x = orig_x * tf.cos(coordi_rotate_d_in_rad) + orig_y * tf.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * tf.sin(coordi_rotate_d_in_rad) + orig_y * tf.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    while tf.reduce_any(transformed_d > 180):
        transformed_d = tf.where(transformed_d > 180, transformed_d - 360, transformed_d)
    while tf.reduce_any(transformed_d <= -180):
        transformed_d = tf.where(transformed_d <= -180, transformed_d + 360, transformed_d)
    return transformed_x, transformed_y, transformed_d


def np_shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = np_rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


def tf_shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = tf_rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.tf_mean = tf.Variable(tf.zeros(shape), dtype=tf.float32, trainable=False)
        self.tf_var = tf.Variable(tf.ones(shape), dtype=tf.float32, trainable=False)

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
        self.tf_mean.assign(tf.constant(self.mean))
        self.tf_var.assign(tf.constant(self.var))

    def set_params(self, mean, var, count):
        self.mean = mean
        self.var = var
        self.count = count
        self.tf_mean.assign(tf.constant(self.mean))
        self.tf_var.assign(tf.constant(self.var))

    def get_params(self, ):
        return self.mean, self.var, self.count


class Preprocessor(object):
    def __init__(self, ob_shape, obs_ptype='normalize', rew_ptype='normalize', rew_scale=None, rew_shift=None, args=None,
                 clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8, **kwargs):
        self.obs_ptype = obs_ptype
        self.ob_rms = RunningMeanStd(shape=ob_shape) if self.obs_ptype == 'normalize' else None
        self.rew_ptype = rew_ptype
        self.ret_rms = RunningMeanStd(shape=()) if self.rew_ptype == 'normalize' else None
        self.rew_scale = rew_scale if self.rew_ptype == 'scale' else None
        self.rew_shift = rew_shift if self.rew_ptype == 'scale' else None

        self.clipob = clipob
        self.cliprew = cliprew

        self.gamma = gamma
        self.epsilon = epsilon
        self.num_agent = None
        self.args = args
        self.obs_scale = self.args.obs_scale
        if 'num_agent' in kwargs.keys():
            self.ret = np.zeros(kwargs['num_agent'])
            self.num_agent = kwargs['num_agent']
        else:
            self.ret = 0

    def convert_ego_coordinate(self, obs):
        obs_ego = obs[:, :self.args.ego_info_dim]
        obs_ego_track = obs[:, self.args.ego_info_dim: self.args.ego_info_dim + self.args.track_info_dim]
        obs_ego_future_point = obs[:, self.args.ego_info_dim + self.args.track_info_dim:
               self.args.ego_info_dim + self.args.track_info_dim + self.args.future_point_num * self.args.per_path_info_dim]
        obs_ego_light_ref_task = obs[:, self.args.ego_info_dim + self.args.track_info_dim +
                                        self.args.future_point_num * self.args.per_path_info_dim: self.args.other_start_dim]

        # convert future points
        obs_ego_all = np.reshape(np.tile(obs_ego, (1, self.args.future_point_num)), (-1, self.args.ego_info_dim))
        obs_ego_future_point = np.reshape(obs_ego_future_point, (-1, self.args.per_path_info_dim))
        transformed_x, transformed_y, transformed_d = np_shift_and_rotate_coordination(obs_ego_future_point[:, 0], obs_ego_future_point[:, 1], obs_ego_future_point[:, 2],
                                                                                       obs_ego_all[:, 3], obs_ego_all[:, 4], obs_ego_all[:, 5])

        obs_ego_future_point_transformed = np.stack([transformed_x, transformed_y, transformed_d, obs_ego_future_point[:, -1]], axis=-1)
        obs_ego_future_point_transformed = np.reshape(obs_ego_future_point_transformed, (-1, self.args.future_point_num * self.args.per_path_info_dim))

        # convert obs_other
        obs_ego_all = np.reshape(np.tile(obs_ego, (1, self.args.other_number)), (-1, self.args.ego_info_dim))
        obs_other = np.reshape(obs[:, self.args.other_start_dim:], (-1, self.args.per_other_dim))

        transformed_x, transformed_y, transformed_d = np_shift_and_rotate_coordination(obs_other[:, 0], obs_other[:, 1], obs_other[:, 3],
                                                                                    obs_ego_all[:, 3], obs_ego_all[:, 4], obs_ego_all[:, 5])

        obs_other_transformed = np.stack([transformed_x, transformed_y, obs_other[:, 2], transformed_d], axis=-1)
        obs_other_transformed = np.concatenate([obs_other_transformed, obs_other[:, 4:]], axis=1)
        obs_other_reshaped = np.reshape(obs_other_transformed, (-1, self.args.per_other_dim * self.args.other_number))

        obs_transformed = np.concatenate([obs_ego, obs_ego_track, obs_ego_future_point_transformed,
                                          obs_ego_light_ref_task, obs_other_reshaped], axis=1)
        return np.squeeze(obs_transformed)

    def tf_convert_ego_coordinate(self, obs):
        obs_ego = obs[:, :self.args.ego_info_dim]
        obs_ego_track = obs[:, self.args.ego_info_dim: self.args.ego_info_dim + self.args.track_info_dim]
        obs_ego_future_point = obs[:, self.args.ego_info_dim + self.args.track_info_dim:
                                      self.args.ego_info_dim + self.args.track_info_dim + self.args.future_point_num * self.args.per_path_info_dim]
        obs_ego_light_ref_task = obs[:, self.args.ego_info_dim + self.args.track_info_dim +
                                        self.args.future_point_num * self.args.per_path_info_dim: self.args.other_start_dim]

        # convert future points
        obs_ego_all = tf.reshape(tf.tile(obs_ego, (1, self.args.future_point_num)), (-1, self.args.ego_info_dim))
        obs_ego_future_point = tf.reshape(obs_ego_future_point, (-1, self.args.per_path_info_dim))
        transformed_x, transformed_y, transformed_d = tf_shift_and_rotate_coordination(obs_ego_future_point[:, 0], obs_ego_future_point[:, 1], obs_ego_future_point[:, 2],
                                                                                       obs_ego_all[:, 3], obs_ego_all[:, 4], obs_ego_all[:, 5])

        obs_ego_future_point_transformed = tf.stack([transformed_x, transformed_y, transformed_d, obs_ego_future_point[:, -1]], axis=-1)
        obs_ego_future_point_transformed = tf.reshape(obs_ego_future_point_transformed, (-1, self.args.future_point_num * self.args.per_path_info_dim))

        # convert obs_other
        obs_ego_all = tf.reshape(tf.tile(obs_ego, (1, self.args.other_number)), (-1, self.args.ego_info_dim))
        obs_other = tf.reshape(obs[:, self.args.other_start_dim:], (-1, self.args.per_other_dim))

        transformed_x, transformed_y, transformed_d = np_shift_and_rotate_coordination(obs_other[:, 0], obs_other[:, 1], obs_other[:, 3],
                                                                                       obs_ego_all[:, 3], obs_ego_all[:, 4], obs_ego_all[:, 5])

        obs_other_transformed = tf.stack([transformed_x, transformed_y, obs_other[:, 2], transformed_d], axis=-1)
        obs_other_transformed = tf.concat([obs_other_transformed, obs_other[:, 4:]], axis=1)
        obs_other_reshaped = tf.reshape(obs_other_transformed, (-1, self.args.per_other_dim * self.args.other_number))

        obs_transformed = tf.concat([obs_ego, obs_ego_track, obs_ego_future_point_transformed,
                                     obs_ego_light_ref_task, obs_other_reshaped], axis=1)
        return obs_transformed

    def process_rew(self, rew, done):
        if self.rew_ptype == 'normalize':
            if self.num_agent is not None:
                self.ret = self.ret * self.gamma + rew
                self.ret_rms.update(self.ret)
                rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
                self.ret = np.where(done == 1, np.zeros(self.ret), self.ret)
            else:
                self.ret = self.ret * self.gamma + rew
                self.ret_rms.update(np.array([self.ret]))
                rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
                self.ret = 0 if done else self.ret
            return rew
        elif self.rew_ptype == 'scale':
            return (rew + self.rew_shift) * self.rew_scale
        else:
            return rew

    def process_obs(self, obs):
        if self.obs_scale:
            return obs * self.obs_scale
        else:
            return obs

    def np_process_obses(self, obses):
        if self.obs_scale:
            return obses * self.obs_scale
        else:
            return obses

    def np_process_rewards(self, rewards):
        if self.rew_ptype == 'normalize':
            rewards = np.clip(rewards / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            return rewards
        elif self.rew_ptype == 'scale':
            return (rewards + self.rew_shift) * self.rew_scale
        else:
            return rewards

    def tf_process_obses(self, obses):
        if self.obs_scale:
            return obses * tf.convert_to_tensor(self.obs_scale, dtype=tf.float32)
        else:
            return tf.convert_to_tensor(obses, dtype=tf.float32)

    def tf_process_rewards(self, rewards):
        with tf.name_scope('reward_process') as scope:
            if self.rew_ptype == 'normalize':
                rewards = tf.clip_by_value(rewards / tf.sqrt(self.ret_rms.tf_var + tf.constant(self.epsilon)),
                                           -self.cliprew,
                                           self.cliprew)
                return rewards
            elif self.rew_ptype == 'scale':
                return (rewards+tf.convert_to_tensor(self.rew_shift, dtype=tf.float32)) \
                       * tf.convert_to_tensor(self.rew_scale, dtype=tf.float32)
            else:
                return tf.convert_to_tensor(rewards, dtype=tf.float32)

    def set_params(self, params):
        if self.ob_rms:
            self.ob_rms.set_params(*params['ob_rms'])
        if self.ret_rms:
            self.ret_rms.set_params(*params['ret_rms'])

    def get_params(self):
        tmp = {}
        if self.ob_rms:
            tmp.update({'ob_rms': self.ob_rms.get_params()})
        if self.ret_rms:
            tmp.update({'ret_rms': self.ret_rms.get_params()})

        return tmp

    def save_params(self, save_dir):
        np.save(save_dir + '/ppc_params.npy', self.get_params())

    def load_params(self, load_dir):
        params = np.load(load_dir + '/ppc_params.npy', allow_pickle=True)
        params = params.item()
        self.set_params(params)
