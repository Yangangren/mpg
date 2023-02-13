#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/06/07
# @Author  : Yangang Ren (Tsinghua Univ.)
# @FileName: dynamics_and_models.py
# =====================================

from math import pi, cos, sin

import bezier
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import logical_and
from matplotlib.patches import Wedge
from shapely.geometry import Polygon, Point, MultiPoint
from matplotlib.path import Path

from env_build.endtoend_env_utils import *

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


class VehicleDynamics(object):
    def __init__(self, ):
        self.vehicle_params = dict(C_f=-155495.0,  # front wheel cornering stiffness [N/rad]
                                   C_r=-155495.0,  # rear wheel cornering stiffness [N/rad]
                                   a=1.19,  # distance from CG to front axle [m]
                                   b=1.46,  # distance from CG to rear axle [m]
                                   mass=1520.,  # mass [kg]
                                   I_z=2642.,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=0.8,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

    def f_xu(self, states, actions, tau):  # states and actions are tensors, [[], [], ...]
        v_x, v_y, r, x, y, phi, front_wheel, a_x = states[:, 0], states[:, 1], states[:, 2], states[:, 3], \
                                                   states[:, 4], states[:, 5], states[:, 6], states[:, 7]
        phi = phi * np.pi / 180.
        front_wheel = front_wheel * np.pi / 180.
        delta_front_wheel, delta_a_x = actions[:, 0], actions[:, 1]
        C_f = tf.convert_to_tensor(self.vehicle_params['C_f'], dtype=tf.float32)
        C_r = tf.convert_to_tensor(self.vehicle_params['C_r'], dtype=tf.float32)
        a = tf.convert_to_tensor(self.vehicle_params['a'], dtype=tf.float32)
        b = tf.convert_to_tensor(self.vehicle_params['b'], dtype=tf.float32)
        mass = tf.convert_to_tensor(self.vehicle_params['mass'], dtype=tf.float32)
        I_z = tf.convert_to_tensor(self.vehicle_params['I_z'], dtype=tf.float32)
        miu = tf.convert_to_tensor(self.vehicle_params['miu'], dtype=tf.float32)
        g = tf.convert_to_tensor(self.vehicle_params['g'], dtype=tf.float32)

        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(a_x))
        F_xr = tf.where(a_x < 0, mass * a_x / 2, mass * a_x)
        miu_f = tf.sqrt(tf.square(miu * F_zf) - tf.square(F_xf)) / F_zf
        miu_r = tf.sqrt(tf.square(miu * F_zr) - tf.square(F_xr)) / F_zr
        alpha_f = tf.atan((v_y + a * r) / (v_x + 1e-8)) - front_wheel
        alpha_r = tf.atan((v_y - b * r) / (v_x + 1e-8))

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (
                              a * C_f - b * C_r) * r - tau * C_f * front_wheel * v_x - tau * mass * tf.square(
                          v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * front_wheel * v_x) / (
                              tau * (tf.square(a) * C_f + tf.square(b) * C_r) - I_z * v_x),
                      x + tau * (v_x * tf.cos(phi) - v_y * tf.sin(phi)),
                      y + tau * (v_x * tf.sin(phi) + v_y * tf.cos(phi)),
                      (phi + tau * r) * 180 / np.pi,  # deg
                      (front_wheel + tau * delta_front_wheel) * 180 / pi,  # deg
                      a_x + tau * delta_a_x
                      ]

        return tf.stack(next_state, 1), tf.stack([alpha_f, alpha_r, miu_f, miu_r], 1)

    def prediction(self, x_1, u_1, frequency):
        x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next, next_params


class EnvironmentModel(object):  # all tensors
    def __init__(self, mode='training'):
        self.mode = mode
        self.vehicle_dynamics = VehicleDynamics()
        self.base_frequency = 10.
        self.obses = None
        self.actions = None
        self.adv_actions = None
        self.reward_info = None
        self.future_n_points = None
        self.future_n_edges = None
        self.path_len = None
        self.veh_num = Para.MAX_VEH_NUM
        self.bike_num = Para.MAX_BIKE_NUM
        self.person_num = Para.MAX_PERSON_NUM
        self.ego_info_dim = Para.EGO_ENCODING_DIM
        self.track_info_dim = Para.TRACK_ENCODING_DIM
        self.road_info_dim = Para.ROAD_ENCODING_DIM
        self.light_info_dim = Para.LIGHT_ENCODING_DIM
        self.task_info_dim = Para.TASK_ENCODING_DIM
        self.adv_action_dim = 4
        self.other_number = sum([self.veh_num, self.bike_num, self.person_num])
        self.other_start_dim = sum([self.ego_info_dim, self.track_info_dim, self.road_info_dim, self.light_info_dim, self.task_info_dim])
        self.per_other_info_dim = Para.PER_OTHER_INFO_DIM

    def reset(self, obses, future_n_points, future_n_edges):  # input are all tensors
        self.obses = obses
        self.actions = None
        self.adv_actions = None
        self.reward_info = None
        self.future_n_points = future_n_points
        self.future_n_edges = future_n_edges
        self.path_len = self.future_n_points.shape[-1]

    def rollout_out(self, actions, adv_actions):
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)
            rewards, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, \
            veh2person4real, veh2speed4real, self.reward_info = self.compute_rewards(self.obses, self.actions)

            self.obses = self.compute_next_obses(self.obses, self.actions, adv_actions)

        return self.obses, rewards, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, \
               veh2bike4real, veh2person4real, veh2speed4real

    def compute_rewards(self, obses, actions):
        obses = self._convert_to_abso(obses)
        obses_ego, obses_track, obses_road, obses_light, obses_task, obses_other = self._split_all(obses)
        obses_bike, obses_person, obses_veh = self._split_other(obses_other)

        with tf.name_scope('compute_reward') as scope:
            bike_infos = obses_bike
            person_infos = obses_person
            veh_infos = obses_veh
            # reward related to control signal
            wheels, a_xs = obses_ego[:, 6] * np.pi / 180, obses_ego[:, 7]
            delta_wheels, delta_a_xs = actions[:, 0], actions[:, 1]
            punish_wheel, punish_delta_wheel = - tf.square(wheels), - tf.square(delta_wheels)
            punish_a_x, punish_delta_a_x = -tf.square(a_xs), - tf.square(delta_a_xs)

            # rewards related to ego stability
            punish_yaw_rate = -tf.square(obses_ego[:, 2])

            # rewards related to tracking error
            devi_lon = -tf.square(obses_track[:, 0])
            devi_lat = -tf.square(obses_track[:, 1])
            devi_phi = -tf.cast(tf.square(obses_track[:, 2] * np.pi / 180.), dtype=tf.float32)
            devi_v = -tf.square(obses_track[:, 3])

            # rewards related to veh2veh collision
            ego_lws = (Para.L - Para.W) / 2.
            ego_front_points = tf.cast(obses_ego[:, 3] + ego_lws * tf.cos(obses_ego[:, 5] * np.pi / 180.),
                                       dtype=tf.float32), \
                               tf.cast(obses_ego[:, 4] + ego_lws * tf.sin(obses_ego[:, 5] * np.pi / 180.),
                                       dtype=tf.float32)
            ego_rear_points = tf.cast(obses_ego[:, 3] - ego_lws * tf.cos(obses_ego[:, 5] * np.pi / 180.),
                                      dtype=tf.float32), \
                              tf.cast(obses_ego[:, 4] - ego_lws * tf.sin(obses_ego[:, 5] * np.pi / 180.),
                                      dtype=tf.float32)

            delta_ego_vs = obses_track[:, 3]  # delta_ego_vs = ego_vs - ref_vs
            veh2speed4training = tf.where(delta_ego_vs > 0.0, tf.square(delta_ego_vs), tf.zeros_like(veh_infos[:, 0]))
            veh2speed4real = tf.where(delta_ego_vs > 0.0, tf.square(delta_ego_vs), tf.zeros_like(veh_infos[:, 0]))

            veh2veh4real = tf.zeros_like(veh_infos[:, 0])
            veh2veh4training = tf.zeros_like(veh_infos[:, 0])

            for veh_index in range(self.veh_num):
                vehs = veh_infos[:, veh_index * self.per_other_info_dim:(veh_index + 1) * self.per_other_info_dim]
                veh_lws = (vehs[:, 4] - vehs[:, 5]) / 2.
                veh_xs, veh_ys, veh_vs, veh_phis = vehs[:, 0], vehs[:, 1], vehs[:, 2], vehs[:, 3]
                veh_front_points = tf.cast(veh_xs + veh_lws * tf.cos(veh_phis * np.pi / 180.), dtype=tf.float32), \
                                   tf.cast(veh_ys + veh_lws * tf.sin(veh_phis * np.pi / 180.), dtype=tf.float32)
                veh_rear_points = tf.cast(veh_xs - veh_lws * tf.cos(veh_phis * np.pi / 180.), dtype=tf.float32), \
                                  tf.cast(veh_ys - veh_lws * tf.sin(veh_phis * np.pi / 180.), dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    for veh_point in [veh_front_points, veh_rear_points]:
                        veh2veh_dist = tf.sqrt(tf.square(ego_point[0] - veh_point[0]) + tf.square(ego_point[1] - veh_point[1]))
                        veh2veh4training += tf.where(veh2veh_dist - 3.5 < 0,
                                                     tf.square(veh2veh_dist - 3.5), tf.zeros_like(veh_infos[:, 0]))
                        veh2veh4real += tf.where(veh2veh_dist - 2.5 < 0,
                                                 tf.square(veh2veh_dist - 2.5), tf.zeros_like(veh_infos[:, 0]))

            veh2bike4real = tf.zeros_like(veh_infos[:, 0])
            veh2bike4training = tf.zeros_like(veh_infos[:, 0])
            for bike_index in range(self.bike_num):
                bikes = bike_infos[:, bike_index * self.per_other_info_dim:(bike_index + 1) * self.per_other_info_dim]
                bike_lws = (bikes[:, 4] - bikes[:, 5]) / 2.
                bike_xs, bike_ys, bike_vs, bike_phis = bikes[:, 0], bikes[:, 1], bikes[:, 2], bikes[:, 3]
                middle_cond = logical_and(logical_and(bike_xs > -Para.CROSSROAD_SIZE_LAT / 2, bike_xs < Para.CROSSROAD_SIZE_LAT / 2),
                                          logical_and(bike_ys > -Para.CROSSROAD_SIZE_LON / 2, bike_ys < Para.CROSSROAD_SIZE_LON / 2))
                bike_front_points = tf.cast(bike_xs + bike_lws * tf.cos(bike_phis * np.pi / 180.), dtype=tf.float32), \
                                    tf.cast(bike_ys + bike_lws * tf.sin(bike_phis * np.pi / 180.), dtype=tf.float32)
                bike_rear_points = tf.cast(bike_xs - bike_lws * tf.cos(bike_phis * np.pi / 180.), dtype=tf.float32), \
                                   tf.cast(bike_ys - bike_lws * tf.sin(bike_phis * np.pi / 180.), dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    for bike_point in [bike_front_points, bike_rear_points]:
                        veh2bike_dist = tf.sqrt(tf.square(ego_point[0] - bike_point[0]) + tf.square(ego_point[1] - bike_point[1]))
                        veh2bike4training += tf.where(tf.logical_and(veh2bike_dist - 3.5 < 0, middle_cond),
                                                      tf.square(veh2bike_dist - 3.5), tf.zeros_like(veh_infos[:, 0]))
                        veh2bike4real += tf.where(tf.logical_and(veh2bike_dist - 2.5 < 0, middle_cond),
                                                  tf.square(veh2bike_dist - 2.5), tf.zeros_like(veh_infos[:, 0]))

            veh2person4real = tf.zeros_like(veh_infos[:, 0])
            veh2person4training = tf.zeros_like(veh_infos[:, 0])
            for person_index in range(self.person_num):
                persons = person_infos[:, person_index * self.per_other_info_dim:(person_index + 1) * self.per_other_info_dim]
                person_xs, person_ys, person_vs, person_phis = persons[:, 0], persons[:, 1], persons[:, 2], persons[:, 3]
                middle_cond = logical_and(logical_and(person_xs > -Para.CROSSROAD_SIZE_LAT / 2, person_xs < Para.CROSSROAD_SIZE_LAT / 2),
                                          logical_and(person_ys > -Para.CROSSROAD_SIZE_LON / 2, person_ys < Para.CROSSROAD_SIZE_LON / 2))
                person_point = tf.cast(person_xs, dtype=tf.float32), tf.cast(person_ys, dtype=tf.float32)
                for ego_point in [ego_front_points, ego_rear_points]:
                    veh2person_dist = tf.sqrt(tf.square(ego_point[0] - person_point[0]) + tf.square(ego_point[1] - person_point[1]))
                    veh2person4training += tf.where(tf.logical_and(veh2person_dist - 3.5 < 0, middle_cond),
                                                    tf.square(veh2person_dist - 3.5), tf.zeros_like(veh_infos[:, 0]))
                    veh2person4real += tf.where(tf.logical_and(veh2person_dist - 2.5 < 0, middle_cond),
                                                tf.square(veh2person_dist - 2.5), tf.zeros_like(veh_infos[:, 0]))

            veh2road4real = tf.zeros_like(veh_infos[:, 0])
            veh2road4training = tf.zeros_like(veh_infos[:, 0])

            veh2road4training += 1.0 * tf.where(obses_road[:, 0] < 1.0, tf.square(obses_road[:, 0] - 1.0), tf.zeros_like(veh_infos[:, 0]))
            veh2road4real += 1.0 * tf.where(obses_road[:, 0] < 1.0, tf.square(obses_road[:, 0] - 1.0), tf.zeros_like(veh_infos[:, 0]))
            veh2road4training += 1.0 * tf.where(obses_road[:, 1] > -1.0, tf.square(obses_road[:, 1] - (-1.0)), tf.zeros_like(veh_infos[:, 0]))
            veh2road4real += 1.0 * tf.where(obses_road[:, 1] > -1.0, tf.square(obses_road[:, 1] - (-1.0)), tf.zeros_like(veh_infos[:, 0]))

            rewards = 0.05 * devi_v + 0.8 * devi_lat + 30 * devi_phi + 0.02 * punish_yaw_rate + \
                      5 * punish_wheel + 0.4 * punish_delta_wheel + \
                      0.05 * punish_a_x + 0.0 * punish_delta_a_x

            punish_term_for_training = veh2veh4training + veh2road4training + veh2bike4training + veh2person4training + veh2speed4training
            real_punish_term = veh2veh4real + veh2road4real + veh2bike4real + veh2person4real + veh2speed4real

            reward_dict = dict(punish_wheel=punish_wheel,
                               punish_delta_wheel=punish_delta_wheel,
                               punish_a_x=punish_a_x,
                               punish_delta_a_x=punish_delta_a_x,
                               punish_yaw_rate=punish_yaw_rate,
                               devi_v=devi_v,
                               devi_longitudinal=devi_lon,
                               devi_lateral=devi_lat,
                               devi_phi=devi_phi,
                               scaled_punish_wheel=5.0 * punish_wheel,
                               scaled_punish_delta_wheel=0.4 * punish_delta_wheel,
                               scaled_punish_a_x=0.05 * punish_a_x,
                               scaled_punish_delta_a_x=0.0 * punish_delta_a_x,
                               scaled_punish_yaw_rate=0.02 * punish_yaw_rate,
                               scaled_devi_v=0.05 * devi_v,
                               scaled_devi_longitudinal=0.0 * devi_lon,
                               scaled_devi_lateral=0.8 * devi_lat,
                               scaled_devi_phi=30 * devi_phi,
                               veh2veh4training=veh2veh4training,
                               veh2road4training=veh2road4training,
                               veh2bike4training=veh2bike4training,
                               veh2person4training=veh2person4training,
                               veh2speedtraining=veh2speed4training,
                               veh2speedreal=veh2speed4real,
                               veh2veh4real=veh2veh4real,
                               veh2road4real=veh2road4real,
                               veh2bike2real=veh2bike4real,
                               veh2person2real=veh2person4real
                               )

            return rewards, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, \
                   veh2person4real, veh2speed4real, reward_dict

    def compute_next_obses(self, obses, actions, adv_actions):
        obses = self._convert_to_abso(obses)
        obses_ego, obses_track, obses_road, obses_light, obses_task, obses_other = self._split_all(obses)
        next_obses_ego = self._ego_predict(obses_ego, actions)
        next_obses_track, indexes = self._compute_next_track_info_vectorized(next_obses_ego)
        next_obses_road = self._compute_next_road_edge(indexes, next_obses_ego)
        next_obses_other = self._other_predict(obses_other, adv_actions)
        next_obses = tf.concat([next_obses_ego, next_obses_track, next_obses_road, obses_light, obses_task, next_obses_other], axis=-1)
        next_obses = self._convert_to_rela(next_obses)
        return next_obses

    def _find_closest_point_batch(self, xs, ys, phis, paths):
        ego_lws = Para.L / 2.
        xs_front, ys_front = tf.cast(xs + ego_lws * tf.cos(phis * np.pi / 180.), dtype=tf.float32),\
                             tf.cast(ys + ego_lws * tf.sin(phis * np.pi / 180), dtype=tf.float32)
        xs_front_tile = tf.tile(tf.reshape(xs_front, (-1, 1)), [1, self.path_len])
        ys_front_tile = tf.tile(tf.reshape(ys_front, (-1, 1)), [1, self.path_len])
        pathx_tile = paths[:, 0, :]
        pathy_tile = paths[:, 1, :]
        dist_array = tf.square(xs_front_tile - pathx_tile) + tf.square(ys_front_tile - pathy_tile)
        indexs_front = tf.argmin(dist_array, 1)

        xs_tile = tf.tile(tf.reshape(xs, (-1, 1)), [1, self.path_len])
        ys_tile = tf.tile(tf.reshape(ys, (-1, 1)), [1, self.path_len])
        dist_array = tf.square(xs_tile - pathx_tile) + tf.square(ys_tile - pathy_tile)
        indexs = tf.argmin(dist_array, 1)
        ref_points = tf.gather(paths, indices=indexs, axis=-1, batch_dims=1)
        return indexs_front, ref_points

    def _compute_next_track_info_vectorized(self, next_ego_infos):
        ego_vxs, ego_vys, ego_rs, ego_xs, ego_ys, ego_phis, ego_wheels, ego_a_xs = [next_ego_infos[:, i] for i in
                                                                                    range(self.ego_info_dim)]

        # find close point
        indexes, ref_points = self._find_closest_point_batch(ego_xs, ego_ys, ego_phis, self.future_n_points)

        ref_xs, ref_ys, ref_phis, ref_vs = [ref_points[:, i] for i in range(4)]
        ref_phis_rad = ref_phis * np.pi / 180

        vector_ref_phi = tf.stack([tf.cos(ref_phis_rad), tf.sin(ref_phis_rad)], axis=-1)
        vector_ref_phi_ccw_90 = tf.stack([-tf.sin(ref_phis_rad), tf.cos(ref_phis_rad)],
                                         axis=-1)  # ccw for counterclockwise
        vector_ego2ref = tf.stack([ref_xs - ego_xs, ref_ys - ego_ys], axis=-1)

        signed_dist_lon = tf.negative(tf.reduce_sum(vector_ego2ref * vector_ref_phi, axis=-1))
        signed_dist_lat = tf.negative(tf.reduce_sum(vector_ego2ref * vector_ref_phi_ccw_90, axis=-1))

        delta_phi = deal_with_phi_diff(ego_phis - ref_phis)
        delta_vs = ego_vxs - ref_vs
        return tf.stack([signed_dist_lon, signed_dist_lat, delta_phi, delta_vs], axis=-1), indexes

    def _compute_next_road_edge(self, indexes, obses_ego):
        road_edges = tf.gather(self.future_n_edges, indices=indexes, axis=-1, batch_dims=1)
        ego_pos = tf.tile(obses_ego[:, 3:5], (1, self.road_info_dim))
        self.road_edges = road_edges

        shifted_pos = road_edges - ego_pos
        coordi_rotate_d_in_rad = obses_ego[:, 5] * np.pi / 180
        delta_y_l = -shifted_pos[:, 0] * tf.sin(coordi_rotate_d_in_rad) + shifted_pos[:, 1] * tf.cos(coordi_rotate_d_in_rad)
        delta_y_r = -shifted_pos[:, 2] * tf.sin(coordi_rotate_d_in_rad) + shifted_pos[:, 3] * tf.cos(coordi_rotate_d_in_rad)
        return tf.stack([delta_y_l, delta_y_r], axis=1)

    def _convert_to_rela(self, obses):
        obses_ego, obses_track, obses_road, obses_light, obses_task, obses_other = self._split_all(obses)
        obses_other_reshape = self._reshape_other(obses_other)
        ego_x, ego_y = obses_ego[:, 3], obses_ego[:, 4]
        ego = tf.concat(
            [tf.stack([ego_x, ego_y], axis=-1), tf.zeros(shape=(ego_x.shape[0], self.per_other_info_dim - 2))],
            axis=-1)
        ego = tf.expand_dims(ego, 1)
        rela = obses_other_reshape - ego
        rela_obses_other = self._reshape_other(rela, reverse=True)
        return tf.concat([obses_ego, obses_track, obses_road, obses_light, obses_task, rela_obses_other], axis=-1)

    def _convert_to_abso(self, rela_obses):
        obses_ego, obses_track, obses_road, obses_light, obses_task, obses_other = self._split_all(rela_obses)
        obses_other_reshape = self._reshape_other(obses_other)
        ego_x, ego_y = obses_ego[:, 3], obses_ego[:, 4]
        ego = tf.concat(
            [tf.stack([ego_x, ego_y], axis=-1), tf.zeros(shape=(ego_x.shape[0], self.per_other_info_dim - 2))],
            axis=-1)
        ego = tf.expand_dims(ego, 1)
        abso = obses_other_reshape + ego
        abso_obses_other = self._reshape_other(abso, reverse=True)

        return tf.concat([obses_ego, obses_track, obses_road, obses_light, obses_task, abso_obses_other], axis=-1)

    def _ego_predict(self, ego_infos, actions):
        ego_next_infos, _ = self.vehicle_dynamics.prediction(ego_infos[:, :self.ego_info_dim], actions, self.base_frequency)
        v_xs, v_ys, rs, xs, ys, phis, wheels, a_xs = ego_next_infos[:, 0], ego_next_infos[:, 1], ego_next_infos[:, 2], \
                                                     ego_next_infos[:, 3], ego_next_infos[:, 4], ego_next_infos[:, 5], \
                                                     ego_next_infos[:, 6], ego_next_infos[:, 7]
        v_xs = tf.clip_by_value(v_xs, 0., 35.)  # todo
        ego_next_infos = tf.stack([v_xs, v_ys, rs, xs, ys, phis, wheels, a_xs], axis=-1)
        return ego_next_infos

    def _other_predict(self, obses_other, adv_noises):
        obses_other_reshape = self._reshape_other(obses_other)
        adv_noises_reshape = tf.reshape(adv_noises, (-1, self.other_number, self.adv_action_dim))

        xs, ys, vs, phis, turn_rad = obses_other_reshape[:, :, 0], obses_other_reshape[:, :, 1], \
                                     obses_other_reshape[:, :, 2], obses_other_reshape[:, :, 3], \
                                     obses_other_reshape[:, :, -1]
        phis_rad = phis * np.pi / 180.

        xs_noise, ys_noise, vs_noise, phis_noise_rad = adv_noises_reshape[:, :, 0], adv_noises_reshape[:, :, 1], \
                                                       adv_noises_reshape[:, :, 2], adv_noises_reshape[:, :, 3]

        xs_delta = vs / self.base_frequency * tf.cos(phis_rad)
        ys_delta = vs / self.base_frequency * tf.sin(phis_rad)
        phis_rad_delta = vs / self.base_frequency * turn_rad

        next_xs, next_ys, next_vs, next_phis_rad = xs + xs_delta + xs_noise, ys + ys_delta + ys_noise, \
                                                   vs + vs_noise, phis_rad + phis_rad_delta + phis_noise_rad
        next_vs = tf.where(next_vs < 0., tf.zeros_like(next_vs), next_vs)
        next_phis_rad = tf.where(next_phis_rad > np.pi, next_phis_rad - 2 * np.pi, next_phis_rad)
        next_phis_rad = tf.where(next_phis_rad <= -np.pi, next_phis_rad + 2 * np.pi, next_phis_rad)
        next_phis = next_phis_rad * 180 / np.pi
        next_info = tf.concat([tf.stack([next_xs, next_ys, next_vs, next_phis], -1), obses_other_reshape[:, :, 4:]],
                              axis=-1)
        next_obses_other = self._reshape_other(next_info, reverse=True)
        return next_obses_other

    def _split_all(self, obses):
        obses_ego = obses[:, :self.ego_info_dim]
        obses_track = obses[:, self.ego_info_dim:self.ego_info_dim + self.track_info_dim]
        obses_road = obses[:, self.ego_info_dim + self.track_info_dim:
                              self.ego_info_dim + self.track_info_dim + self.road_info_dim]
        obses_light = obses[:, self.ego_info_dim + self.track_info_dim + self.road_info_dim:
                               self.ego_info_dim + self.track_info_dim + self.road_info_dim + self.light_info_dim]
        obses_task = obses[:, self.ego_info_dim + self.track_info_dim + self.road_info_dim + self.light_info_dim:
                              self.ego_info_dim + self.track_info_dim + self.road_info_dim + self.light_info_dim + self.task_info_dim]
        obses_other = obses[:, self.other_start_dim:]

        return obses_ego, obses_track, obses_road, obses_light, obses_task, obses_other

    def _split_other(self, obses_other):
        obses_bike = obses_other[:, :self.bike_num * self.per_other_info_dim]
        obses_person = obses_other[:, self.bike_num * self.per_other_info_dim:
                                      (self.bike_num + self.person_num) * self.per_other_info_dim]
        obses_veh = obses_other[:, (self.bike_num + self.person_num) * self.per_other_info_dim:]
        return obses_bike, obses_person, obses_veh

    def _reshape_other(self, obses_other, reverse=False):
        if reverse:
            return tf.reshape(obses_other, (-1, self.other_number * self.per_other_info_dim))
        else:
            return tf.reshape(obses_other, (-1, self.other_number, self.per_other_info_dim))

    def _action_transformation_for_end2end(self, actions):  # [-1, 1]
        actions = tf.clip_by_value(actions, -1.05, 1.05)
        delta_wheel_norm, delta_a_x_norm = actions[:, 0], actions[:, 1]
        delta_wheel = 0.4 * delta_wheel_norm
        delta_a_x = 4.5 * delta_a_x_norm

        return tf.stack([delta_wheel, delta_a_x], 1)

    def render(self, mode='human'):
        if mode == 'human':
            # plot basic map
            extension = 40
            dotted_line_style = '--'
            solid_line_style = '-'

            plt.cla()
            ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axis("equal")

            ax.add_patch(plt.Rectangle((-Para.CROSSROAD_SIZE_LAT / 2 - extension, Para.OFFSET_L),
                                       extension, Para.GREEN_BELT_LAT, edgecolor='black', facecolor='green',
                                       linewidth=1))
            ax.add_patch(plt.Rectangle((Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_R),
                                       extension, Para.GREEN_BELT_LAT, edgecolor='black', facecolor='green',
                                       linewidth=1))
            ax.add_patch(plt.Rectangle((Para.OFFSET_U, Para.CROSSROAD_SIZE_LON / 2),
                                       Para.GREEN_BELT_LON, extension, edgecolor='black', facecolor='green',
                                       linewidth=1))
            ax.add_patch(plt.Rectangle((Para.OFFSET_D - Para.GREEN_BELT_LON, -Para.CROSSROAD_SIZE_LON / 2 - extension),
                                       Para.GREEN_BELT_LON, extension, edgecolor='black', facecolor='green',
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
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_U - (
                    Para.LANE_NUMBER_LON_IN - 1) * Para.LANE_WIDTH_3 - Para.LANE_WIDTH_2 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH],
                     [
                         Para.OFFSET_L + Para.GREEN_BELT_LAT + Para.LANE_NUMBER_LAT_OUT * Para.LANE_WIDTH_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH,
                         Para.CROSSROAD_SIZE_LON / 2],
                     color='black', linewidth=1)
            plt.plot([-Para.CROSSROAD_SIZE_LAT / 2,
                      Para.OFFSET_D - Para.GREEN_BELT_LON - Para.LANE_NUMBER_LON_OUT * Para.LANE_WIDTH_3 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH],
                     [
                         Para.OFFSET_L - Para.LANE_NUMBER_LAT_IN * Para.LANE_WIDTH_1 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH,
                         -Para.CROSSROAD_SIZE_LON / 2],
                     color='black', linewidth=1)
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
                      Para.OFFSET_D + (
                              Para.LANE_NUMBER_LON_IN - 1) * Para.LANE_WIDTH_3 + Para.LANE_WIDTH_2 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH],
                     [
                         Para.OFFSET_R - Para.LANE_NUMBER_LAT_OUT * Para.LANE_WIDTH_1 - Para.BIKE_LANE_WIDTH - Para.PERSON_LANE_WIDTH,
                         -Para.CROSSROAD_SIZE_LON / 2],
                     color='black', linewidth=1)
            plt.plot([Para.CROSSROAD_SIZE_LAT / 2,
                      Para.OFFSET_U + Para.GREEN_BELT_LON + Para.LANE_NUMBER_LON_OUT * Para.LANE_WIDTH_3 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH],
                     [
                         Para.OFFSET_R + Para.GREEN_BELT_LAT + Para.LANE_NUMBER_LAT_IN * Para.LANE_WIDTH_1 + Para.BIKE_LANE_WIDTH + Para.PERSON_LANE_WIDTH,
                         Para.CROSSROAD_SIZE_LON / 2],
                     color='black', linewidth=1)

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
            v_light = 1
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
            #     ax.add_patch(
            #         plt.Rectangle((-Para.CROSSROAD_SIZE_LON / 2 + j1 + ii * 1.6, Para.CROSSROAD_SIZE_LON / 2 - 0.5 - 4),
            #                       0.8, 4,
            #                       color='lightgray', alpha=0.5))
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
                    ax.add_patch(plt.Rectangle((x + LU_x, y + LU_y), w, l, edgecolor=item_color, facecolor=item_color,
                                               angle=-(90 - a), zorder=30))
                else:
                    ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color, linestyle=linestyle)
                    ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color, linestyle=linestyle)
                    ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color, linestyle=linestyle)
                    ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color, linestyle=linestyle)

            def plot_phi_line(type, x, y, phi, color):
                if type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                    line_length = 2
                elif type == 'DEFAULT_PEDTYPE':
                    line_length = 1
                else:
                    line_length = 5
                x_forw, y_forw = x + line_length * cos(phi * pi / 180.), \
                                 y + line_length * sin(phi * pi / 180.)
                plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

            obses = self._convert_to_abso(self.obses)
            obses = obses.numpy()
            obses_ego, obses_track, obses_road, obses_light, obses_task, obses_other = self._split_all(obses)

            # plot others
            for index in range(self.veh_num + self.bike_num + self.person_num):
                item = obses_other[:, self.per_other_info_dim * index:self.per_other_info_dim * (index + 1)]
                other_x, other_y, other_v, other_phi, other_l, other_w = item[:, 0], item[:, 1], item[:, 2], item[:, 3], item[:, 4], item[:, 5]
                if index < self.bike_num:
                    type = 'bicycle_1'
                elif self.bike_num <= index < self.person_num + self.bike_num:
                    type = 'DEFAULT_PEDTYPE'
                else:
                    type = 'veh'
                if 1:
                    plot_phi_line(type, other_x, other_y, other_phi, 'black')
                    draw_rotate_rec(type, other_x, other_y, other_phi, other_l, other_w, 'black')

            # plot own car
            delta_lon, delta_lat, delta_phi, delta_v = np.squeeze(obses_track, axis=0)
            ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi, ego_wheel, ego_ax = np.squeeze(obses_ego, axis=0)

            plot_phi_line('self_car', ego_x, ego_y, ego_phi, 'fuchsia')
            draw_rotate_rec("self_car", ego_x, ego_y, ego_phi, Para.L, Para.W, 'fuchsia')

            plt.plot(self.future_n_points[0][0], self.future_n_points[0][1], 'g.')
            plt.plot(self.future_n_edges[0][0], self.future_n_edges[0][1], '*', color='darkred')
            plt.plot(self.future_n_edges[0][2], self.future_n_edges[0][3], 'x', color='slateblue')

            road_edges = np.squeeze(self.road_edges, axis=0)
            obses_road = np.squeeze(obses_road, axis=0)
            for item in range(0, len(road_edges), 2):
                ax.plot(road_edges[item], road_edges[item+1], 'o', color='pink')

            # plot text
            text_x, text_y_start = -110, 60
            ge = iter(range(0, 1000, 4))
            plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
            plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
            plt.text(text_x, text_y_start - next(ge), 'delta_lon: {:.2f}m'.format(delta_lon))
            plt.text(text_x, text_y_start - next(ge), 'delta_lat: {:.2f}m'.format(delta_lat))
            plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
            plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))

            plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
            # plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.exp_v))
            plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))

            plt.text(text_x, text_y_start - next(ge), r'front_wheel: ${:.2f}\degree$'.format(ego_wheel))
            plt.text(text_x, text_y_start - next(ge), r'acc: ${:.2f}m/s^2$'.format(ego_ax))
            plt.text(text_x, text_y_start - next(ge), ' ')

            if self.actions is not None:
                delta_steer, delta_a_x = self.actions[0, 0], self.actions[0, 1]
                plt.text(text_x, text_y_start - next(ge), r'delta_steer: {:.3f}rad (${:.2f}\degree$)'.format(delta_steer, delta_steer * 180 / np.pi))
                plt.text(text_x, text_y_start - next(ge), r'delta_a_x: ${:.2f}m/s^3$'.format(delta_a_x))

            plt.text(text_x, text_y_start - next(ge), ' ')
            plt.text(text_x, text_y_start - next(ge), 'road_dist_l: {:.2f}m'.format(obses_road[0]))
            plt.text(text_x, text_y_start - next(ge), 'road_dist_r: {:.2f}m'.format(obses_road[1]))

            text_x, text_y_start = 80, 60
            ge = iter(range(0, 1000, 4))

            # reward info
            if self.reward_info is not None:
                for key, val in self.reward_info.items():
                    plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, np.squeeze(val.numpy(), axis=0)))

            plt.xlim(-(Para.CROSSROAD_SIZE_LAT / 2 + extension), Para.CROSSROAD_SIZE_LAT / 2 + extension)
            plt.ylim(-(Para.CROSSROAD_SIZE_LON / 2 + extension), Para.CROSSROAD_SIZE_LON / 2 + extension)
            plt.pause(0.001)


def deal_with_phi_diff(phi_diff):
    phi_diff = tf.where(phi_diff > 180., phi_diff - 360., phi_diff)
    phi_diff = tf.where(phi_diff < -180., phi_diff + 360., phi_diff)
    return phi_diff


def get_bezier_middle_point(control_point1, control_point4, k=0.6):
    x1, y1, phi1 = control_point1
    x4, y4, phi4 = control_point4
    phi1 = phi1 * pi / 180
    phi4 = phi4 * pi / 180

    x2 = x1 * ((cos(phi1) ** 2) * k + sin(phi1) ** 2) + y1 * (-sin(phi1) * cos(phi1) * (1 - k)) + x4 * (
                (cos(phi1) ** 2) * (1 - k)) + y4 * (sin(phi1) * cos(phi1) * (1 - k))
    y2 = x1 * (-sin(phi1) * cos(phi1) * (1 - k)) + y1 * (cos(phi1) ** 2 + (sin(phi1) ** 2) * k) + x4 * (
                sin(phi1) * cos(phi1) * (1 - k)) + y4 * ((sin(phi1) ** 2) * (1 - k))

    x3 = x1 * (cos(phi4) ** 2) * (1 - k) + y1 * (sin(phi4) * cos(phi4) * (1 - k)) + x4 * (
                (cos(phi4) ** 2) * k + sin(phi4) ** 2) + y4 * (-sin(phi4) * cos(phi4) * (1 - k))
    y3 = x1 * (sin(phi4) * cos(phi4) * (1 - k)) + y1 * ((sin(phi4) ** 2) * (1 - k)) + x4 * (
                -sin(phi4) * cos(phi4) * (1 - k)) + y4 * (cos(phi4) ** 2 + (sin(phi4) ** 2) * k)

    control_point2 = x2, y2
    control_point3 = x3, y3
    return control_point2, control_point3


class ReferencePath(object):
    def __init__(self, task, green_or_red='green'):
        self.task = task
        self.path_list = {}
        self.path_len_list = []
        self.control_points = []
        self.green_or_red = green_or_red
        self._construct_ref_path(self.task)
        self.path = None
        self.path_index = None
        self.set_path(green_or_red)

    def set_path(self, green_or_red='green', path_index=None):
        if path_index is None:
            path_index = np.random.choice(len(self.path_list[self.green_or_red]))
        self.path_index = path_index
        self.path = self.path_list[green_or_red][path_index]

    def get_future_n_point(self, ego_x, ego_y, n, dt=0.1):  # not include the current closest point
        idx, _ = self._find_closest_point(ego_x, ego_y)
        future_n_x, future_n_y, future_n_phi, future_n_v = [], [], [], []
        for _ in range(n):
            x, y, phi, v = self.idx2point(idx)
            ds = v * dt
            s = 0
            while s < ds:
                if idx + 1 >= len(self.path[0]):
                    break
                next_x, next_y, _, _ = self.idx2point(idx + 1)
                s += np.sqrt(np.square(next_x - x) + np.square(next_y - y))
                x, y = next_x, next_y
                idx += 1
            x, y, phi, v = self.idx2point(idx)
            future_n_x.append(x)
            future_n_y.append(y)
            future_n_phi.append(phi)
            future_n_v.append(v)
        future_n_point = np.stack([np.array(future_n_x, dtype=np.float32), np.array(future_n_y, dtype=np.float32),
                                   np.array(future_n_phi, dtype=np.float32), np.array(future_n_v, dtype=np.float32)],
                                  axis=0)
        return future_n_point

    def tracking_error_vector(self, ego_x, ego_y, ego_phi, ego_v):
        _, (x0, y0, phi0, v0) = self._find_closest_point(ego_x, ego_y)
        phi0_rad = phi0 * np.pi / 180
        # np.sin(phi0_rad) * x - np.cos(phi0_rad) * y - np.sin(phi0_rad) * x0 + np.cos(phi0_rad) * y0 = 0
        a, b, c = (x0, y0), (x0 + cos(phi0_rad), y0 + sin(phi0_rad)), (ego_x, ego_y)
        dist_a2c = np.sqrt(np.square(ego_x - x0) + np.square(ego_y - y0))
        dist_c2line = abs(sin(phi0_rad) * ego_x - cos(phi0_rad) * ego_y - sin(phi0_rad) * x0 + cos(phi0_rad) * y0)
        signed_dist_lateral = self._judge_sign_left_or_right(a, b, c) * dist_c2line
        signed_dist_longi = self._judge_sign_ahead_or_behind(a, b, c) * np.sqrt(
            np.abs(dist_a2c ** 2 - dist_c2line ** 2))
        return np.array([signed_dist_longi, signed_dist_lateral, deal_with_phi_diff(ego_phi - phi0), ego_v - v0])

    def tracking_error_vector_vectorized(self, ego_x, ego_y, ego_phi, ego_v):
        ego_lws = Para.L / 2.
        ego_front_x, ego_front_y = ego_x + ego_lws * np.cos(ego_phi * np.pi / 180.), ego_y + ego_lws * np.sin(ego_phi * np.pi / 180)

        _, (x0, y0, phi0, v0) = self._find_closest_point(ego_x, ego_y)
        _, (x1, y1, phi1, v1) = self._find_closest_point(ego_front_x, ego_front_y)
        phi0_rad = phi0 * np.pi / 180

        vector_ref_phi = np.array([np.cos(phi0_rad), np.sin(phi0_rad)])
        vector_ref_phi_ccw_90 = np.array([-np.sin(phi0_rad), np.cos(phi0_rad)])  # ccw for counterclockwise
        vector_ego2ref = np.array([x0 - ego_x, y0 - ego_y])

        signed_dist_longi = np.negative(np.dot(vector_ego2ref, vector_ref_phi))
        signed_dist_lateral = np.negative(np.dot(vector_ego2ref, vector_ref_phi_ccw_90))

        return np.array([signed_dist_longi, signed_dist_lateral, deal_with_phi_diff(ego_phi - phi0), ego_v - v0]), (x0, y0, phi0, v0), (x1, y1, phi1, v1)

    def idx2point(self, idx):
        return self.path[0][idx], self.path[1][idx], self.path[2][idx], self.path[3][idx]

    def _judge_sign_left_or_right(self, a, b, c):
        # see https://www.cnblogs.com/manyou/archive/2012/02/23/2365538.html for reference
        # return +1 for left and -1 for right in our case
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        featured = (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)
        if abs(featured) < 1e-8:
            return 0.
        else:
            return featured / abs(featured)

    def _judge_sign_ahead_or_behind(self, a, b, c):
        # return +1 if head else -1
        x1, y1 = a
        x2, y2 = b
        x3, y3 = c
        vector1 = np.array([x2 - x1, y2 - y1])
        vector2 = np.array([x3 - x1, y3 - y1])
        mul = np.sum(vector1 * vector2)
        if abs(mul) < 1e-8:
            return 0.
        else:
            return mul / np.abs(mul)

    def _construct_ref_path(self, task):
        sl = 40                                                   # straight length
        dece_dist = 20
        meter_pointnum_ratio = 30
        planed_trj_g = []
        planed_trj_r = []

        if task == 'left':
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            end_offsets = [Para.OFFSET_L + Para.GREEN_BELT_LAT + sum(lane_width_flag[:i]) + 0.5 * lane_width_flag[i] for
                           i in range(Para.LANE_NUMBER_LAT_OUT)]
            start_offsets = [Para.OFFSET_D + Para.LANE_WIDTH_2 * 0.5]
            rho = [3 / 10]                 # control the shape and number of bezier curve
            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    for k in rho:
                        control_point1 = start_offset, -Para.CROSSROAD_SIZE_LON / 2, 90
                        control_point4 = -Para.CROSSROAD_SIZE_LAT / 2, end_offset, 180
                        control_point2, control_point3 = get_bezier_middle_point(control_point1, control_point4, k)
                        self.control_points.append([control_point1, control_point2, control_point3, control_point4])

                        node = np.asfortranarray(
                            [[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                             [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                            dtype=np.float32)
                        curve = bezier.Curve(node, degree=3)
                        s_vals = np.linspace(0, 1.0, int(curve.length) * meter_pointnum_ratio)
                        trj_data = curve.evaluate_multi(s_vals).astype(np.float32)

                        start_straight_line_x = (Para.OFFSET_D + Para.LANE_WIDTH_2 * 0.5) * np.ones(
                            shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[:-1]
                        start_straight_line_y = np.linspace(-Para.CROSSROAD_SIZE_LON / 2 - sl, -Para.CROSSROAD_SIZE_LON / 2,
                                                            sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                        end_straight_line_x = np.linspace(-Para.CROSSROAD_SIZE_LAT / 2, -Para.CROSSROAD_SIZE_LAT / 2 - sl,
                                                          sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                        end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                        planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                     np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)

                        xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                        xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                        phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi

                        vs_green = np.array([8.33] * len(start_straight_line_x) + [7.0] * (len(trj_data[0]) - 1) + [8.33] *
                                            len(end_straight_line_x), dtype=np.float32)
                        vs_red_0 = np.array(
                            [8.33] * (len(start_straight_line_x) - meter_pointnum_ratio * (sl - dece_dist + int(Para.L))),
                            dtype=np.float32)
                        vs_red_1 = np.linspace(8.33, 0.0, meter_pointnum_ratio * dece_dist, endpoint=True, dtype=np.float32)
                        vs_red_2 = np.array(
                            [0.0] * (meter_pointnum_ratio * int(Para.L / 2) + len(trj_data[0]) - 1) + [0.0] * len(
                                end_straight_line_x), dtype=np.float32)
                        vs_red = np.append(np.append(vs_red_0, vs_red_1), vs_red_2)
                        planed_trj_green = xs_1, ys_1, phis_1, vs_green
                        planed_trj_red = xs_1, ys_1, phis_1, vs_red
                        planed_trj_g.append(planed_trj_green)
                        planed_trj_r.append(planed_trj_red)
                        self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))
            self.path_list = {'green': planed_trj_g, 'red': planed_trj_r}

        elif task == 'straight':
            lane_width_flag = [Para.LANE_WIDTH_3, Para.LANE_WIDTH_3, Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            end_offsets = [Para.OFFSET_U + Para.GREEN_BELT_LON + sum(lane_width_flag[:i]) + 0.5 * lane_width_flag[i]
                           for i in range(Para.LANE_NUMBER_LON_OUT)]
            start_offsets = [Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 * 0.5]
            rho = [3 / 10]                       # control the shape and number of bezier curve

            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    for k in rho:
                        control_point1 = start_offset, -Para.CROSSROAD_SIZE_LON / 2, 90
                        control_point4 = end_offset, Para.CROSSROAD_SIZE_LON / 2, 90
                        control_point2, control_point3 = get_bezier_middle_point(control_point1, control_point4, k)
                        self.control_points.append([control_point1, control_point2, control_point3, control_point4])
                        node = np.asfortranarray(
                            [[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                             [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]]
                            , dtype=np.float32)
                        curve = bezier.Curve(node, degree=3)
                        s_vals = np.linspace(0, 1.0, int(curve.length) * meter_pointnum_ratio)
                        trj_data = curve.evaluate_multi(s_vals)
                        trj_data = trj_data.astype(np.float32)
                        start_straight_line_x = start_offset * np.ones(shape=(sl * meter_pointnum_ratio,),
                                                                       dtype=np.float32)[:-1]
                        start_straight_line_y = np.linspace(-Para.CROSSROAD_SIZE_LON / 2 - sl, -Para.CROSSROAD_SIZE_LON / 2,
                                                            sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                        end_straight_line_x = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                        end_straight_line_y = np.linspace(Para.CROSSROAD_SIZE_LON / 2, Para.CROSSROAD_SIZE_LON / 2 + sl,
                                                          sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                        planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                     np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                        xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                        xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                        phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi
                        vs_green = np.array([8.33] * len(start_straight_line_x) + [7.0] * (len(trj_data[0]) - 1) + [8.33] *
                                            len(end_straight_line_x), dtype=np.float32)
                        vs_red_0 = np.array(
                            [8.33] * (len(start_straight_line_x) - meter_pointnum_ratio * (sl - dece_dist + int(Para.L))),
                            dtype=np.float32)
                        vs_red_1 = np.linspace(8.33, 0.0, meter_pointnum_ratio * dece_dist, endpoint=True, dtype=np.float32)
                        vs_red_2 = np.array(
                            [0.0] * (meter_pointnum_ratio * int(Para.L / 2) + len(trj_data[0]) - 1) + [0.0] * len(
                                end_straight_line_x), dtype=np.float32)
                        vs_red = np.append(np.append(vs_red_0, vs_red_1), vs_red_2)
                        planed_trj_green = xs_1, ys_1, phis_1, vs_green
                        planed_trj_red = xs_1, ys_1, phis_1, vs_red
                        planed_trj_g.append(planed_trj_green)
                        planed_trj_r.append(planed_trj_red)
                        self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))
            self.path_list = {'green': planed_trj_g, 'red': planed_trj_r}

        else:
            assert task == 'right'
            lane_width_flag = [Para.LANE_WIDTH_1, Para.LANE_WIDTH_1, Para.LANE_WIDTH_1,
                               Para.PERSON_LANE_WIDTH + Para.BIKE_LANE_WIDTH]
            end_offsets = [Para.OFFSET_R - sum(lane_width_flag[:i]) - 0.5 * lane_width_flag[i] for i in
                           range(Para.LANE_NUMBER_LAT_OUT)]
            start_offsets = [Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_3 + Para.LANE_WIDTH_3 * 0.5]
            rho = [3 / 10]                        # control the shape and number of bezier curve

            for start_offset in start_offsets:
                for end_offset in end_offsets:
                    for k in rho:
                        control_point1 = start_offset, -Para.CROSSROAD_SIZE_LON / 2, 90
                        control_point4 = Para.CROSSROAD_SIZE_LAT / 2, end_offset, 0
                        control_point2, control_point3 = get_bezier_middle_point(control_point1, control_point4, k)
                        self.control_points.append([control_point1, control_point2, control_point3, control_point4])
                        node = np.asfortranarray(
                            [[control_point1[0], control_point2[0], control_point3[0], control_point4[0]],
                             [control_point1[1], control_point2[1], control_point3[1], control_point4[1]]],
                            dtype=np.float32)
                        curve = bezier.Curve(node, degree=3)
                        s_vals = np.linspace(0, 1.0, int(curve.length) * meter_pointnum_ratio)
                        trj_data = curve.evaluate_multi(s_vals)
                        trj_data = trj_data.astype(np.float32)
                        start_straight_line_x = start_offset * np.ones(shape=(sl * meter_pointnum_ratio,),
                                                                       dtype=np.float32)[:-1]
                        start_straight_line_y = np.linspace(-Para.CROSSROAD_SIZE_LON / 2 - sl, -Para.CROSSROAD_SIZE_LON / 2,
                                                            sl * meter_pointnum_ratio, dtype=np.float32)[:-1]
                        end_straight_line_x = np.linspace(Para.CROSSROAD_SIZE_LAT / 2, Para.CROSSROAD_SIZE_LAT / 2 + sl,
                                                          sl * meter_pointnum_ratio, dtype=np.float32)[1:]
                        end_straight_line_y = end_offset * np.ones(shape=(sl * meter_pointnum_ratio,), dtype=np.float32)[1:]
                        planed_trj = np.append(np.append(start_straight_line_x, trj_data[0]), end_straight_line_x), \
                                     np.append(np.append(start_straight_line_y, trj_data[1]), end_straight_line_y)
                        xs_1, ys_1 = planed_trj[0][:-1], planed_trj[1][:-1]
                        xs_2, ys_2 = planed_trj[0][1:], planed_trj[1][1:]
                        phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi

                        vs_green = np.array([8.33] * len(start_straight_line_x) + [7.0] * (len(trj_data[0]) - 1) + [8.33] *
                                            len(end_straight_line_x), dtype=np.float32)

                        planed_trj_green = xs_1, ys_1, phis_1, vs_green
                        planed_trj_red = xs_1, ys_1, phis_1, vs_green  # the same velocity design for turning right
                        planed_trj_g.append(planed_trj_green)
                        planed_trj_r.append(planed_trj_red)
                        self.path_len_list.append((sl * meter_pointnum_ratio, len(trj_data[0]), len(xs_1)))
            self.path_list = {'green': planed_trj_g, 'red': planed_trj_r}

    def _find_closest_point(self, x, y, ratio=10):
        path_len = len(self.path[0])
        reduced_idx = np.arange(0, path_len, ratio)
        reduced_path_x, reduced_path_y = self.path[0][reduced_idx], self.path[1][reduced_idx]
        dists = np.square(x - reduced_path_x) + np.square(y - reduced_path_y)
        idx = np.argmin(dists) * ratio
        return idx, self.idx2point(idx)

    def plot_path(self, x, y):
        plt.axis('equal')
        color = ['blue', 'coral', 'darkcyan', 'pink']
        for i, path in enumerate(self.path_list['green']):
            plt.plot(path[0], path[1], color=color[i], alpha=1.0)

        for _, point in enumerate(self.control_points):
            for item in point:
                plt.scatter(item[0], item[1], color='red')
        print(self.path_len_list)

        index, closest_point = self._find_closest_point(np.array([x], np.float32),
                                                        np.array([y], np.float32))
        plt.plot(x, y, 'b*')
        plt.plot(closest_point[0], closest_point[1], 'ro')
        plt.show()


def test_ref_path():
    path = ReferencePath('right', 0)
    path.plot_path(1.875, 0)


def test_future_n_data():
    path = ReferencePath('straight', '0')
    plt.axis('equal')
    current_i = 600
    plt.plot(path.path[0], path.path[1])
    future_data_list = path.future_n_data(current_i, 5)
    plt.plot(path.indexs2points(current_i)[0], path.indexs2points(current_i)[1], 'go')
    for point in future_data_list:
        plt.plot(point[0], point[1], 'r*')
    plt.show()


def test_compute_next_track_info():
    model = EnvironmentModel()
    next_ego_infos = np.array(
        [[10., 0., 0, Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON / 2 - 10, 90]])
    ref_points = np.array([[Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON / 2 - 10, 90, 10]])
    next_obses_track = model._compute_next_track_info_vectorized(next_ego_infos, ref_points)
    print(next_obses_track.numpy())

    next_ego_infos = np.array(
        [[10., 0., 0, Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON / 2 - 10, 90]])
    ref_points = np.array([[Para.OFFSET_D + Para.LANE_WIDTH_2 / 2 + 5, -Para.CROSSROAD_SIZE_LON / 2 - 10, 90, 10]])
    next_obses_track = model._compute_next_track_info_vectorized(next_ego_infos, ref_points)
    print(next_obses_track.numpy())

    next_ego_infos = np.array(
        [[10., 0., 0, Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON / 2 - 10, 90]])
    ref_points = np.array([[Para.OFFSET_D + Para.LANE_WIDTH_2 / 2 - 5, -Para.CROSSROAD_SIZE_LON / 2 - 10, 90, 10]])
    next_obses_track = model._compute_next_track_info_vectorized(next_ego_infos, ref_points)
    print(next_obses_track.numpy())

    next_ego_infos = np.array(
        [[10., 0., 0, Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON / 2 - 8, 90]])
    ref_points = np.array([[Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON / 2 - 10, 90, 10]])
    next_obses_track = model._compute_next_track_info_vectorized(next_ego_infos, ref_points)
    print(next_obses_track.numpy())

    next_ego_infos = np.array(
        [[10., 0., 0, Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON / 2 - 15, 90]])
    ref_points = np.array([[Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON / 2 - 10, 90, 10]])
    next_obses_track = model._compute_next_track_info_vectorized(next_ego_infos, ref_points)
    print(next_obses_track.numpy())


def test_tracking_error_vector():
    # path = ReferencePath('straight', green_or_red='green')
    # x, y, phi, v = Para.OFFSET_D + Para.LANE_WIDTH_2 / 2, -Para.CROSSROAD_SIZE_LON/2 - 10, 90, 10
    # tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    # print(tracking_error_vector, [-0.026, -3.625,  0, 1.67])
    #
    # x, y, phi, v = Para.OFFSET_D + Para.LANE_WIDTH_2 * 3, -Para.CROSSROAD_SIZE_LON/2 - 10, 90, 10
    # tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    # print(tracking_error_vector, [-0.026, 4.5,  0, 1.67])
    #
    # x, y, phi, v = Para.OFFSET_D + Para.LANE_WIDTH_2 + Para.LANE_WIDTH_1 / 2, -Para.CROSSROAD_SIZE_LON/2 - 10, 90, 10
    # tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    # print(tracking_error_vector, [-0.026, -0.125,  0, 1.67])

    path = ReferencePath('left', green_or_red='green')
    x, y, phi, v = -Para.CROSSROAD_SIZE_LAT / 2 - 10, Para.OFFSET_L + Para.GREEN_BELT_LAT + Para.LANE_WIDTH_1 / 2, 180, 10
    tracking_error_vector_vec = path.tracking_error_vector_vectorized(x, y, phi, v)
    tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    print(tracking_error_vector_vec, tracking_error_vector, [-0.075, -7.5, 0, 1.67])
    print(np.sum(tracking_error_vector_vec - tracking_error_vector) < 1e-8)

    x, y, phi, v = -Para.CROSSROAD_SIZE_LAT / 2 - 10, Para.OFFSET_L + Para.GREEN_BELT_LAT + Para.LANE_WIDTH_1 * 3, 180, 10
    tracking_error_vector_vec = path.tracking_error_vector_vectorized(x, y, phi, v)
    tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    print(tracking_error_vector_vec, tracking_error_vector, [-0.075, 9.375, 0, 1.67])
    print(np.sum(tracking_error_vector_vec - tracking_error_vector) < 1e-8)

    path = ReferencePath('right', green_or_red='green')
    x, y, phi, v = Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_R - Para.LANE_WIDTH_1 / 2, 180, 10
    tracking_error_vector_vec = path.tracking_error_vector_vectorized(x, y, phi, v)
    tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    print(tracking_error_vector_vec, tracking_error_vector, [-0.0, -7.5, 0, 1.67])
    print(np.sum(tracking_error_vector_vec - tracking_error_vector) < 1e-8)

    x, y, phi, v = Para.CROSSROAD_SIZE_LAT / 2, Para.OFFSET_R - Para.LANE_WIDTH_1 * 3, 0, 10
    tracking_error_vector_vec = path.tracking_error_vector_vectorized(x, y, phi, v)
    tracking_error_vector = path.tracking_error_vector(x, y, phi, v)
    print(tracking_error_vector_vec, tracking_error_vector, [-0.075, 9.375, 0, 1.67])
    print(np.sum(tracking_error_vector_vec - tracking_error_vector) < 1e-8)


def test_model():
    from endtoend import CrossroadEnd2endMix
    env = CrossroadEnd2endMix()
    model = EnvironmentModel()
    while 1:
        obs, info = env.reset()
        for i in range(35):
            obs_list, future_point_list = [], []
            obs_list.append(obs)
            future_point_list.append(info['future_n_point'])
            action = np.array([0, -1], dtype=np.float32)
            # obs, reward, done, info = env.step(action)
            env.render()
            obses = np.stack(obs_list, 0)
            future_points = np.array(future_point_list)
            model.reset(obses)
            print(obses.shape, future_points.shape)
            for rollout_step in range(10):
                actions = tf.tile(tf.constant([[0.5, 0]], dtype=tf.float32), tf.constant([len(obses), 1]))
                obses, rewards, punish_term_for_training, real_punish_term, veh2veh4real, veh2road4real, \
                veh2bike4real, veh2person4real = model.rollout_out(actions, future_points[:, :, i])
                model.render()


def test_ref():
    import numpy as np
    import matplotlib.pyplot as plt
    ref = ReferencePath('left', '0')
    path1, path2, path3 = ref.path_list[LIGHT_PHASE_TO_GREEN_OR_RED[0]]
    path1, path2, path3 = [ite[1200:-1200] for ite in path1], \
                          [ite[1200:-1200] for ite in path2], \
                          [ite[1200:-1200] for ite in path3]
    x1, y1, phi1, v1 = path1
    x2, y2, phi2, v1 = path2
    x3, y3, phi3, v1 = path3

    plt.plot(y1, x1, 'r')
    plt.plot(y2, x2, 'g')
    plt.plot(y3, x3, 'b')
    z1 = np.polyfit(y1, x1, 3, rcond=None, full=False, w=None, cov=False)
    print(type(list(z1)))
    p1_fit = np.poly1d(z1)
    print(z1, p1_fit)
    plt.plot(y1, p1_fit(y1), 'r*')
    plt.show()


if __name__ == '__main__':
    # test_model()
    test_tracking_error_vector()
    # test_compute_next_track_info()
