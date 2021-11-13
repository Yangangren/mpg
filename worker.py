#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: worker.py
# =====================================

import logging

import gym
import numpy as np
import tensorflow as tf

from preprocessor import Preprocessor
from utils.misc import judge_is_nan, args2envkwargs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# logger.setLevel(logging.INFO)


class OffPolicyWorker(object):
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    """just for sample"""

    def __init__(self, policy_cls, env_id, args, worker_id):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.worker_id = worker_id
        self.args = args
        self.env = gym.make(env_id, **args2envkwargs(args))
        self.policy_with_value = policy_cls(self.args)
        self.batch_size = self.args.batch_size
        self.obs, self.info = self.env.reset()
        self.done = False
        self.preprocessor = Preprocessor((self.args.obs_dim, ), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                          self.args.reward_scale, self.args.reward_shift, args=self.args, gamma=self.args.gamma)
        # obses_rela = self.preprocessor.tf_convert_ego_coordinate(np.tile(self.obs[np.newaxis, :], (5, 1)))
        self.explore_sigma = self.args.explore_sigma
        self.iteration = 0
        self.num_sample = 0
        self.sample_times = 0
        self.stats = {}
        logger.info('Worker initialized')

    def get_stats(self):
        self.stats.update(dict(worker_id=self.worker_id,
                               num_sample=self.num_sample,
                               # ppc_params=self.get_ppc_params()
                               )
                          )
        return self.stats

    def save_weights(self, save_dir, iteration):
        self.policy_with_value.save_weights(save_dir, iteration)

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def apply_gradients(self, iteration, grads):
        self.iteration = iteration
        self.policy_with_value.apply_gradients(tf.constant(iteration, dtype=tf.int32), grads)

    def _get_state(self, obs, mask):
        obs_other, _ = self.policy_with_value.compute_attn(obs[self.args.other_start_dim:][np.newaxis, :],
                                                           mask[np.newaxis, :])
        obs_other = obs_other.numpy()[0]
        state = np.concatenate((obs[:self.args.other_start_dim], obs_other), axis=0)
        return state

    def get_ppc_params(self):
        return self.preprocessor.get_params()

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def save_ppc_params(self, save_dir):
        self.preprocessor.save_params(save_dir)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def sample(self):
        batch_data = []
        for _ in range(self.batch_size):
            obs_transformed = self.preprocessor.convert_ego_coordinate(self.obs[np.newaxis, :])
            processed_obs = self.preprocessor.process_obs(obs_transformed)
            mask = self.info['mask']
            state = self._get_state(processed_obs, mask)
            judge_is_nan([processed_obs])
            action, logp = self.policy_with_value.compute_action(state[np.newaxis, :])
            if self.explore_sigma is not None:
                action += np.random.normal(0, self.explore_sigma, np.shape(action))
            try:
                judge_is_nan([action])
            except ValueError:
                print('processed_obs', processed_obs)
                print('preprocessor_params', self.preprocessor.get_params())
                print('policy_weights', self.policy_with_value.policy.trainable_weights)
                action, logp = self.policy_with_value.compute_action(processed_obs[np.newaxis, :])
                judge_is_nan([action])
                raise ValueError
            obs_tp1, reward, self.done, info = self.env.step(action.numpy()[0])
            batch_data.append((obs_tp1, self.done, info['path_index'], info['mask']))
            if self.done:
                self.obs, self.info = self.env.reset()
            else:
                self.obs = obs_tp1.copy()
                self.info = info.copy()
            # self.env.render()

        if self.worker_id == 1 and self.sample_times % self.args.worker_log_interval == 0:
            logger.info('Worker_info: {}'.format(self.get_stats()))

        self.num_sample += len(batch_data)
        self.sample_times += 1
        return batch_data

    def sample_with_count(self):
        batch_data = self.sample()
        return batch_data, len(batch_data)
