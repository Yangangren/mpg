#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/30
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: load_policy.py
# =====================================
import argparse
import json

import tensorflow as tf
import numpy as np

from env_build.utils.policy import AttentionPolicy4Toyota
from env_build.utils.preprocessor import Preprocessor


class LoadPolicy(object):
    def __init__(self, exp_dir, iter):
        model_dir = exp_dir + '/models'
        parser = argparse.ArgumentParser()
        params = json.loads(open(exp_dir + '/config.json').read())
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        self.args = parser.parse_args()
        self.policy = AttentionPolicy4Toyota(self.args)
        self.policy.load_weights(model_dir, iter)
        self.preprocessor = Preprocessor(self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         args=self.args, gamma=self.args.gamma)

    @tf.function
    def run_batch(self, obses, masks):
        processed_obses = self.preprocessor.process_obs(obses)
        states, weights = self._get_states(processed_obses, masks)
        actions = self.policy.compute_mode(states)
        if self.args.noise_mode == 'adv_noise':
            adv_actions, _ = self.policy.compute_adv_action(states, tf.cast(masks, dtype=tf.bool))
        elif self.args.noise_mode == 'no_noise':
            adv_actions = tf.zeros(shape=(actions.shape[0], self.args.adv_act_dim * self.args.other_number))
        else:
            import tensorflow_probability as tfp
            mean = tf.zeros(shape=(actions.shape[0], self.args.adv_act_dim * self.args.other_number))
            std = tf.ones(shape=(actions.shape[0], self.args.adv_act_dim * self.args.other_number))
            act_dist = tfp.distributions.MultivariateNormalDiag(mean, std)
            act_dist = tfp.distributions.TransformedDistribution(distribution=act_dist, bijector=tfb.Tanh())
            adv_actions = act_dist.sample()
        return actions, weights, adv_actions

    @tf.function
    def obj_value_batch(self, obses, masks):
        processed_obses = self.preprocessor.process_obs(obses)
        states, _ = self._get_states(processed_obses, masks)
        values = self.policy.compute_obj_v(states)
        return values

    def _get_states(self, mb_obs, mb_mask):
        mb_obs_others, mb_attn_weights = self.policy.compute_attn(mb_obs[:, self.args.other_start_dim:], mb_mask)
        mb_state = tf.concat((mb_obs[:, :self.args.other_start_dim], mb_obs_others), axis=1)
        return mb_state, mb_attn_weights

