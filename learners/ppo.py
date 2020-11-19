#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/9
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ppo.py
# =====================================

import numpy as np
import gym
from utils.misc import TimerStat, safemean, judge_is_nan
import logging
from collections import deque
from preprocessor import Preprocessor
import tensorflow as tf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PPOLearner(tf.Module):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, policy_cls, args):
        super().__init__()
        self.args = args
        env = gym.make(self.args.env_id)
        obs_space, act_space = env.observation_space, env.action_space
        env.close()
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.batch_data = None
        self.policy_gradient_timer = TimerStat()
        self.v_gradient_timer = TimerStat()
        self.mb_learning_timer = TimerStat()
        self.stats = {}
        self.epinfobuf = deque(maxlen=100)
        self.permutation = None

    def get_stats(self):
        return self.stats

    def post_processing(self, batch_data):
        tmp = {'batch_obs': np.asarray(list(map(lambda x: x[0], batch_data)), dtype=np.float32),
               'batch_actions': np.asarray(list(map(lambda x: x[1], batch_data)), dtype=np.float32),
               'batch_rewards': np.asarray(list(map(lambda x: x[2], batch_data)), dtype=np.float32),
               'batch_obs_tp1': np.asarray(list(map(lambda x: x[3], batch_data)), dtype=np.float32),
               'batch_dones': np.asarray(list(map(lambda x: x[4], batch_data)), dtype=np.float32),
               'batch_logps': np.asarray(list(map(lambda x: x[5], batch_data)), dtype=np.float32),
               }

        return tmp

    def get_batch_data(self, batch_data, epinfos):
        self.batch_data = self.post_processing(batch_data)
        batch_advs, batch_tdlambda_returns, batch_values = self.compute_advantage()
        self.batch_data.update(dict(batch_advs=batch_advs,
                                    batch_tdlambda_returns=batch_tdlambda_returns,
                                    batch_values=batch_values))
        self.epinfobuf.extend(epinfos)
        self.stats.update(eprewmean=safemean([epinfo['r'] for epinfo in self.epinfobuf]),
                          eplenmean=safemean([epinfo['l'] for epinfo in self.epinfobuf]))

    def batch_data_count(self):
        return len(self.batch_data['batch_obs'])

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def compute_advantage(self):  # require data is in order
        n_steps = len(self.batch_data['batch_rewards'])
        batch_obs = self.batch_data['batch_obs']
        batch_rewards = self.batch_data['batch_rewards']
        batch_values = self.policy_with_value.compute_vf(batch_obs).numpy()[:, 0]  # len = n_steps + 1
        batch_advs = np.zeros_like(self.batch_data['batch_rewards'])
        lastgaelam = 0
        for t in reversed(range(n_steps - 1)):
            nextnonterminal = 1 - self.batch_data['batch_dones'][t + 1]
            delta = batch_rewards[t] + self.args.gamma * batch_values[t + 1] * nextnonterminal - batch_values[t]
            batch_advs[t] = lastgaelam = delta + self.args.lam * self.args.gamma * nextnonterminal * lastgaelam
        batch_tdlambda_returns = batch_advs + batch_values

        return batch_advs, batch_tdlambda_returns, batch_values

    @tf.function
    def value_forward_and_backward(self, mb_obses, target, mb_oldvs):
        with self.tf.GradientTape() as tape:
            v_pred = self.policy_with_value.compute_vf(mb_obses)
            # vpredclipped = mb_oldvs + self.tf.clip_by_value(v_pred - mb_oldvs,
            #                                                 -self.args.ppo_loss_clip,
            #                                                 self.args.ppo_loss_clip)
            v_loss1 = self.tf.square(v_pred - target)
            # v_loss2 = self.tf.square(vpredclipped - target)
            v_loss = .5 * self.tf.reduce_mean(v_loss1) # * self.tf.reduce_mean(self.tf.maximum(v_loss1, v_loss2))
            value_mean = self.tf.reduce_mean(v_pred)
        value_gradient = tape.gradient(v_loss, self.policy_with_value.value.trainable_weights)
        return v_loss, value_gradient, value_mean

    @tf.function
    def policy_forward_and_backward(self, mb_obses, mb_actions, mb_neglogps, mb_advs):
        mb_advs = (mb_advs - self.tf.reduce_mean(mb_advs)) / (self.tf.keras.backend.std(mb_advs) + 1e-8)
        with self.tf.GradientTape() as tape:
            policy_entropy = self.policy_with_value.compute_entropy(mb_obses)
            current_neglogp = -self.policy_with_value.compute_logps(mb_obses, mb_actions)
            ratio = self.tf.exp(mb_neglogps - current_neglogp)
            pg_loss1 = -ratio * mb_advs
            pg_loss2 = -mb_advs * self.tf.clip_by_value(ratio, 1 - self.args.ppo_loss_clip, 1 + self.args.ppo_loss_clip)
            clipped_loss = self.tf.reduce_mean(self.tf.maximum(pg_loss1, pg_loss2))
            pg_loss = clipped_loss - self.args.ent_coef * policy_entropy
            clipfrac = self.tf.reduce_mean(self.tf.cast(
                self.tf.greater(self.tf.abs(ratio - 1.0), self.args.ppo_loss_clip), self.tf.float32))

        policy_gradient = tape.gradient(pg_loss, self.policy_with_value.policy.trainable_weights)
        return pg_loss, policy_gradient, clipped_loss, policy_entropy, clipfrac

    def compute_gradient_over_ith_minibatch(self, i):  # compute gradient of the i-th mini-batch
        if i == 0:
            self.permutation = np.arange(self.batch_data_count())
            np.random.shuffle(self.permutation)
        with self.mb_learning_timer:
            start_idx, end_idx = i * self.args.mini_batch_size, (i + 1) * self.args.mini_batch_size
            mbinds = self.permutation[start_idx:end_idx]
            mb_obs = self.tf.constant(self.batch_data['batch_obs'][mbinds])
            mb_advs = self.tf.constant(self.batch_data['batch_advs'][mbinds])
            mb_tdlambda_returns = self.tf.constant(self.batch_data['batch_tdlambda_returns'][mbinds])
            mb_actions = self.tf.constant(self.batch_data['batch_actions'][mbinds])
            mb_neglogps = self.tf.constant(-self.batch_data['batch_logps'][mbinds])
            mb_oldvs = self.tf.constant(self.batch_data['batch_values'][mbinds])

            with self.v_gradient_timer:
                v_loss, value_gradient, value_mean = self.value_forward_and_backward(mb_obs, mb_tdlambda_returns, mb_oldvs)
                # judge_is_nan([v_loss])
                # judge_is_nan(value_gradient)
                # judge_is_nan([value_mean])

            with self.policy_gradient_timer:
                pg_loss, policy_gradient, clipped_loss, policy_entropy, clipfrac = \
                    self.policy_forward_and_backward(mb_obs, mb_actions, mb_neglogps, mb_advs)
                # judge_is_nan([pg_loss])
                # judge_is_nan(policy_gradient)
                # judge_is_nan([policy_entropy])
            # pg_loss, all_grad, clipped_loss, policy_entropy, clipfrac, v_loss, value_mean \
            #     = self.get_grad(mb_obs, mb_actions, mb_neglogps, mb_advs, mb_tdlambda_returns, mb_oldvs)
            value_gradient, value_gradient_norm = self.tf.clip_by_global_norm(value_gradient,
                                                                              self.args.gradient_clip_norm)
            policy_gradient, policy_gradient_norm = self.tf.clip_by_global_norm(policy_gradient,
                                                                                self.args.gradient_clip_norm)
        self.stats.update(dict(
            v_timer=self.v_gradient_timer.mean,
            pg_time=self.policy_gradient_timer.mean,
            mb_learning_time=self.mb_learning_timer.mean,
            v_loss=v_loss.numpy(),
            policy_loss=pg_loss.numpy(),
            clipped_loss=clipped_loss.numpy(),
            policy_entropy=policy_entropy.numpy(),
            value_mean=value_mean.numpy(),
            target_mean=np.mean(mb_tdlambda_returns),
            value_gradient_norm=value_gradient_norm.numpy(),
            policy_gradient_norm=policy_gradient_norm.numpy(),
            clipfrac=clipfrac.numpy()
        ))

        gradient_tensor = value_gradient + policy_gradient
        return gradient_tensor
