#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: worker.py
# =====================================

import logging
import os

import gym
import numpy as np
from utils.monitor import Monitor
from preprocessor import Preprocessor
from utils.misc import judge_is_nan, TimerStat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# logger.setLevel(logging.INFO)


class OnPolicyWorker(object):
    """
    Act as both actor and learner
    """
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, policy_cls, learner_cls, env_id, args, worker_id):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.worker_id = worker_id
        self.args = args
        env = gym.make(env_id)
        self.env = Monitor(env)
        obs_space, act_space = self.env.observation_space, self.env.action_space
        self.learner = learner_cls(policy_cls, self.args)
        self.policy_with_value = policy_cls(obs_space, act_space, self.args)
        self.sample_batch_size = self.args.sample_batch_size
        self.obs = self.env.reset()
        self.done = False
        self.preprocessor = Preprocessor(obs_space, self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)
        self.log_dir = self.args.log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.stats = {}
        self.sampling_timer = TimerStat()
        self.processing_timer = TimerStat()
        logger.info('Worker initialized')

    def get_stats(self):
        return self.stats

    def shuffle(self):
        self.learner.shuffle()

    def save_weights(self, save_dir, iteration):
        self.policy_with_value.save_weights(save_dir, iteration)

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def apply_gradients(self, iteration, grads):
        iteration = self.tf.convert_to_tensor(iteration)
        self.policy_with_value.apply_gradients(iteration, grads)

    def get_ppc_params(self):
        return self.preprocessor.get_params()

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def save_ppc_params(self, save_dir):
        self.preprocessor.save_params(save_dir)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def sample_and_process(self):
        with self.sampling_timer:
            batch_data = []
            epinfos = []
            for _ in range(self.sample_batch_size):
                # judge_is_nan(self.obs)
                processed_obs = self.preprocessor.process_obs(self.obs)
                processed_obs_tensor = self.tf.convert_to_tensor(processed_obs[np.newaxis, :])
                action, logp = self.policy_with_value.compute_action(processed_obs_tensor)
                # judge_is_nan(action)
                # judge_is_nan(logp)
                action, logp = action.numpy()[0], logp.numpy()[0]
                obs_tp1, reward, self.done, info = self.env.step(action)
                processed_rew = self.preprocessor.process_rew(reward, self.done)

                batch_data.append((processed_obs.copy(), action, processed_rew, obs_tp1, self.done, logp))
                self.obs = self.env.reset() if self.done else obs_tp1.copy()
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)
        with self.processing_timer:
            self.learner.get_batch_data(batch_data, epinfos)
        self.stats.update(dict(worker_sampling_time=self.sampling_timer.mean,
                               worker_processing_time=self.processing_timer.mean))
        if self.args.reward_preprocess_type == 'normalize':
            self.stats.update(dict(ret_rms_var=self.preprocessor.ret_rms.var,
                                   ret_rms_mean=self.preprocessor.ret_rms.mean))

    def compute_gradient_over_ith_minibatch(self, i):
        self.learner.set_weights(self.get_weights())
        grad = self.learner.compute_gradient_over_ith_minibatch(i)
        learner_stats = self.learner.get_stats()
        self.stats.update(learner_stats)
        return grad


def debug_worker():
    from train_script import built_PPO_parser
    from policy import PolicyWithValue
    from learners.ppo import PPOLearner
    env_id = 'Pendulum-v0'
    worker_id = 0
    args = built_PPO_parser()
    worker = OnPolicyWorker(PolicyWithValue, PPOLearner, env_id, args, worker_id)
    for _ in range(10):
        worker.sample_and_process()


if __name__ == '__main__':
    debug_worker()

