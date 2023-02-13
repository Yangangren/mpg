#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: evaluator.py
# =====================================

import copy
import logging
import os

import gym
import numpy as np

from preprocessor import Preprocessor
from utils.dummy_vec_env import DummyVecEnv
from utils.misc import TimerStat, args2envkwargs
from env_build.endtoend import CrossroadEnd2endMix

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Evaluator(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, policy_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        kwargs = copy.deepcopy(vars(self.args))
        if self.args.env_id == 'PathTracking-v0':
            self.env = gym.make(self.args.env_id, num_agent=self.args.num_eval_agent, num_future_data=self.args.num_future_data)
        elif self.args.env_id == 'CrossroadEnd2endMix-v0':
            self.env = CrossroadEnd2endMix(**args2envkwargs(args))
        else:
            env = gym.make(self.args.env_id)
            self.env = DummyVecEnv(env)
        self.policy_with_value = policy_cls(**kwargs)
        self.iteration = 0
        if self.args.mode == 'training':
            self.log_dir = self.args.log_dir + '/evaluator'
        else:
            self.log_dir = self.args.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor(**kwargs)

        self.writer = self.tf.summary.create_file_writer(self.log_dir)
        self.stats = {}
        self.eval_timer = TimerStat()
        self.eval_times = 0

    def get_stats(self):
        self.stats.update(dict(eval_time=self.eval_timer.mean))
        return self.stats

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def evaluate_saved_model(self, model_load_dir, ppc_params_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)
        # self.load_ppc_params(ppc_params_load_dir)

    def run_an_episode(self, steps=None, render=True):
        track_reward_list = []
        total_reward_list = []
        reward_info_dict_list = []
        done = 0
        obs, info = self.env.reset()
        if render: self.env.render(weights=np.zeros((14,)))
        if steps is not None:
            for _ in range(steps):
                processed_obs = self.preprocessor.tf_process_obses(obs)
                action = self.policy_with_value.compute_mode(processed_obs[np.newaxis, :])
                obs, reward, done, info = self.env.step(action.numpy()[0])
                total_reward = reward - self.args.init_punish_factor * info['reward_info']['real_punish_term']
                reward_info_dict_list.append(info['reward_info'])
                if render: self.env.render(weights=attn_weights)
                track_reward_list.append(reward)
                total_reward_list.append(total_reward)
        else:
            while not done:
                processed_obs = self.preprocessor.tf_process_obses(obs)
                action = self.policy_with_value.compute_mode(processed_obs[np.newaxis, :])
                obs, reward, done, info = self.env.step(action.numpy()[0])
                total_reward = reward - self.args.init_punish_factor * info['reward_info']['real_punish_term']
                if render: self.env.render(weights=attn_weights)
                track_reward_list.append(reward)
                total_reward_list.append(total_reward)
        episode_track_return = sum(track_reward_list)
        episode_total_return = sum(total_reward_list)
        episode_len = len(track_reward_list)
        info_dict = dict()
        for key in reward_info_dict_list[0].keys():
            info_key = list(map(lambda x: x[key], reward_info_dict_list))
            mean_key = sum(info_key) / len(info_key)
            info_dict.update({key: mean_key})
        info_dict.update(dict(episode_track_return=episode_track_return,
                              episode_total_return=episode_total_return,
                              episode_len=episode_len))
        return info_dict

    def run_n_episode(self, n):
        list_of_info_dict = []
        for _ in range(n):
            logger.info('logging {}-th episode'.format(_))
            info_dict = self.run_an_episode(self.args.fixed_steps, self.args.eval_render)
            list_of_info_dict.append(info_dict.copy())
        n_info_dict = dict()
        for key in list_of_info_dict[0].keys():
            info_key = list(map(lambda x: x[key], list_of_info_dict))
            mean_key = sum(info_key) / len(info_key)
            n_info_dict.update({key: mean_key})
        return n_info_dict

    def run_n_episodes_parallel(self, n):
        logger.info('logging {} episodes in parallel'.format(n))
        metrics_list = []
        obses_list = []
        actions_list = []
        rewards_list = []
        obses = self.env.reset()
        if self.args.eval_render: self.env.render()
        for _ in range(self.args.fixed_steps):
            processed_obses = self.preprocessor.tf_process_obses(obses)
            actions = self.policy_with_value.compute_mode(processed_obses)
            obses_list.append(obses)
            actions_list.append(actions)
            obses, rewards, dones, _ = self.env.step(actions.numpy())
            if self.args.eval_render: self.env.render()
            rewards_list.append(rewards)
        for i in range(n):
            obs_list = [obses[i] for obses in obses_list]
            action_list = [actions[i] for actions in actions_list]
            reward_list = [rewards[i] for rewards in rewards_list]
            episode_return = sum(reward_list)
            episode_len = len(reward_list)
            info_dict = dict()
            info_dict.update(dict(obs_list=np.array(obs_list),
                                  action_list=np.array(action_list),
                                  reward_list=np.array(reward_list),
                                  episode_return=episode_return,
                                  episode_len=episode_len))
            metrics_list.append(self.metrics_for_an_episode(info_dict))
        out = {}
        for key in metrics_list[0].keys():
            value_list = list(map(lambda x: x[key], metrics_list))
            out.update({key: sum(value_list) / len(value_list)})
        return metrics_list, out


    def metrics_for_an_episode(self, episode_info):  # user defined, transform episode info dict to metric dict
        key_list = ['episode_return', 'episode_len']
        episode_return = episode_info['episode_return']
        episode_len = episode_info['episode_len']
        value_list = [episode_return, episode_len]
        if self.args.env_id == 'PathTracking-v0':
            delta_v_list = list(map(lambda x: x[0], episode_info['obs_list']))
            delta_y_list = list(map(lambda x: x[3], episode_info['obs_list']))
            delta_phi_list = list(map(lambda x: x[4], episode_info['obs_list']))
            steer_list = list(map(lambda x: x[0]*1.2 * np.pi / 9, episode_info['action_list']))
            acc_list = list(map(lambda x: x[1]*3., episode_info['action_list']))

            rew_list = episode_info['reward_list']
            stationary_rew_mean = sum(rew_list[20:])/len(rew_list[20:])

            delta_y_mse = np.sqrt(np.mean(np.square(np.array(delta_y_list))))
            delta_phi_mse = np.sqrt(np.mean(np.square(np.array(delta_phi_list))))
            delta_v_mse = np.sqrt(np.mean(np.square(np.array(delta_v_list))))
            steer_mse = np.sqrt(np.mean(np.square(np.array(steer_list))))
            acc_mse = np.sqrt(np.mean(np.square(np.array(acc_list))))
            key_list.extend(['delta_y_mse', 'delta_phi_mse', 'delta_v_mse',
                             'stationary_rew_mean', 'steer_mse', 'acc_mse'])
            value_list.extend([delta_y_mse, delta_phi_mse, delta_v_mse,
                               stationary_rew_mean, steer_mse, acc_mse])
        elif self.args.env_id == 'InvertedPendulumConti-v0':
            x_list = list(map(lambda x: x[0], episode_info['obs_list']))
            theta_list = list(map(lambda x: x[1], episode_info['obs_list']))
            xdot_list = list(map(lambda x: x[2], episode_info['obs_list']))
            thetadot_list = list(map(lambda x: x[3], episode_info['obs_list']))
            x_mean, x_var = np.mean(np.array(x_list)), np.var(np.array(x_list))
            theta_mean, theta_var = np.mean(np.array(theta_list)), np.var(np.array(theta_list))
            xdot_mean, xdot_var = np.mean(np.array(xdot_list)), np.var(np.array(xdot_list))
            thetadot_mean, thetadot_var = np.mean(np.array(thetadot_list)), np.var(np.array(thetadot_list))
            x_mse, theta_mse = np.sqrt(np.mean(np.square(np.array(x_list)))),\
                               np.sqrt(np.mean(np.square(np.array(theta_list))))

            xdot_mse, thetadot_mse = np.sqrt(np.mean(np.square(np.array(xdot_list)))),\
                                     np.sqrt(np.mean(np.square(np.array(thetadot_list))))
            x_mse_25, theta_mse_25 = np.sqrt(np.mean(np.square(np.array(x_list)[:25]))), \
                                     np.sqrt(np.mean(np.square(np.array(theta_list)[:25])))
            xdot_mse_25, thetadot_mse_25 = np.sqrt(np.mean(np.square(np.array(xdot_list)[:25]))), \
                                     np.sqrt(np.mean(np.square(np.array(thetadot_list)[:25])))
            key_list.extend(['x_mean', 'x_var', 'theta_mean', 'theta_var',
                             'xdot_mean', 'xdot_var', 'thetadot_mean', 'thetadot_var',
                             'x_mse', 'theta_mse', 'xdot_mse', 'thetadot_mse',
                             'x_mse_25', 'theta_mse_25', 'xdot_mse_25', 'thetadot_mse_25'])
            value_list.extend([x_mean, x_var, theta_mean, theta_var,
                               xdot_mean, xdot_var, thetadot_mean, thetadot_var,
                               x_mse, theta_mse, xdot_mse, thetadot_mse,
                               x_mse_25, theta_mse_25, xdot_mse_25, thetadot_mse_25])

        return dict(zip(key_list, value_list))

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def run_evaluation(self, iteration):
        with self.eval_timer:
            self.iteration = iteration
            n_info_dict = self.run_n_episode(self.args.num_eval_episode)
            with self.writer.as_default():
                for key, val in n_info_dict.items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                for key, val in self.get_stats().items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                self.writer.flush()
        if self.eval_times % self.args.eval_log_interval == 0:
            logger.info('Evaluator_info: {}, {}'.format(self.get_stats(),n_info_dict))
        self.eval_times += 1

    def get_eval_times(self):
        return self.eval_times

def test_trained_model(model_dir, ppc_params_dir, iteration):
    from train_script import built_mixedpg_parser
    from policy import PolicyWithQs
    args = built_mixedpg_parser()
    evaluator = Evaluator(PolicyWithQs, args.env_id, args)
    evaluator.load_weights(model_dir, iteration)
    evaluator.load_ppc_params(ppc_params_dir)
    return evaluator.metrics(1000, render=False, reset=False)

def test_evaluator():
    from train_script import built_SAC_parser
    from policy import PolicyWithQs
    args = built_SAC_parser()
    evaluator = Evaluator(PolicyWithQs, args.env_id, args)
    evaluator.run_evaluation(3)

if __name__ == '__main__':
    test_evaluator()
