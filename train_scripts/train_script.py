#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/8/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: train_script.py
# =====================================

import argparse
import datetime
import json
import logging
import os

import gym
import ray

from buffer import PrioritizedReplayBuffer, ReplayBuffer
from evaluator import Evaluator
from learners.ampc import AMPCLearner
from learners.mpg_learner import MPGLearner
from learners.nadp import NADPLearner
from learners.ndpg import NDPGLearner
from learners.sac import SACLearner
from learners.td3 import TD3Learner
from optimizer import OffPolicyAsyncOptimizer, SingleProcessOffPolicyOptimizer
from policy import PolicyWithQs
from tester import Tester
from trainer import Trainer
from worker import OffPolicyWorker
from envs_and_models.path_tracking_env import PathTrackingEnv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
NAME2WORKERCLS = dict([('OffPolicyWorker', OffPolicyWorker)])
NAME2LEARNERCLS = dict([('MPG', MPGLearner),
                        ('AMPC', AMPCLearner),
                        ('NADP', NADPLearner),
                        ('NDPG', NDPGLearner),
                        ('TD3', TD3Learner),
                        ('SAC', SACLearner)
                        ])
NAME2BUFFERCLS = dict([('normal', ReplayBuffer), ('priority', PrioritizedReplayBuffer), ('None', None)])
NAME2OPTIMIZERCLS = dict([('OffPolicyAsync', OffPolicyAsyncOptimizer),
                          ('SingleProcessOffPolicy', SingleProcessOffPolicyOptimizer)])
NAME2POLICYCLS = dict([('PolicyWithQs', PolicyWithQs)])
NAME2EVALUATORCLS = dict([('Evaluator', Evaluator), ('None', None)])
NUM_WORKER = 2
NUM_LEARNER = 12
NUM_BUFFER = 2


def set_seed(seed):
    import random
    import tensorflow as tf
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print('Using seed %s' % (str(seed)))

def built_NADP_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_mode', type=str, default='adv_noise')  # adv_noise rand_noise no_noise adv_noise_smooth
    parser.add_argument('--mode', type=str, default='testing') # training testing
    mode, noise_mode = parser.parse_args().mode, parser.parse_args().noise_mode
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rho', type=float, default=20)

    if mode == 'testing':
        test_dir = '../results/NADP/{}/experiment-2020-09-23-20-52-24'.format(noise_mode)
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        disturb_env = [-0.174 + i * 0.0348 for i in range(11)]
        params.update(dict(test_dir=test_dir,
                           disturb_env=disturb_env,
                           test_iter_list=[99000] * len(disturb_env),
                           test_log_dir=test_log_dir,
                           num_eval_episode=5,
                           num_eval_agent=5,
                           eval_log_interval=1,
                           fixed_steps=200,
                           eval_render=True))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--policy_type', type=str, default='PolicyWithQs')
    parser.add_argument('--worker_type', type=str, default='OffPolicyWorker')
    parser.add_argument('--evaluator_type', type=str, default='Evaluator')
    parser.add_argument('--buffer_type', type=str, default='normal')
    parser.add_argument('--optimizer_type', type=str, default='OffPolicyAsync')
    parser.add_argument('--off_policy', type=str, default=True)

    # env
    parser.add_argument('--env_id', default='PathTracking-v0')
    parser.add_argument('--num_agent', type=int, default=8)
    parser.add_argument('--num_future_data', type=int, default=0)
    parser.add_argument('--adv_act_dim', default=None)

    # learner
    parser.add_argument('--alg_name', default='NADP')
    parser.add_argument('--M', type=int, default=1)
    parser.add_argument('--num_rollout_list_for_policy_update', type=list, default=[25])
    parser.add_argument('--num_rollout_list_for_q_estimation', type=list, default=[25])
    parser.add_argument('--gamma', type=float, default=0.98)
    parser.add_argument('--gradient_clip_norm', type=float, default=3)
    parser.add_argument('--num_batch_reuse', type=int, default=1)

    # worker
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--worker_log_interval', type=int, default=5)
    parser.add_argument('--explore_sigma', type=float, default=None)

    # buffer
    parser.add_argument('--max_buffer_size', type=int, default=500000)
    parser.add_argument('--replay_starts', type=int, default=3000)
    parser.add_argument('--replay_batch_size', type=int, default=256)
    parser.add_argument('--replay_alpha', type=float, default=0.6)
    parser.add_argument('--replay_beta', type=float, default=0.4)
    parser.add_argument('--buffer_log_interval', type=int, default=40000)

    # tester and evaluator
    parser.add_argument('--num_eval_episode', type=int, default=5)
    parser.add_argument('--eval_log_interval', type=int, default=1)
    parser.add_argument('--fixed_steps', type=int, default=200)
    parser.add_argument('--eval_render', type=bool, default=False)
    num_eval_episode = parser.parse_args().num_eval_episode
    parser.add_argument('--num_eval_agent', type=int, default=num_eval_episode)

    # policy and model
    parser.add_argument('--obs_dim', type=int, default=None)
    parser.add_argument('--act_dim', type=int, default=None)
    parser.add_argument('--value_model_cls', type=str, default='MLP')
    parser.add_argument('--value_num_hidden_layers', type=int, default=2)
    parser.add_argument('--value_num_hidden_units', type=int, default=256)
    parser.add_argument('--value_hidden_activation', type=str, default='elu')
    parser.add_argument('--value_lr_schedule', type=list, default=[8e-5, 100000, 8e-6])
    parser.add_argument('--policy_model_cls', type=str, default='MLP')
    parser.add_argument('--policy_num_hidden_layers', type=int, default=2)
    parser.add_argument('--policy_num_hidden_units', type=int, default=256)
    parser.add_argument('--policy_hidden_activation', type=str, default='elu')
    parser.add_argument('--policy_out_activation', type=str, default='tanh')
    parser.add_argument('--policy_lr_schedule', type=list, default=[3e-5, 100000, 3e-6])
    parser.add_argument('--alpha', default=None)
    parser.add_argument('--alpha_lr_schedule', type=list, default=None)
    parser.add_argument('--policy_only', type=bool, default=False)
    parser.add_argument('--double_Q', type=bool, default=False)
    parser.add_argument('--target', type=bool, default=True)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--delay_update', type=int, default=1)
    parser.add_argument('--deterministic_policy', type=bool, default=True)
    parser.add_argument('--action_range', type=float, default=None)

    # adversary policy
    parser.add_argument('--adv_policy_model_cls', type=str, default='MLP')
    parser.add_argument('--adv_policy_lr_schedule', type=list, default=[3e-5, 100000, 3e-6])
    parser.add_argument('--adv_policy_num_hidden_layers', type=int, default=2)
    parser.add_argument('--adv_policy_num_hidden_units', type=int, default=256)
    parser.add_argument('--adv_policy_hidden_activation', type=str, default='elu')
    parser.add_argument('--adv_deterministic_policy', default=False, action='store_true')
    parser.add_argument('--adv_policy_out_activation', type=str, default='tanh')
    parser.add_argument('--update_adv_interval', type=int, default=1)
    parser.add_argument('--adv_act_bound', default=None)

    # preprocessor
    parser.add_argument('--obs_ptype', type=str, default='scale')
    num_future_data = parser.parse_args().num_future_data
    parser.add_argument('--obs_scale', type=list, default=[1., 1., 2., 1., 2.4, 1 / 1200] + [1.] * num_future_data)
    parser.add_argument('--rew_ptype', type=str, default='scale')
    parser.add_argument('--rew_scale', type=float, default=0.01)
    parser.add_argument('--rew_shift', type=float, default=0.)


    # optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKER)
    parser.add_argument('--num_learners', type=int, default=NUM_LEARNER)
    parser.add_argument('--num_buffers', type=int, default=NUM_BUFFER)
    parser.add_argument('--max_weight_sync_delay', type=int, default=300)
    parser.add_argument('--grads_queue_size', type=int, default=25)
    parser.add_argument('--grads_max_reuse', type=int, default=25)
    parser.add_argument('--eval_interval', type=int, default=3000)
    parser.add_argument('--save_interval', type=int, default=3000)
    parser.add_argument('--log_interval', type=int, default=100)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = '../results/NADP/{}/experiment-{time}'.format(noise_mode, time=time_now)
    parser.add_argument('--result_dir', type=str, default=results_dir)
    parser.add_argument('--log_dir', type=str, default=results_dir + '/logs')
    parser.add_argument('--model_dir', type=str, default=results_dir + '/models')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_load_ite', type=int, default=None)
    parser.add_argument('--ppc_load_dir', type=str, default=None)

    return parser.parse_args()


def built_parser(alg_name):
    args = built_NADP_parser()
    env = PathTrackingEnv(**vars(args))
    args.obs_dim, args.act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    args.adv_act_dim = env.adv_action_space.shape[0]
    return args

def main(alg_name):
    args = built_parser(alg_name)
    set_seed(args.seed)                          # todo:cannot work in all files
    logger.info('begin training agents with parameter {}'.format(str(args)))
    if args.mode == 'training':
        ray.init(object_store_memory=5120*1024*1024)
        os.makedirs(args.result_dir)
        with open(args.result_dir + '/config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        trainer = Trainer(policy_cls=NAME2POLICYCLS[args.policy_type],
                          worker_cls=NAME2WORKERCLS[args.worker_type],
                          learner_cls=NAME2LEARNERCLS[args.alg_name],
                          buffer_cls=NAME2BUFFERCLS[args.buffer_type],
                          optimizer_cls=NAME2OPTIMIZERCLS[args.optimizer_type],
                          evaluator_cls=NAME2EVALUATORCLS[args.evaluator_type],
                          args=args)
        if args.model_load_dir is not None:
            logger.info('loading model')
            trainer.load_weights(args.model_load_dir, args.model_load_ite)
        if args.ppc_load_dir is not None:
            logger.info('loading ppc parameter')
            trainer.load_ppc_params(args.ppc_load_dir)
        trainer.train()

    elif args.mode == 'testing':
        os.makedirs(args.test_log_dir)
        with open(args.test_log_dir + '/test_config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        tester = Tester(policy_cls=NAME2POLICYCLS[args.policy_type],
                        evaluator_cls=NAME2EVALUATORCLS[args.evaluator_type],
                        args=args)
        tester.test()


if __name__ == '__main__':
    main('NADP')
