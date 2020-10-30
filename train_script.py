#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: train_script.py
# =====================================

import argparse
import datetime
import json
import logging
import os

import ray

from buffer import PrioritizedReplayBuffer, ReplayBuffer
from learners.ampc import AMPCLearner
from optimizer import OffPolicyAsyncOptimizer
from policy import PolicyWithQs
from tester import Tester
from trainer import Trainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

# logging.getLogger().setLevel(logging.INFO)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NAME2LEARNERCLS = dict([('AMPC', AMPCLearner)])
NAME2BUFFERCLS = dict([('normal', ReplayBuffer), ('priority', PrioritizedReplayBuffer), ('None', None)])
NAME2OPTIMIZERCLS = dict([('OffPolicyAsync', OffPolicyAsyncOptimizer)])

def built_ampc_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = './results/toyota/experiment-2020-09-03-17-04-11'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[150000],
                           test_log_dir=test_log_dir,
                           num_eval_episode=5,
                           eval_log_interval=1,
                           fixed_steps=80))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # trainer
    parser.add_argument('--policy_type', type=str, default='PolicyWithQs')
    parser.add_argument('--buffer_type', type=str, default='normal')
    parser.add_argument('--optimizer_type', type=str, default='OffPolicyAsync')
    parser.add_argument('--off_policy', type=str, default=True)

    # env
    parser.add_argument("--env_id", default='CrossroadEnd2end-v0')
    parser.add_argument('--num_future_data', type=int, default=0)
    parser.add_argument('--training_task', type=str, default='left')

    # learner
    parser.add_argument("--alg_name", default='AMPC')
    parser.add_argument('--M', type=int, default=1)
    parser.add_argument('--num_rollout_list_for_policy_update', type=list, default=[15])
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--gradient_clip_norm", type=float, default=10)
    parser.add_argument("--init_punish_factor", type=float, default=10.)
    parser.add_argument("--pf_enlarge_interval", type=int, default=10000)
    parser.add_argument("--pf_amplifier", type=float, default=1.)

    # worker
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument("--worker_log_interval", type=int, default=5)
    parser.add_argument('--explore_sigma', type=float, default=None)

    # buffer
    parser.add_argument('--max_buffer_size', type=int, default=50000)
    parser.add_argument('--replay_starts', type=int, default=3000)
    parser.add_argument('--replay_batch_size', type=int, default=256)
    parser.add_argument('--replay_alpha', type=float, default=0.6)
    parser.add_argument('--replay_beta', type=float, default=0.4)
    parser.add_argument("--buffer_log_interval", type=int, default=40000)

    # tester and evaluator
    parser.add_argument("--num_eval_episode", type=int, default=2)
    parser.add_argument("--eval_log_interval", type=int, default=1)
    parser.add_argument("--fixed_steps", type=int, default=50)
    parser.add_argument("--eval_render", type=bool, default=True)

    # policy and model
    parser.add_argument("--policy_only", default=True, action='store_true')
    parser.add_argument("--policy_lr_schedule", type=list, default=[3e-5, 150000, 3e-6])
    parser.add_argument("--value_lr_schedule", type=list, default=[8e-5, 150000, 8e-6])
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_hidden_units', type=int, default=256)
    parser.add_argument("--deterministic_policy", default=True, action='store_true')
    parser.add_argument("--policy_out_activation", type=str, default='tanh')

    # preprocessor
    parser.add_argument('--obs_dim', default=None)
    parser.add_argument('--act_dim', default=None)
    parser.add_argument("--obs_preprocess_type", type=str, default='scale')
    num_future_data = parser.parse_args().num_future_data
    training_task = parser.parse_args().training_task
    TASK2VEHNUM = {'left': 6, 'straight': 7, 'right': 5}
    obs_scale_factor = [0.2, 1., 2., 1/20., 1/20, 1/180.] + \
                       [1., 1/15., 0.2] * (1 + num_future_data) + \
                       [1/20., 1/20., 0.2, 1/180.] * TASK2VEHNUM[training_task]
    parser.add_argument("--obs_scale_factor", type=list, default=obs_scale_factor)
    parser.add_argument("--reward_preprocess_type", type=str, default='scale')
    parser.add_argument("--reward_scale_factor", type=float, default=1.)

    # optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_updated_steps', type=int, default=150000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_learners', type=int, default=5)
    parser.add_argument('--num_buffers', type=int, default=1)
    parser.add_argument('--max_weight_sync_delay', type=int, default=300)
    parser.add_argument('--grads_queue_size', type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=100)

    # IO
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = './results/toyota/experiment-{time}'.format(time=time_now)
    parser.add_argument("--result_dir", type=str, default=results_dir)
    parser.add_argument("--log_dir", type=str, default=results_dir + '/logs')
    parser.add_argument("--model_dir", type=str, default=results_dir + '/models')
    parser.add_argument("--model_load_dir", type=str, default=None)
    parser.add_argument("--model_load_ite", type=int, default=None)
    parser.add_argument("--ppc_load_dir", type=str, default=None)

    return parser.parse_args()

def built_parser(alg_name):
    if alg_name == 'AMPC':
        return built_ampc_parser()

def main(alg_name):
    args = built_parser(alg_name)
    logger.info('begin training agents with parameter {}'.format(str(args)))
    if args.mode == 'training':
        ray.init(object_store_memory=5120*1024*1024)
        os.makedirs(args.result_dir)
        with open(args.result_dir + '/config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        trainer = Trainer(policy_cls=PolicyWithQs,
                          learner_cls=NAME2LEARNERCLS[args.alg_name],
                          buffer_cls=NAME2BUFFERCLS[args.buffer_type],
                          optimizer_cls=NAME2OPTIMIZERCLS[args.optimizer_type],
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
        tester = Tester(policy_cls=PolicyWithQs,
                        args=args)
        tester.test()


if __name__ == '__main__':
    main('AMPC')
