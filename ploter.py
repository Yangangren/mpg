#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/25
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ploter.py
# =====================================

import copy
import os
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import json
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.core.util import event_pb2
from tester import Tester
from policy import Policy4Toyota
from evaluator import Evaluator
NAME2POLICIES = dict([('Policy4Toyota', Policy4Toyota)])
NAME2EVALUATORS = dict([('Evaluator', Evaluator)])

sns.set(style="darkgrid")

palette = [(1.0, 0.48627450980392156, 0.0),
                    (0.9098039215686274, 0.0, 0.043137254901960784),
                    (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
                    (0.6235294117647059, 0.2823529411764706, 0.0),]

WINDOWSIZE = 5


def min_n(inp_list, n):
    return sorted(inp_list)[:n]


def plot_opt_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot = ['learner_stats/scalar/obj_loss',
                'learner_stats/scalar/obj_v_loss',
                'learner_stats/scalar/pg_loss',
                'learner_stats/scalar/punish_factor',
                'learner_stats/scalar/real_punish_term',
                'learner_stats/scalar/veh2road4real',
                'learner_stats/scalar/veh2veh4real',
                ]
    env_list = ['CrossroadEnd2endAdv-v0']
    task_list = ['adv_noise', 'no_noise']
    palette = "bright"
    lbs = ['adv_noise', 'no_noise']
    dir_str = './results/{}/{}'
    df_list = []
    for alg in env_list:
        for task in task_list:
            data2plot_dir = dir_str.format(alg, task)
            data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
            for num_run, dir in enumerate(data2plot_dirs_list):
                opt_dir = data2plot_dir + '/' + dir + '/logs/optimizer'
                opt_file = os.path.join(opt_dir,
                                         [file_name for file_name in os.listdir(opt_dir) if
                                          file_name.startswith('events')][0])
                opt_summarys = tf.data.TFRecordDataset([opt_file])
                data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
                data_in_one_run_of_one_alg.update({'iteration': []})
                for opt_summary in opt_summarys:
                    event = event_pb2.Event.FromString(opt_summary.numpy())
                    for v in event.summary.value:
                        t = tf.make_ndarray(v.tensor)
                        for tag in tag2plot:
                            if tag in v.tag:
                                data_in_one_run_of_one_alg[tag].append(float(t))
                                data_in_one_run_of_one_alg['iteration'].append(int(event.step))
                len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
                period = int(len1 / len2)
                data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i * period] / 10000. for
                                                           i in range(len2)]

                data_in_one_run_of_one_alg = {key: val[2:] for key, val in data_in_one_run_of_one_alg.items()}
                data_in_one_run_of_one_alg.update(dict(algorithm=alg, task=task, num_run=num_run))
                df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
                for tag in tag2plot:
                    df_in_one_run_of_one_alg[tag+'_smo'] = df_in_one_run_of_one_alg[tag].rolling(WINDOWSIZE, min_periods=1).mean()
                df_list.append(df_in_one_run_of_one_alg)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    figsize = (10, 8)
    axes_size = [0.12, 0.12, 0.87, 0.87]
    fontsize = 25

    f1 = plt.figure(1, figsize=figsize)
    ax1 = f1.add_axes([0.13, 0.12, 0.86, 0.87])
    sns.lineplot(x="iteration", y="learner_stats/scalar/obj_loss_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette,)
    # plt.ylim(0, 100)
    handles, labels = ax1.get_legend_handles_labels()
    labels = lbs
    ax1.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=fontsize)
    ax1.set_ylabel('$J_{\\rm actor}$', fontsize=fontsize)
    ax1.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./loss_actor.pdf')

    f2 = plt.figure(2, figsize=figsize)
    ax2 = f2.add_axes([0.15, 0.12, 0.85, 0.86])
    sns.lineplot(x="iteration", y="learner_stats/scalar/obj_v_loss_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    plt.ylim(0, 1000)
    ax2.set_ylabel('$J_{\\rm critic}$', fontsize=fontsize)
    ax2.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./loss_critic.pdf')

    f3 = plt.figure(3, figsize=figsize)
    ax3 = f3.add_axes([0.12, 0.12, 0.87, 0.87])
    sns.lineplot(x="iteration", y="learner_stats/scalar/real_punish_term_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    ax3.set_ylabel('$J_{\\rm penalty}$', fontsize=fontsize)
    ax3.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./loss_penalty.pdf')

    f4 = plt.figure(4, figsize=figsize)
    ax4 = f4.add_axes([0.11, 0.12, 0.86, 0.87])
    sns.lineplot(x="iteration", y="learner_stats/scalar/veh2veh4real", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    ax4.set_ylabel('Ego-to-veh Penalty', fontsize=fontsize)
    ax4.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    plt.ylim(0, 3)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./ego2veh_penalty.pdf')
    plt.show()


def plot_eva_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot = ['evaluation/total_return',
                'evaluation/track_return',
                ]
    env_list = ['CrossroadEnd2endAdv-v0']
    task_list = ['adv_noise', 'no_noise']
    palette = "bright"
    lbs = ['adv_noise', 'no_noise']
    dir_str = './results/{}/{}'
    df_list = []
    for alg in env_list:
        for task in task_list:
            data2plot_dir = dir_str.format(alg, task)
            data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
            for num_run, dir in enumerate(data2plot_dirs_list):
                opt_dir = data2plot_dir + '/' + dir + '/logs/evaluator'
                opt_file = os.path.join(opt_dir,
                                        [file_name for file_name in os.listdir(opt_dir) if
                                         file_name.startswith('events')][0])
                opt_summarys = tf.data.TFRecordDataset([opt_file])
                data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
                data_in_one_run_of_one_alg.update({'iteration': []})
                for opt_summary in opt_summarys:
                    event = event_pb2.Event.FromString(opt_summary.numpy())
                    for v in event.summary.value:
                        t = tf.make_ndarray(v.tensor)
                        for tag in tag2plot:
                            if tag in v.tag:
                                data_in_one_run_of_one_alg[tag].append(float(t))
                                data_in_one_run_of_one_alg['iteration'].append(int(event.step))
                len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
                period = int(len1 / len2)
                data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i * period] / 10000. for
                                                           i in range(len2)]

                data_in_one_run_of_one_alg = {key: val[:] for key, val in data_in_one_run_of_one_alg.items()}
                data_in_one_run_of_one_alg.update(dict(algorithm=alg, task=task, num_run=num_run))
                df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
                for tag in tag2plot:
                    df_in_one_run_of_one_alg[tag+'_smo'] = df_in_one_run_of_one_alg[tag].rolling(WINDOWSIZE, min_periods=1).mean()
                df_list.append(df_in_one_run_of_one_alg)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    figsize = (10, 8)
    axes_size = [0.12, 0.12, 0.87, 0.87]
    fontsize = 25

    f1 = plt.figure(1, figsize=figsize)
    ax1 = f1.add_axes([0.13, 0.12, 0.86, 0.87])
    sns.lineplot(x="iteration", y="evaluation/total_return_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette,)
    plt.ylim(-2000, 0)
    handles, labels = ax1.get_legend_handles_labels()
    labels = lbs
    ax1.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=fontsize)
    ax1.set_ylabel('$J_{\\rm TAR}$', fontsize=fontsize)
    ax1.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./eva_total_return.pdf')

    f2 = plt.figure(2, figsize=figsize)
    ax2 = f2.add_axes([0.15, 0.12, 0.85, 0.86])
    sns.lineplot(x="iteration", y="evaluation/track_return_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    # plt.ylim(0, 1000)
    ax2.set_ylabel('$J_{\\rm track return}$', fontsize=fontsize)
    ax2.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./eva_track_return.pdf')

    plt.show()


def plot_trained_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot = ['evaluation/total_return',
                'evaluation/track_return',
                ]
    env_list = ['CrossroadEnd2endAdv-v0']
    task_list = ['adv_noise', 'no_noise', 'rand_noise']
    palette = "bright"
    lbs = ['adv_noise', 'no_noise', 'rand_noise']
    dir_str = './results/{}/{}'
    df_list = []
    for alg in env_list:
        for task in task_list:
            data2plot_dir = dir_str.format(alg, task)
            data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
            for num_run, dir in enumerate(data2plot_dirs_list):
                opt_dir = data2plot_dir + '/' + dir + '/tester'
                opt_file = os.path.join(opt_dir,
                                        [file_name for file_name in os.listdir(opt_dir) if
                                         file_name.startswith('events')][0])
                opt_summarys = tf.data.TFRecordDataset([opt_file])
                data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
                data_in_one_run_of_one_alg.update({'iteration': []})
                for opt_summary in opt_summarys:
                    event = event_pb2.Event.FromString(opt_summary.numpy())
                    for v in event.summary.value:
                        t = tf.make_ndarray(v.tensor)
                        for tag in tag2plot:
                            if tag in v.tag:
                                data_in_one_run_of_one_alg[tag].append(float(t))
                                data_in_one_run_of_one_alg['iteration'].append(int(event.step))
                len1, len2 = len(data_in_one_run_of_one_alg['iteration']), len(data_in_one_run_of_one_alg[tag2plot[0]])
                period = int(len1 / len2)
                data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i * period] / 10000. for
                                                           i in range(len2)]

                data_in_one_run_of_one_alg = {key: val[:] for key, val in data_in_one_run_of_one_alg.items()}
                data_in_one_run_of_one_alg.update(dict(algorithm=alg, task=task, num_run=num_run))
                df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
                for tag in tag2plot:
                    df_in_one_run_of_one_alg[tag+'_smo'] = df_in_one_run_of_one_alg[tag].rolling(WINDOWSIZE, min_periods=1).mean()
                df_list.append(df_in_one_run_of_one_alg)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    figsize = (10, 8)
    axes_size = [0.12, 0.12, 0.87, 0.87]
    fontsize = 25

    f1 = plt.figure(1, figsize=figsize)
    ax1 = f1.add_axes([0.13, 0.12, 0.86, 0.87])
    sns.lineplot(x="iteration", y="evaluation/total_return_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette,)
    plt.ylim(-1000, 0)
    handles, labels = ax1.get_legend_handles_labels()
    labels = lbs
    ax1.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=fontsize)
    ax1.set_ylabel('TAR', fontsize=fontsize)
    ax1.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./test_total_return.pdf')

    f2 = plt.figure(2, figsize=figsize)
    ax2 = f2.add_axes([0.15, 0.12, 0.85, 0.86])
    sns.lineplot(x="iteration", y="evaluation/track_return_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    plt.ylim(-400, 0)
    ax2.set_ylabel('track return', fontsize=fontsize)
    ax2.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./test_track_return.pdf')

    plt.show()

def main(dirs_dict_for_plot=None):
    # os.makedirs(args.test_log_dir)
    # with open(args.test_log_dir + '/test_config.json', 'w', encoding='utf-8') as f:
    #     json.dump(vars(args), f, ensure_ascii=False, indent=4)

    env_list = ['CrossroadEnd2endAdv-v0']
    task_list = ['adv_noise', 'no_noise', 'rand_noise']
    dir_str = './results/{}/{}'
    for alg in env_list:
        for task in task_list:
            data2plot_dir = dir_str.format(alg, task)
            data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
            for num_run, dir in enumerate(data2plot_dirs_list):
                parser = argparse.ArgumentParser()
                parser.add_argument('--adv_env_id', default='CrossroadEnd2endAdvTest-v0')
                test_dir = data2plot_dir + '/' + dir
                print('current policy:', test_dir)
                params = json.loads(open(test_dir + '/config.json').read())
                time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                test_log_dir = test_dir + '/tester'
                params.update(dict(test_dir=test_dir,
                                   test_iter_list=None,
                                   test_log_dir=test_log_dir,
                                   num_eval_episode=4,
                                   eval_log_interval=1,
                                   fixed_steps=120,
                                   eval_render=False))
                for key, val in params.items():
                    parser.add_argument("-" + key, default=val)
                args = parser.parse_args()
                args.mode = 'testing'
                args.test_iter_list = list(range(0, args.max_iter, args.save_interval))
                tester = Tester(policy_cls=NAME2POLICIES[args.policy_type],
                                evaluator_cls=NAME2EVALUATORS[args.evaluator_type],
                                args=args)
                tester.test()


if __name__ == "__main__":
    # plot_opt_results_of_all_alg_n_runs()
    plot_eva_results_of_all_alg_n_runs()
    # main()
    # plot_trained_results_of_all_alg_n_runs()