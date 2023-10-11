#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2023/04/25
# @Author  : Yangang Ren (Tsinghua Univ.)
# @FileName: ploter.py
# @Function: Plot track error and robust test
# for Chapter 3 of Doctoral dissertation
# =====================================

import copy
import os

import argparse
import datetime
import numpy as np
import json
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.core.util import event_pb2
from tester import Tester
from policy import PolicyWithQs
from evaluator import Evaluator
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
zhfont1 = fm.FontProperties(fname=r'D:\simsun.ttf',size=14)

plt.rc('font', family='Times New Roman')
# plt.rcParams['mathtext.fontset'] = 'stix'
import matplotlib
matplotlib.rcParams['mathtext.default'] = 'regular'

from matplotlib import rcParams
from matplotlib.pyplot import MultipleLocator

config = {
    "font.family": "serif",
    "font.size": 25,
    "mathtext.fontset": "stix",
    "font.serif": ["STSONG"],
}
rcParams.update(config)

plt.rcParams["font.sans-serif"] = ["STSONG"]
plt.rcParams["axes.unicode_minus"] = False
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams[u'font.sans-serif'] = ['SIMSUN']
# plt.rcParams['axes.unicode_minus'] = False
plt.rc('font', family='Times New Roman')
# config = {
#     "font.family":'serif',
#     "font.size": 18,
#     "mathtext.fontset":'stix',
#     "font.serif": ['SimSun'],
# }
# plt.rcParams.update(config)

NAME2POLICIES = dict([('PolicyWithQs', PolicyWithQs)])
NAME2EVALUATORS = dict([('Evaluator', Evaluator)])

sns.set(style=None)


def plot_eval_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot = ['evaluation/episode_return', 'evaluation/delta_y_mse', 'evaluation/delta_phi_mse', 'evaluation/delta_v_mse']
    # tar: 'NADP'; error: 'error_plot'
    env_list = ['error_plot']
    task_list = ['adv_noise', 'aaac', 'no_noise', 'adv_noise_smooth_uniform', 'adv_noise_smooth'][::-1]
    palette = "bright"
    lbs = ['RARL', 'RPG', 'ADP', 'SAAC-u', 'SAAC-a'][::-1]
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
                if task == 'adv_noise':
                    WINDOWSIZE = 2
                elif task == 'aaac':
                    WINDOWSIZE = 2
                else:
                    WINDOWSIZE = 2
                data_in_one_run_of_one_alg.update(dict(algorithm=alg, task=task, num_run=num_run))
                df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
                for tag in tag2plot:
                    df_in_one_run_of_one_alg[tag+'_smo'] = df_in_one_run_of_one_alg[tag].rolling(WINDOWSIZE, min_periods=1).mean()
                df_list.append(df_in_one_run_of_one_alg)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    fontsize = 30

    f4 = plt.figure(4, figsize=(10, 8))
    ax4 = f4.add_axes([0.10, 0.115, 0.88, 0.85])
    sns.lineplot(x="iteration", y="evaluation/delta_y_mse_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, style='task')
    ax4.set_xlabel(r"迭代次数($\times 10^4$)", fontsize=fontsize)
    ax4.set_ylabel(r"位置误差($\mathrm {m}$)", fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize-5)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize-5)
    handles, labels = ax4.get_legend_handles_labels()
    labels = lbs
    ax4.legend(handles=handles, labels=labels, loc='upper right', frameon=False, prop={'size':25, 'family': 'Times New Roman'})
    plt.xlim(0., 10)
    plt.ylim(0., 25)
    plt.savefig('./results/error_plot/position_error.pdf')

    f5 = plt.figure(5, figsize=(10, 8))
    ax5 = f5.add_axes([0.085, 0.11, 0.90, 0.87])
    sns.lineplot(x="iteration", y="evaluation/delta_v_mse_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    ax5.set_xlabel(r"迭代次数$[\times 10^4]$", fontproperties=zhfont1, fontsize=fontsize)
    ax5.set_ylabel(r"速度误差$[\rm m/s]$", fontproperties=zhfont1, fontsize=fontsize)
    # handles, labels = ax5.get_legend_handles_labels()
    # ax5.legend(handles=handles[0:], labels=labels[0:])
    plt.xlim(0., 10)
    # plt.ylim(0., 40)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize-2)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize-2)
    plt.savefig('./results/error_plot/velocity_error.pdf')

    f6 = plt.figure(6, figsize=(10, 8))
    ax6 = f6.add_axes([0.12, 0.115, 0.85, 0.87])
    sns.lineplot(x="iteration", y="evaluation/delta_phi_mse_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, style='task', legend=False)
    ax6.set_xlabel(r"迭代次数($\times 10^4$)", fontsize=fontsize)
    ax6.set_ylabel(r'航向角误差($\rm rad$)', fontsize=fontsize)
    handles, labels = ax6.get_legend_handles_labels()
    labels = lbs
    # ax6.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=fontsize)
    plt.xlim(0., 10)
    # plt.ylim(0., 10)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize-5)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize-5)
    plt.savefig('./results/error_plot/heading_error.pdf')


def plot_robust_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    WINDOWSIZE = 2
    tag2plot = ['evaluation/episode_return', 'evaluation/delta_y_mse', 'evaluation/delta_phi_mse', 'evaluation/delta_v_mse']
    env_list = ['robust test']
    task_list = ['no_noise', 'adv_noise_smooth'][::-1]
    palette = "bright"
    lbs = ['ADP', 'SAAC'][::-1]
    dir_str = './results/{}/{}'
    df_list = []
    for alg in env_list:
        for task in task_list:
            data2plot_dir = dir_str.format(alg, task)
            data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
            for num_run, dir in enumerate(data2plot_dirs_list):
                opt_dir = data2plot_dir + '/' + dir + '/logs/tester'
                for iter_run, file in enumerate(os.listdir(opt_dir)):
                    final_file = opt_dir + '/' + file
                    opt_file = os.path.join(final_file,
                                            [file_name for file_name in os.listdir(final_file) if
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
                    data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i * period] for
                                                               i in range(len2)]
                    data_in_one_run_of_one_alg['iteration'] = [data_in_one_run_of_one_alg['iteration'][i]* 0.06 - 0.3 for i in range(len(data_in_one_run_of_one_alg['iteration']))]
                    data_in_one_run_of_one_alg = {key: val[:] for key, val in data_in_one_run_of_one_alg.items()}
                    data_in_one_run_of_one_alg.update(dict(algorithm=alg, task=task, num_run=num_run, iter_run=iter_run))
                    df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
                    for tag in tag2plot:
                        df_in_one_run_of_one_alg[tag+'_smo'] = df_in_one_run_of_one_alg[tag].rolling(WINDOWSIZE, min_periods=1).mean()
                    df_list.append(df_in_one_run_of_one_alg)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    fontsize = 30

    f1 = plt.figure(1, figsize=(12, 8))
    ax1 = f1.add_axes([0.115, 0.12, 0.86, 0.86])
    sns.lineplot(x="iteration", y="evaluation/episode_return_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, ci=90, style='task')
    # plt.ylim(-500, 0)
    plt.xlim(-0.3, 0.3)
    handles, labels = ax1.get_legend_handles_labels()
    labels = lbs
    ax1.legend(handles=handles, labels=labels, loc='lower right', frameon=False, prop={'size':25, 'family': 'Times New Roman'})
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize-5)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize-5)
    ax1.set_xlabel(r"横向速度偏移($\rm m/s$)", fontsize=fontsize)
    ax1.set_ylabel('平均累计回报', fontproperties=zhfont1, fontsize=fontsize)
    plt.savefig('./results/robust test/robust_test_return.pdf')

    f4 = plt.figure(4, figsize=(16, 8))
    ax4 = f4.add_axes([0.08, 0.11, 0.89, 0.87])
    sns.lineplot(x="iteration", y="evaluation/delta_y_mse_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette)
    ax4.set_ylabel('Position Error [m]', fontsize=fontsize)
    ax4.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    handles, labels = ax4.get_legend_handles_labels()
    labels = lbs
    ax4.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=fontsize)
    # plt.xlim(0., 10)
    # plt.ylim(0., 25)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./results/robust test/robust_test_position_error.pdf')

    f5 = plt.figure(5, figsize=(16, 8))
    ax5 = f5.add_axes([0.07, 0.11, 0.90, 0.87])
    sns.lineplot(x="iteration", y="evaluation/delta_v_mse_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    ax5.set_ylabel('Velocity Error [m/s]', fontsize=fontsize)
    ax5.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    # handles, labels = ax5.get_legend_handles_labels()
    # ax5.legend(handles=handles[0:], labels=labels[0:])
    # plt.xlim(0., 10)
    # plt.ylim(0., 40)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./results/robust test/robust_test_velocity_error.pdf')

    f6 = plt.figure(6, figsize=(16, 8))
    ax6 = f6.add_axes([0.09, 0.11, 0.89, 0.87])
    sns.lineplot(x="iteration", y="evaluation/delta_phi_mse_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    ax6.set_ylabel('Heading Angle Error [rad]', fontsize=fontsize)
    ax6.set_xlabel("Iteration [x10000]", fontsize=fontsize)
    handles, labels = ax6.get_legend_handles_labels()
    labels = lbs
    # ax6.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=fontsize)
    # plt.xlim(0., 10)
    # plt.ylim(0., 10)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig('./results/robust test/robust_test_heading_error.pdf')


def main(dirs_dict_for_plot=None):
    env_list = ['tar_plot']
    task_list = ['adv_noise_smooth_uniform']   # 'adv_noise', 'adv_noise_smooth'
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
                disturb_env = []
                test_log_dir = test_dir + '/tester'
                params.update(dict(test_dir=test_dir,
                                   disturb_env=disturb_env,
                                   test_iter_list=None,
                                   test_log_dir=test_log_dir,
                                   num_eval_episode=5,
                                   eval_log_interval=1,
                                   fixed_steps=200,
                                   eval_render=False,
                                   ))
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
    # main()
    plot_eval_results_of_all_alg_n_runs()
    plot_robust_results_of_all_alg_n_runs()