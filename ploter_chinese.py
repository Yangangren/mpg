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
zhfont1=fm.FontProperties(fname='/home/guanyang/simsun.ttf', size=14)
enfont1=fm.FontProperties(fname='/home/guanyang/Times New Roman.ttf', size=20)
# plt.rc('font', family='Times New Roman')
plt.rc('font', family='Times New Roman')
import matplotlib
matplotlib.rcParams['mathtext.default'] = 'regular'
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams[u'font.sans-serif'] = ['SIMSUN']
# plt.rcParams['axes.unicode_minus'] = False

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

def help_func():
    if 1:
        tag2plot = ['episode_return', 'episode_len', 'delta_y_mse', 'delta_phi_mse', 'delta_v_mse',
                    'stationary_rew_mean', 'steer_mse', 'acc_mse']
        alg_list = ['no_noise', 'adv_noise', 'adv_noise_smooth', 'adv_noise_smooth_uniform']
        lbs = ['ADP', 'RaRL', 'SaAC', 'SaAC-u']
        palette = "bright"
        goal_perf_list = [-200, -100, -50, -30, -20, -10, -5]
        dir_str = './results/NADP/{}'
    else:
        tag2plot = ['episode_return', 'episode_len', 'x_mse', 'theta_mse', 'xdot_mse', 'thetadot_mse']
        alg_list = ['MPG-v2', 'NADP', 'TD3', 'SAC']
        lbs = ['MPG-v2', r'$n$-step ADP', 'TD3', 'SAC']
        palette = [(1.0, 0.48627450980392156, 0.0),
                   (0.9098039215686274, 0.0, 0.043137254901960784),
                   (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
                   (0.6235294117647059, 0.2823529411764706, 0.0),]
        goal_perf_list = [-20, -10, -2, -1, -0.5, -0.1, -0.01]
        dir_str = './results/{}/data2plot_mujoco'
    return tag2plot, alg_list, lbs, palette, goal_perf_list, dir_str


def plot_eval_results_of_all_alg_n_runs(dirs_dict_for_plot=None, fname=None):
    tag2plot = ['evaluation/episode_return', 'evaluation/delta_y_mse', 'evaluation/delta_phi_mse', 'evaluation/delta_v_mse']
    # tar: 'NADP'; error: 'error_plot'
    env_list = ['NADP']
    task_list = ['adv_noise', 'no_noise', 'aaac', 'adv_noise_smooth_uniform', 'adv_noise_smooth'][::-1]
    palette = "bright"
    lbs = ['RARL', 'ADP', 'RPG', 'SAAC-u', 'SAAC-a'][::-1]
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
    figsize = (12, 8)
    fontsize = 25

    if fname is not None:
        f2 = plt.figure(1, figsize=figsize)
        f2.add_axes([0.13, 0.12, 0.86, 0.87])
        ax2 = sns.boxplot(x="iteration", y="evaluation/episode_return", hue="task",
                          data=total_dataframe, palette=palette,)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylim(-10000, 50)
        handles, labels = ax2.get_legend_handles_labels()
        labels = lbs
        ax2.legend(handles=handles, labels=labels, loc='lower right', frameon=False, fontsize=fontsize)
        ax2.set_ylabel('平均累计收益', fontsize=fontsize)
        ax2.set_xlabel(r"迭代次数$[\times 10^4]$", fontsize=fontsize)
        # plt.yticks(fontsize=fontsize-2)
        # plt.xticks(fontsize=fontsize-2)
        plt.savefig('./results/tar_plot/test_mean_std.pdf')
        plt.show()

    else:
        f1 = plt.figure(1, figsize=(12, 8))
        ax1 = f1.add_axes([0.13, 0.12, 0.85, 0.86])
        sns.lineplot(x="iteration", y="evaluation/episode_return_smo", hue="task",
                     data=total_dataframe, linewidth=2, palette=palette, ci=90, err_style=None)
        plt.ylim(-600, 0)
        plt.xlim(0, 10.)
        handles, labels = ax1.get_legend_handles_labels()
        labels = lbs
        ax1.legend(handles=handles, labels=labels, loc='lower right', frameon=False, fontsize=fontsize + 6, prop=enfont1)
        ax1.set_ylabel('平均累计收益', fontproperties=zhfont1, fontsize=fontsize)
        ax1.set_xlabel(r"迭代次数$[\times 10^4]$", fontproperties=zhfont1, fontsize=fontsize)
        plt.yticks(fontproperties=enfont1, fontsize=fontsize-2)
        plt.xticks(fontproperties=enfont1, fontsize=fontsize-2)
        plt.savefig('./results/tar_plot/test_total_return.pdf')
        plt.close(f1)

        f2 = plt.figure(2, figsize=(16, 8))
        ax2 = f2.add_axes([0.13, 0.12, 0.72, 0.87])
        legend_list = []
        for seed, df in enumerate(df_list):
            if df['task'][0] == 'adv_noise':
                ax2.plot(df['iteration'], df["evaluation/episode_return_smo"], linewidth=2)
                legend_list.append('seed' + " " + str(seed))
        plt.ylim(-3000, 0)
        handles, labels = ax1.get_legend_handles_labels()
        # ax2.legend(handles=handles, labels=legend_list, loc='upper right', frameon=False, fontsize=fontsize)
        ax2.legend(legend_list, fontsize=20, frameon=False, loc=(1.0, 0.2))
        ax2.set_ylabel('Total Average Return', fontsize=fontsize)
        ax2.set_xlabel(r"迭代次数$[\times 10^4]$", fontproperties=zhfont1, fontsize=fontsize)
        plt.yticks(fontproperties=enfont1, fontsize=fontsize-2)
        plt.xticks(fontproperties=enfont1, fontsize=fontsize-2)
        plt.savefig('./results/tar_plot/rarl_every_return.pdf')
        plt.close(f2)

        f3 = plt.figure(3, figsize=(16, 8))
        ax3 = f3.add_axes([0.13, 0.12, 0.72, 0.87])
        for df in df_list:
            if df['task'][0] == 'adv_noise_smooth':
                sns.lineplot(x=df['iteration'], y=df["evaluation/episode_return_smo"], linewidth=2, palette=palette, ax=ax3)
        plt.ylim(-1000, 0)
        handles, labels = ax1.get_legend_handles_labels()
        # ax3.legend(handles=handles, labels=labels, loc='lower right', frameon=False, fontsize=fontsize)
        ax3.set_ylabel('Total Average Return', fontsize=fontsize)
        ax3.set_xlabel(r"Iteration$[\times 10^4]$", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig('./results/tar_plot/saac_every_return.pdf')
        plt.close(f3)

        f4 = plt.figure(4, figsize=(10, 8))
        ax4 = f4.add_axes([0.10, 0.11, 0.88, 0.87])
        sns.lineplot(x="iteration", y="evaluation/delta_y_mse_smo", hue="task",
                     data=total_dataframe, linewidth=2, palette=palette)
        ax4.set_xlabel(r"迭代次数$[\times 10^4]$", fontproperties=zhfont1, fontsize=fontsize)
        ax4.set_ylabel(r"位置误差[$\mathrm {m}$]", fontproperties=zhfont1, fontsize=fontsize)
        plt.yticks(fontproperties=enfont1, fontsize=fontsize-2)
        plt.xticks(fontproperties=enfont1, fontsize=fontsize-2)
        handles, labels = ax4.get_legend_handles_labels()
        labels = lbs
        ax4.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=fontsize + 6, prop=enfont1)
        plt.xlim(0., 10)
        plt.ylim(0., 25)
        plt.savefig('./results/tar_plot/position_error.pdf')

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
        plt.yticks(fontproperties=enfont1, fontsize=fontsize-2)
        plt.xticks(fontproperties=enfont1, fontsize=fontsize-2)
        plt.savefig('./results/tar_plot/velocity_error.pdf')

        f6 = plt.figure(6, figsize=(10, 8))
        ax6 = f6.add_axes([0.10, 0.11, 0.88, 0.87])
        sns.lineplot(x="iteration", y="evaluation/delta_phi_mse_smo", hue="task",
                     data=total_dataframe, linewidth=2, palette=palette, legend=False)
        ax6.set_xlabel(r"迭代次数$[\times 10^4]$", fontproperties=zhfont1, fontsize=fontsize)
        ax6.set_ylabel(r'航向角误差[$\rm rad$]', fontproperties=zhfont1, fontsize=fontsize)
        handles, labels = ax6.get_legend_handles_labels()
        labels = lbs
        # ax6.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=fontsize)
        plt.xlim(0., 10)
        # plt.ylim(0., 10)
        plt.yticks(fontproperties=enfont1, fontsize=fontsize-2)
        plt.xticks(fontproperties=enfont1, fontsize=fontsize-2)
        plt.savefig('./results/tar_plot/heading_error.pdf')

        allresults = {}
        results2print = {}
        result2conv = {}
        for alg, group in total_dataframe.groupby('task'):
            if alg == 'adv_noise_smooth':
                for ite, group1 in group.groupby('num_run'):
                    print(ite, list(group1["evaluation/episode_return_smo"])[-2:])
            if alg == 'adv_noise':
                for ite, group1 in group.groupby('num_run'):
                    print(ite, list(group1["evaluation/episode_return_smo"])[-3:])
            allresults.update({alg: []})
            result2conv.update({alg: []})
            stop_flag = 0
            for ite, group1 in group.groupby('iteration'):
                mean = group1['evaluation/episode_return_smo'].mean()
                if stop_flag != 1 and mean > -25.:
                    result2conv[alg].append(ite)
                    stop_flag = 1
                std = group1['evaluation/episode_return_smo'].std()
                allresults[alg].append((mean, std))
            if stop_flag == 0:
                result2conv[alg].append(np.inf)

        for alg, result in allresults.items():
            mean, std = sorted(result, key=lambda x: x[0])[-1]
            results2print.update({alg: [mean, std]})

        print(results2print)
        print(result2conv)


def plot_robust_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    WINDOWSIZE = 2
    tag2plot = ['evaluation/episode_return', 'evaluation/delta_y_mse', 'evaluation/delta_phi_mse', 'evaluation/delta_v_mse']
    env_list = ['robust test']
    task_list = ['no_noise', 'adv_noise_smooth'][::-1]
    palette = "bright"
    lbs = ['ADP', 'SAAC-a'][::-1]
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
    fontsize = 25

    f1 = plt.figure(1, figsize=(12, 8))
    ax1 = f1.add_axes([0.115, 0.12, 0.86, 0.86])
    sns.lineplot(x="iteration", y="evaluation/episode_return_smo", hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, ci=90)
    # plt.ylim(-500, 0)
    plt.xlim(-0.3, 0.3)
    handles, labels = ax1.get_legend_handles_labels()
    labels = lbs
    ax1.legend(handles=handles, labels=labels, loc='lower right', frameon=False, fontsize=fontsize, prop=enfont1)
    plt.yticks(fontproperties=enfont1, fontsize=fontsize-2)
    plt.xticks(fontproperties=enfont1, fontsize=fontsize-2)
    ax1.set_xlabel(r"横向速度偏移[$\rm m/s$]", fontproperties=zhfont1, fontsize=fontsize)
    ax1.set_ylabel('平均累计收益', fontproperties=zhfont1, fontsize=fontsize)
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


def compute_convergence_speed(goal_perf, dirs_dict_for_plot=None):
    _, alg_list, _, _, _, dir_str = help_func()
    result_dict = {}
    for alg in alg_list:
        result_dict.update({alg: []})
        data2plot_dir = dir_str.format(alg)
        data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
        for num_run, dir in enumerate(data2plot_dirs_list):
            stop_flag = 0
            eval_dir = data2plot_dir + '/' + dir + '/tester'
            eval_file = os.path.join(eval_dir,
                                     [file_name for file_name in os.listdir(eval_dir) if
                                      file_name.startswith('events')][0])
            eval_summarys = tf.data.TFRecordDataset([eval_file])

            for eval_summary in eval_summarys:
                if stop_flag != 1:
                    event = event_pb2.Event.FromString(eval_summary.numpy())
                    for v in event.summary.value:
                        if stop_flag != 1:
                            t = tf.make_ndarray(v.tensor)
                            step = float(event.step)
                            if 'evaluation/episode_return' in v.tag:
                                if t > goal_perf:
                                    result_dict[alg].append(step)
                                    stop_flag = 1
            if stop_flag == 0:
                result_dict[alg].append(np.inf)
    for tag, value in result_dict.items():
        print(tag, sum(value)/len(value))
    return result_dict


def min_n(inp_list, n):
    return sorted(inp_list)[:n]


def plot_trained_results_of_all_alg_n_runs(dirs_dict_for_plot=None, fname=None):
    tag2plot = ['evaluation/episode_return']
    env_list = ['NADP']
    task_list = ['adv_noise', 'adv_noise_smooth']
    palette = "bright"
    lbs = ['RARL', 'SaAC']
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
                if fname is not None:
                    data_in_one_run_of_one_alg = {key: val[0::5] for key, val in data_in_one_run_of_one_alg.items()}
                    WINDOWSIZE = 5
                else:
                    data_in_one_run_of_one_alg = {key: val[:] for key, val in data_in_one_run_of_one_alg.items()}
                    WINDOWSIZE = 1
                data_in_one_run_of_one_alg.update(dict(algorithm=alg, task=task, num_run=num_run))
                df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
                for tag in tag2plot:
                    df_in_one_run_of_one_alg[tag+'_smo'] = df_in_one_run_of_one_alg[tag].rolling(WINDOWSIZE, min_periods=1).mean()
                df_list.append(df_in_one_run_of_one_alg)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    figsize = (12, 8)
    fontsize = 25

    if fname is not None:
        f2 = plt.figure(1, figsize=figsize)
        f2.add_axes([0.13, 0.12, 0.86, 0.87])
        ax2 = sns.boxplot(x="iteration", y="evaluation/episode_return", hue="task",
                     data=total_dataframe, palette=palette,)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylim(-10000, 50)
        handles, labels = ax2.get_legend_handles_labels()
        labels = lbs
        ax2.legend(handles=handles, labels=labels, loc='lower right', frameon=False, fontsize=fontsize)
        ax2.set_ylabel('Total Average Return', fontsize=fontsize)
        ax2.set_xlabel(r"Iteration $[\times 10^4]$", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig('./test_mean_std.pdf')
        plt.show()

    else:
        f1 = plt.figure(1, figsize=(12, 8))
        ax1 = f1.add_axes([0.13, 0.12, 0.86, 0.87])
        sns.lineplot(x="iteration", y="evaluation/episode_return_smo", hue="task",
                     data=total_dataframe, linewidth=2, palette=palette,)
        plt.ylim(-4000, 0)
        handles, labels = ax1.get_legend_handles_labels()
        labels = lbs
        ax1.legend(handles=handles, labels=labels, loc='lower right', frameon=False, fontsize=fontsize)
        ax1.set_ylabel('Total Average Return', fontsize=fontsize)
        ax1.set_xlabel(r"Iteration $[\times 10^4]$", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig('./test_total_return.pdf')
        plt.close(f1)

        f2 = plt.figure(1, figsize=(16, 8))
        ax2 = f2.add_axes([0.13, 0.12, 0.72, 0.87])
        legend_list = []
        for seed, df in enumerate(df_list):
            if df['task'][0] == 'adv_noise':
                ax2.plot(df['iteration'], df["evaluation/episode_return_smo"], linewidth=2)
                legend_list.append('seed' + " " + str(seed))
        plt.ylim(-10000, 0)
        handles, labels = ax1.get_legend_handles_labels()
        # ax2.legend(handles=handles, labels=legend_list, loc='upper right', frameon=False, fontsize=fontsize)
        ax2.legend(legend_list, fontsize=20, frameon=False, loc=(1.0, 0.2))
        ax2.set_ylabel('Total Average Return', fontsize=fontsize)
        ax2.set_xlabel(r"Iteration $[\times 10^4]$", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig('./rarl_every_return.pdf')
        plt.show()

        f3 = plt.figure(1, figsize=(16, 8))
        ax3 = f3.add_axes([0.13, 0.12, 0.72, 0.87])
        for df in df_list:
            if df['task'][0] == 'adv_noise_smooth':
                sns.lineplot(x=df['iteration'], y=df["evaluation/episode_return_smo"], linewidth=2, palette=palette, ax=ax3)
        plt.ylim(-8000, 0)
        handles, labels = ax1.get_legend_handles_labels()
        # ax3.legend(handles=handles, labels=labels, loc='lower right', frameon=False, fontsize=fontsize)
        ax3.set_ylabel('Total Average Return', fontsize=fontsize)
        ax3.set_xlabel(r"Iteration$[\times 10^4]$", fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig('./saac_every_return.pdf')
        plt.show()


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
    # plot_opt_results_of_all_alg_n_runs()
    plot_eval_results_of_all_alg_n_runs()
    # plot_robust_results_of_all_alg_n_runs()
    # print(compute_convergence_speed(-5.))
    # plot_trained_results_of_all_alg_n_runs(fname=None)