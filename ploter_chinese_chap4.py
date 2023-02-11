#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2023/01/26
# @Author  : Yangang Ren (Tsinghua Univ.)
# @Modification: finish the ploter for thesis figures
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
import tensorboard.backend.application
NAME2POLICIES = dict([('Policy4Toyota', Policy4Toyota)])
NAME2EVALUATORS = dict([('Evaluator', Evaluator)])
import matplotlib.font_manager as fm
zhfont1 = fm.FontProperties(fname=r'D:\simsun.ttf',size=14)

plt.rc('font', family='Times New Roman')
# plt.rcParams['mathtext.fontset'] = 'stix'
import matplotlib
matplotlib.rcParams['mathtext.default'] = 'regular'

sns.set(style=None)

palette = [(1.0, 0.48627450980392156, 0.0),
                    (0.9098039215686274, 0.0, 0.043137254901960784),
                    (0.5450980392156862, 0.16862745098039217, 0.8862745098039215),
                    (0.6235294117647059, 0.2823529411764706, 0.0),]

WINDOWSIZE = 5


def min_n(inp_list, n):
    return sorted(inp_list)[:n]


def plot_eva_results_of_all_alg_n_runs(dirs_dict_for_plot=None):
    tag2plot = ['Evaluation/1. TAR-RL iter', 'Loss/Actor loss-RL iter', 'Loss/collision2v-RL iter']
    env_list = ['idsim']
    task_list = ['SAAC']
    palette = "bright"
    lbs = ['SAAC']
    dir_str = './results/{}/{}'
    df_list = []
    for alg in env_list:
        for task in task_list:
            data2plot_dir = dir_str.format(alg, task)
            data2plot_dirs_list = dirs_dict_for_plot[alg] if dirs_dict_for_plot is not None else os.listdir(data2plot_dir)
            for num_run, dir in enumerate(data2plot_dirs_list):
                opt_dir = data2plot_dir + '/' + dir
                opt_file = os.path.join(opt_dir,
                                        [file_name for file_name in os.listdir(opt_dir) if
                                         file_name.startswith('events')][0])
                data_in_one_run_of_one_alg = {key: [] for key in tag2plot}
                data_in_one_run_of_one_alg.update({'iteration': []})
                data = read_tensorboard(opt_file)
                for tag, value in data.items():
                    if tag in tag2plot:
                        data_in_one_run_of_one_alg[tag] = value['y']
                        data_in_one_run_of_one_alg['iteration'] = value['x']
                data_in_one_run_of_one_alg['iteration'] = data_in_one_run_of_one_alg['iteration'] / 10000
                data_in_one_run_of_one_alg = {key: val[0::5] for key, val in data_in_one_run_of_one_alg.items()}
                data_in_one_run_of_one_alg.update(dict(algorithm=alg, task=task, num_run=num_run))
                data_in_one_run_of_one_alg['total_loss'] = data_in_one_run_of_one_alg['Loss/Actor loss-RL iter'] + data_in_one_run_of_one_alg['Loss/collision2v-RL iter']
                df_in_one_run_of_one_alg = pd.DataFrame(data_in_one_run_of_one_alg)
                for tag in tag2plot + ['total_loss']:
                    df_in_one_run_of_one_alg[tag+'_smo'] = df_in_one_run_of_one_alg[tag].rolling(WINDOWSIZE, min_periods=1).mean()
                    if tag == 'Loss/collision2v-RL iter':
                        df_in_one_run_of_one_alg[tag + '_smo'] = df_in_one_run_of_one_alg[tag + '_smo'] * 3.0
                df_list.append(df_in_one_run_of_one_alg)
    total_dataframe = df_list[0].append(df_list[1:], ignore_index=True) if len(df_list) > 1 else df_list[0]
    figsize = (10, 8)
    fontsize = 24

    font_legend = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': fontsize,
        # 'style': 'italic'  # 使字变斜
        # 'usetex' : True,  # legend 无得设 `usetex` 这项
    }

    f1 = plt.figure(1, figsize=figsize)
    ax1 = f1.add_axes([0.10, 0.115, 0.85, 0.87])
    sns.lineplot(x="iteration", y="total_loss"+'_smo', hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    # plt.ylim(0, 40)
    # plt.xlim(0, 20)
    ax1.set_ylabel('策略训练损失', fontproperties=zhfont1, fontsize=fontsize)
    ax1.set_xlabel(r"迭代次数 $[\times 10^4]$", fontproperties=zhfont1, fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize - 4)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize - 4)
    plt.savefig('./chap04_loss.pdf')

    f2 = plt.figure(2, figsize=figsize)
    ax2 = f2.add_axes([0.12, 0.12, 0.845, 0.86])
    sns.lineplot(x="iteration", y="Evaluation/1. TAR-RL iter"+'_smo', hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    # plt.ylim(0, 400)
    # plt.xlim(0, 20)
    ax2.set_ylabel('平均累计回报', fontproperties=zhfont1, fontsize=fontsize)
    ax2.set_xlabel(r"迭代次数 $[\times 10^4]$", fontproperties=zhfont1, fontsize=fontsize)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize - 4)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize - 4)
    plt.savefig('./chap04_tar.pdf')

    f3 = plt.figure(3, figsize=figsize)
    ax3 = f3.add_axes([0.11, 0.12, 0.85, 0.87])
    sns.lineplot(x="iteration", y="Loss/collision2v-RL iter"+'_smo', hue="task",
                 data=total_dataframe, linewidth=2, palette=palette, legend=False)
    handles, labels = ax3.get_legend_handles_labels()
    labels = lbs
    # ax3.legend(handles=handles, labels=labels, loc='upper right', frameon=False, fontsize=fontsize, prop=font_legend)
    ax3.set_ylabel('约束性能', fontproperties=zhfont1, fontsize=fontsize)
    ax3.set_xlabel(r"迭代次数 $[\times 10^4]$", fontproperties=zhfont1, fontsize=fontsize)
    # plt.xlim(0, 20)
    plt.yticks(fontproperties='Times New Roman', fontsize=fontsize - 4)
    plt.xticks(fontproperties='Times New Roman', fontsize=fontsize - 4)
    plt.savefig('./chap04_loss_penalty.pdf')


def read_tensorboard(path):
    """
    input the dir of the tensorboard log
    """
    import tensorboard
    from tensorboard.backend.event_processing import event_accumulator

    tensorboard.backend.application.logger.setLevel("ERROR")
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    # print("All available keys in Tensorboard", ea.scalars.Keys())
    valid_key_list = ea.scalars.Keys()

    output_dict = dict()
    for key in valid_key_list:
        event_list = ea.scalars.Items(key)
        x, y = [], []
        for e in event_list:
            x.append(e.step)
            y.append(e.value)

        data_dict = {"x": np.array(x), "y": np.array(y)}
        output_dict[key] = data_dict
    return output_dict


if __name__ == "__main__":
    plot_eva_results_of_all_alg_n_runs()