#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: buffer.py
# =====================================

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from utils.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, args, buffer_id):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        """
        self.args = args
        self.buffer_id = buffer_id
        self._storage = {'left': [], 'straight': [], 'right': []}
        self._storage_idx = {'left': 0, 'straight': 0, 'right': 0}
        self._maxsize = self.args.max_buffer_size // len(self._storage_idx.keys())
        self.replay_starts = self.args.replay_starts
        self.replay_batch_size = self.args.replay_batch_size
        self.stats = {}
        self.replay_times = 0
        logger.info('Buffer initialized')

    def get_stats(self):
        self.stats.update(dict(storage=self.__len__()))
        return self.stats

    def __len__(self):
        return sum([len(item) for item in self._storage.values()])

    def add(self, obs_ego_next, obs_others_next, veh_num_next, done, ref_index, task, weight):
        data = (obs_ego_next, obs_others_next, veh_num_next, done, ref_index)
        if self._storage_idx[task] >= len(self._storage[task]):
            self._storage[task].append(data)
        else:
            self._storage[task][self._storage_idx[task]] = data

        self._storage_idx[task] = (self._storage_idx[task] + 1) % self._maxsize
        # if self._storage_idx[task] == 0:
        #     print(self.buffer_id, task)
        #     print([len(item) for item in self._storage.values()])
        #     print(self._storage_idx.values())
        #     # print(self.__len__())

    def _encode_sample(self, idxes_dict):
        obses_ego_next, obses_other_next, vehs_num_next, dones, ref_indexs = [], [], [], [], []
        for task, value in idxes_dict.items():
            for i in value:
                data = self._storage[task][i]
                obs_ego_next, obs_other_next, veh_num_next, done, ref_index = data
                obses_ego_next.append(np.array(obs_ego_next, copy=False))
                obses_other_next.append(np.array(obs_other_next, copy=False))
                vehs_num_next.append(veh_num_next)
                dones.append(done)
                ref_indexs.append(ref_index)
        obses_others_next = np.concatenate(([obses_other_next[i] for i in range(len(obses_other_next))]), axis=0)
        # print(vehs_mode_next.shape, obses_others_next.shape, np.sum(np.array(vehs_num_next)))

        return np.array(obses_ego_next), np.array(obses_others_next), np.array(vehs_num_next), \
               np.array(dones), np.array(ref_indexs)

    def sample_idxes(self, batch_size):
        idx_dict = {'left': [], 'straight': [], 'right': []}
        for task in idx_dict.keys():
            idx_dict[task].extend([random.randint(0, len(self._storage[task]) - 1) for _ in range(batch_size // 3)])
        return idx_dict

    def sample_with_idxes(self, idxes):
        return list(self._encode_sample(idxes)) + [idxes,]

    def sample(self, batch_size):
        idxes = self.sample_idxes(batch_size)
        return self.sample_with_idxes(idxes)

    def add_batch(self, batch):
        for task, values in batch.items():
            for trans in values:
                self.add(*trans, task, 0)

    def replay(self):
        if self.__len__() < self.replay_starts:
            return None
        if self.buffer_id == 1 and self.replay_times % self.args.buffer_log_interval == 0:
            logger.info('Buffer info: {}, Elements info {}'.format(self.get_stats(), self._storage_idx.values()))

        self.replay_times += 1
        return self.sample(self.replay_batch_size)
