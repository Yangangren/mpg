#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2022/06/07
# @Author  : Yangang Ren (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from env_build.utils.model import MLPNet, AttentionNet, ADVNet

NAME2MODELCLS = dict([('MLP', MLPNet), ('Attention', AttentionNet), ('Adversary', ADVNet)])


class AttentionPolicy4Toyota(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.noise_mode == 'adv_noise':
            obs_dim, act_dim, adv_act_dim = self.args.state_dim, self.args.act_dim, self.args.adv_act_dim
            n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
            value_model_cls, policy_model_cls, attn_model_cls, adv_policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                                                NAME2MODELCLS[self.args.policy_model_cls], \
                                                                NAME2MODELCLS[self.args.attn_model_cls], \
                                                                NAME2MODELCLS[self.args.adv_policy_model_cls]
            self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                           output_activation=self.args.policy_out_activation)
            policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
            self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

            self.adv_policy = adv_policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, adv_act_dim * 2, name='adv_policy',
                                           output_activation=self.args.adv_policy_out_activation)
            adv_policy_lr_schedule = PolynomialDecay(*self.args.adv_policy_lr_schedule)
            self.adv_policy_optimizer = self.tf.keras.optimizers.Adam(adv_policy_lr_schedule, name='adv_adam_opt')

            self.obj_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='obj_v',
                                         output_activation='softplus')
            obj_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
            self.obj_value_optimizer = self.tf.keras.optimizers.Adam(obj_value_lr_schedule, name='objv_adam_opt')

            # add AttentionNet
            attn_in_total_dim, attn_in_per_dim, attn_out_dim = self.args.attn_in_total_dim, \
                                                               self.args.attn_in_per_dim, \
                                                               self.args.attn_out_dim
            self.attn_net = attn_model_cls(attn_in_total_dim, attn_in_per_dim, attn_out_dim, name='attn_net')
            attn_lr_schedule = PolynomialDecay(*self.args.attn_lr_schedule)
            self.attn_optimizer = self.tf.keras.optimizers.Adam(attn_lr_schedule, name='adam_opt_attn')

            self.models = (self.obj_v, self.policy, self.attn_net, self.adv_policy)
            self.optimizers = (self.obj_value_optimizer, self.policy_optimizer, self.attn_optimizer, self.adv_policy_optimizer)

        else:
            obs_dim, act_dim = self.args.state_dim, self.args.act_dim
            n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
            value_model_cls, policy_model_cls, attn_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                                NAME2MODELCLS[self.args.policy_model_cls], \
                                                NAME2MODELCLS[self.args.attn_model_cls]
            self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                           output_activation=self.args.policy_out_activation)
            policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
            self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

            self.obj_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='obj_v',
                                         output_activation='softplus')

            obj_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
            self.obj_value_optimizer = self.tf.keras.optimizers.Adam(obj_value_lr_schedule, name='objv_adam_opt')

            # add AttentionNet
            attn_in_total_dim, attn_in_per_dim, attn_out_dim = self.args.attn_in_total_dim, \
                                                               self.args.attn_in_per_dim, \
                                                               self.args.attn_out_dim
            self.attn_net = attn_model_cls(attn_in_total_dim, attn_in_per_dim, attn_out_dim, name='attn_net')
            attn_lr_schedule = PolynomialDecay(*self.args.attn_lr_schedule)
            self.attn_optimizer = self.tf.keras.optimizers.Adam(attn_lr_schedule, name='adam_opt_attn')

            self.models = (self.obj_v, self.policy, self.attn_net)
            self.optimizers = (self.obj_value_optimizer, self.policy_optimizer, self.attn_optimizer)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        obj_v_len = len(self.obj_v.trainable_weights)
        pg_len = len(self.policy.trainable_weights)
        attn_len = len(self.attn_net.trainable_weights)
        obj_v_grad, policy_grad, attn_grad, adv_policy_grad = grads[:obj_v_len], grads[obj_v_len:obj_v_len+pg_len], \
                                                              grads[obj_v_len + pg_len:obj_v_len + pg_len + attn_len], \
                                                              grads[obj_v_len + pg_len + attn_len:]
        self.obj_value_optimizer.apply_gradients(zip(obj_v_grad, self.obj_v.trainable_weights))
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
        self.attn_optimizer.apply_gradients(zip(attn_grad, self.attn_net.trainable_weights))
        if (self.args.noise_mode == 'adv_noise') and (iteration % self.args.update_adv_interval == 0):
            self.adv_policy_optimizer.apply_gradients(zip(adv_policy_grad, self.adv_policy.trainable_weights))

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        act_dist = self.tfp.distributions.TransformedDistribution(distribution=act_dist,
                                                                  bijector=self.tfb.Tanh())
        return act_dist

    def _logits2dist_adv(self, cyc_logits, ped_logits, veh_logits):
        cyc_mean, cyc_log_std = self.tf.split(cyc_logits, num_or_size_splits=2, axis=-1)
        cyc_mean = tf.tile(cyc_mean, [1, self.args.bike_num])
        # cyc_mean = tf.zeros_like(cyc_mean)
        cyc_log_std = tf.tile(cyc_log_std, [1, self.args.bike_num])

        ped_mean, ped_log_std = self.tf.split(ped_logits, num_or_size_splits=2, axis=-1)
        ped_mean = tf.tile(ped_mean, [1, self.args.ped_num])
        ped_log_std = tf.tile(ped_log_std, [1, self.args.ped_num])

        veh_mean, veh_log_std = self.tf.split(veh_logits, num_or_size_splits=2, axis=-1)
        veh_mean = tf.tile(veh_mean, [1, self.args.veh_num])
        veh_log_std = tf.tile(veh_log_std, [1, self.args.veh_num])

        mean = tf.concat([cyc_mean, ped_mean, veh_mean], axis=-1)
        log_std = tf.concat([cyc_log_std, ped_log_std, veh_log_std], axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))

        act_dist = self.tfp.distributions.TransformedDistribution(distribution=act_dist,
                                                                  bijector=self.tfb.Tanh())
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_obj_v(self, obs):
        with self.tf.name_scope('compute_obj_v') as scope:
            return tf.squeeze(self.obj_v(obs), axis=1)

    @tf.function
    def compute_adv_action(self, obs, mask):
        with self.tf.name_scope('compute_adv_action') as scope:
            cyc_logits, ped_logits, veh_logits = self.adv_policy(obs)
            if self.args.adv_deterministic_policy:
                cyc_mean, cyc_log_std = self.tf.split(cyc_logits, num_or_size_splits=2, axis=-1)
                ped_mean, ped_log_std = self.tf.split(ped_logits, num_or_size_splits=2, axis=-1)
                veh_mean, veh_log_std = self.tf.split(veh_logits, num_or_size_splits=2, axis=-1)
                mean = tf.concat([cyc_mean, ped_mean, veh_mean], axis=-1)
                return mean, 0.
            else:
                mask = self.tf.reshape(mask, (-1, self.args.other_number, 1))
                act_dist = self._logits2dist_adv(cyc_logits, ped_logits, veh_logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                actions_reshape = self.tf.reshape(actions, (-1, self.args.other_number, self.args.adv_act_dim))
                action_temp = tf.where(mask, actions_reshape, self.tf.stop_gradient(actions_reshape))
                actions = self.tf.reshape(action_temp, (-1, self.args.other_number * self.args.adv_act_dim))
            return actions, logps

    @tf.function
    def compute_attn(self, obs_others, mask):
        with self.tf.name_scope('compute_attn') as scope:
            return self.attn_net([obs_others, mask]) # return (logits, weights) tuple


if __name__ == '__main__':
    pass
