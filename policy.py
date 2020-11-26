#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import tensorflow as tf
import numpy as np
from gym import spaces
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from model import MLPNet

NAME2MODELCLS = dict([('MLP', MLPNet),])


class PolicyWithQs(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, obs_space, act_space, args):
        super().__init__()
        self.args = args
        assert isinstance(obs_space, spaces.Box)
        assert isinstance(act_space, spaces.Box)
        obs_dim = obs_space.shape[0] if args.obs_dim is None else self.args.obs_dim
        act_dim = act_space.shape[0] if args.act_dim is None else self.args.act_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                            NAME2MODELCLS[self.args.policy_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        self.policy_target = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy_target',
                                              output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='policy_adam_opt')

        self.Q1 = value_model_cls(obs_dim + act_dim, n_hiddens, n_units, hidden_activation, 1, name='Q1')
        self.Q1_target = value_model_cls(obs_dim + act_dim, n_hiddens, n_units, hidden_activation, 1, name='Q1_target')
        self.Q1_target.set_weights(self.Q1.get_weights())
        self.Q1_optimizer = self.tf.keras.optimizers.Adam(self.tf.keras.optimizers.schedules.PolynomialDecay(
            *self.args.value_lr_schedule), name='Q1_adam_opt')

        self.Q2 = value_model_cls(obs_dim + act_dim, n_hiddens, n_units, hidden_activation, 1, name='Q2')
        self.Q2_target = value_model_cls(obs_dim + act_dim, n_hiddens, n_units, hidden_activation, 1, name='Q2_target')
        self.Q2_target.set_weights(self.Q2.get_weights())
        self.Q2_optimizer = self.tf.keras.optimizers.Adam(self.tf.keras.optimizers.schedules.PolynomialDecay(
            *self.args.value_lr_schedule), name='Q2_adam_opt')

        if self.args.policy_only:
            self.target_models = ()
            self.models = (self.policy,)
            self.optimizers = (self.policy_optimizer,)
        else:
            if self.args.double_Q:
                assert self.args.target
                self.target_models = (self.Q1_target, self.Q2_target, self.policy_target,)
                self.models = (self.Q1, self.Q2, self.policy,)
                self.optimizers = (self.Q1_optimizer, self.Q2_optimizer, self.policy_optimizer,)
            elif self.args.target:
                self.target_models = (self.Q1_target, self.policy_target,)
                self.models = (self.Q1, self.policy,)
                self.optimizers = (self.Q1_optimizer, self.policy_optimizer,)
            else:
                self.target_models = ()
                self.models = (self.Q1, self.policy,)
                self.optimizers = (self.Q1_optimizer, self.policy_optimizer,)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        target_model_pairs = [(target_model.name, target_model) for target_model in self.target_models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + target_model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        target_model_pairs = [(target_model.name, target_model) for target_model in self.target_models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + target_model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')

    def get_weights(self):
        return [model.get_weights() for model in self.models] + \
               [model.get_weights() for model in self.target_models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            if i < len(self.models):
                self.models[i].set_weights(weight)
            else:
                self.target_models[i-len(self.models)].set_weights(weight)

    def apply_gradients(self, iteration, grads):
        if self.args.policy_only:
            policy_grad = grads
            self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
        else:
            if self.args.double_Q:
                q_weights_len = len(self.Q1.trainable_weights)
                policy_weights_len = len(self.policy.trainable_weights)
                q1_grad, q2_grad, policy_grad = grads[:q_weights_len], grads[q_weights_len:2*q_weights_len], \
                                                grads[2*q_weights_len:2*q_weights_len+policy_weights_len]
                self.Q1_optimizer.apply_gradients(zip(q1_grad, self.Q1.trainable_weights))
                self.Q2_optimizer.apply_gradients(zip(q2_grad, self.Q2.trainable_weights))
                if iteration % self.args.delay_update == 0:
                    self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
                    self.update_policy_target()
                    self.update_Q1_target()
                    self.update_Q2_target()
            else:
                q_weights_len = len(self.Q1.trainable_weights)
                policy_weights_len = len(self.policy.trainable_weights)
                q1_grad, policy_grad = grads[:q_weights_len], grads[q_weights_len:q_weights_len+policy_weights_len]
                self.Q1_optimizer.apply_gradients(zip(q1_grad, self.Q1.trainable_weights))
                if iteration % self.args.delay_update == 0:
                    self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))
                    if self.args.target:
                        self.update_policy_target()
                        self.update_Q1_target()

    def update_Q1_target(self):
        tau = self.args.tau
        source_params = self.Q1.get_weights()
        target_params = self.Q1_target.get_weights()
        self.Q1_target.set_weights([
            tau * source + (1.0 - tau) * target
            for source, target in zip(source_params, target_params)
        ])

    def update_Q2_target(self):
        tau = self.args.tau
        source_params = self.Q2.get_weights()
        target_params = self.Q2_target.get_weights()
        self.Q2_target.set_weights([
            tau * source + (1.0 - tau) * target
            for source, target in zip(source_params, target_params)
        ])

    def update_policy_target(self):
        tau = self.args.tau
        source_params = self.policy.get_weights()
        target_params = self.policy_target.get_weights()
        self.policy_target.set_weights([
            tau * source + (1.0 - tau) * target
            for source, target in zip(source_params, target_params)
        ])

    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        act_dist = self.tfd.MultivariateNormalDiag(mean, self.tf.exp(log_std))
        if self.args.action_range is not None:
            act_dist = (
                self.tfp.distributions.TransformedDistribution(
                    distribution=act_dist,
                    bijector=self.tfb.Chain(
                        [self.tfb.Affine(scale_identity_multiplier=self.args.action_range),
                         self.tfb.Tanh()])
                ))
        return act_dist

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_target_action(self, obs):
        with self.tf.name_scope('compute_target_action') as scope:
            logits = self.policy_target(obs)
            if self.args.deterministic_policy:
                mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
                return self.args.action_range * self.tf.tanh(mean) if self.args.action_range is not None else mean, 0.
            else:
                act_dist = self._logits2dist(logits)
                actions = act_dist.sample()
                logps = act_dist.log_prob(actions)
                return actions, logps

    @tf.function
    def compute_Q1(self, obs, act):
        with self.tf.name_scope('compute_Q1') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q1(Q_inputs), axis=1)

    @tf.function
    def compute_Q2(self, obs, act):
        with self.tf.name_scope('compute_Q2') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q2(Q_inputs), axis=1)

    @tf.function
    def compute_Q1_target(self, obs, act):
        with self.tf.name_scope('compute_Q1_target') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q1_target(Q_inputs), axis=1)

    @tf.function
    def compute_Q2_target(self, obs, act):
        with self.tf.name_scope('compute_Q2_target') as scope:
            Q_inputs = self.tf.concat([obs, act], axis=-1)
            return tf.squeeze(self.Q2_target(Q_inputs), axis=1)

    @property
    def log_alpha(self):
        return self.alpha_model.log_alpha


def test_policy():
    import gym
    from train_script import built_mixedpg_parser
    args = built_mixedpg_parser()
    print(args.obs_dim, args.act_dim)
    env = gym.make('PathTracking-v0')
    policy = PolicyWithQs(env.observation_space, env.action_space, args)
    obs = np.random.random((128, 6))
    act = np.random.random((128, 2))
    Qs = policy.compute_Qs(obs, act)
    print(Qs)

def test_policy2():
    from train_script import built_mixedpg_parser
    import gym
    args = built_mixedpg_parser()
    env = gym.make('Pendulum-v0')
    policy_with_value = PolicyWithQs(env.observation_space, env.action_space, args)

def test_policy_with_Qs():
    from train_script import built_mixedpg_parser
    import gym
    import numpy as np
    import tensorflow as tf
    args = built_mixedpg_parser()
    args.obs_dim = 3
    env = gym.make('Pendulum-v0')
    policy_with_value = PolicyWithQs(env.observation_space, env.action_space, args)
    # print(policy_with_value.policy.trainable_weights)
    # print(policy_with_value.Qs[0].trainable_weights)
    obses = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float32)

    with tf.GradientTape() as tape:
        acts, _ = policy_with_value.compute_action(obses)
        Qs = policy_with_value.compute_Qs(obses, acts)[0]
        print(Qs)
        loss = tf.reduce_mean(Qs)

    gradient = tape.gradient(loss, policy_with_value.policy.trainable_weights)
    print(gradient)

def test_mlp():
    import tensorflow as tf
    import numpy as np
    policy = tf.keras.Sequential([tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
                                  tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
                                  tf.keras.layers.Dense(1, activation='elu')])
    value = tf.keras.Sequential([tf.keras.layers.Dense(128, input_shape=(4,), activation='elu'),
                                  tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
                                  tf.keras.layers.Dense(1, activation='elu')])
    print(policy.trainable_variables)
    print(value.trainable_variables)
    with tf.GradientTape() as tape:
        obses = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float32)
        obses = tf.convert_to_tensor(obses)
        acts = policy(obses)
        a = tf.reduce_mean(acts)
        print(acts)
        Qs = value(tf.concat([obses, acts], axis=-1))
        print(Qs)
        loss = tf.reduce_mean(Qs)

    gradient = tape.gradient(loss, policy.trainable_weights)
    print(gradient)


if __name__ == '__main__':
    test_policy_with_Qs()
