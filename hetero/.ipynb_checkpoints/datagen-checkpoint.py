'''
datagen.py generates data in simulations.

Most variables and functions are self-explanatory in their names, indicating their intended purposes.

Comments are provided for some functions or variables that are not self-explanatory.
'''

from typing import Iterable, Any

import attrs
import numpy as np
from scipy.stats import norm as normal_distribution
from hetero.bases import bspline_expand, legendre_expand, sine_expand
from hetero.config import DTYPE
from hetero.policies import UniformRandomPolicy
from hetero.utils import LabelPartitioner


@(attrs.frozen)
class SARSDataWithLabels(object):
    current_features: np.array
    actions: np.array
    rewards: np.array
    next_features: np.array
    labels: Iterable[Any]
    partitioner: LabelPartitioner
    time_step_indices: Iterable[int]

    def num_actions(self):
        return self.actions.shape[1]

    def num_feature_dims(self):
        return self.current_features.shape[1]

    def num_beta_dims(self):
        return self.num_actions() * self.num_feature_dims()

    def J(self):
        return self.num_feature_dims()

    def M(self):
        return self.num_actions()

    def N(self):
        return self.partitioner.num_unique_labels()

    def NT(self):
        return self.current_features.shape[0]

    def replace_labels(self, new_labels):
        if len(new_labels) == self.NT():
            print(f"new_labels.length={len(new_labels)} matches number of records")
            new_partitioner = LabelPartitioner(new_labels)
            return SARSDataWithLabels(
                self.current_features,
                self.actions,
                self.rewards,
                self.next_features,
                new_labels,
                new_partitioner,
                self.time_step_indices,
            )
        else:
            print(f"new_labels.length={len(new_labels)} matches num_unique_labels")
            assert len(new_labels) == self.partitioner.num_unique_labels()
            converter = dict(zip(self.partitioner.index_to_label_mapping(), new_labels))
            new_labels = [converter[x] for x in self.labels]
            return self.replace_labels(new_labels)


def _apply_policy(config, states, noise, pi):
    actions = pi.action(states)
    actions_0, actions_1 = np.split(actions, 2, axis=-1)
    actions_0, actions_1 = actions_0.squeeze(), actions_1.squeeze()
    trans_actions = actions_1 - actions_0
    scale = config.action_transition_coeff * trans_actions
    return (
        actions,
        trans_actions,
        np.stack([states[:, :, 0] * scale, states[:, :, 1] * (-scale)], axis=2) + noise,
    )


def generate_data_from_config(config, pi_sample=None):
    rand = np.random.RandomState(config.seed)
    pi_sample = pi_sample or UniformRandomPolicy(config.NUM_ACTIONS, rand)

    state_shape_per_step = [
        config.NUM_GROUPS,
        config.num_trajectories,
        config.NUM_STATE_DIMS,
    ]
    init_state = rand.normal(
        config.init_state_mu, config.init_state_sigma, size=state_shape_per_step
    ).astype(DTYPE)
    noise_shape = [config.num_total_steps()] + state_shape_per_step
    
    match config.noise_type:
        case "NORMAL":
            noise = rand.normal(0.0, config.noise_sigma, size=noise_shape).astype(DTYPE)
        case "STUDENT":
            noise = rand.standard_t(config.noise_student_degree, size=noise_shape) * config.noise_sigma
        case _:
            raise ValueError(
                'noise_type can only be ["NORMAL", "STUDENT"], but got: '
                + config.noise_type
            )

    if config.group_reward_coeff_override is None:
        group_reward_coeff = rand.uniform(
            low=config.group_reward_coeff_range[0],
            high=config.group_reward_coeff_range[1],
            size=[config.NUM_GROUPS, config.J()],
        ).astype(DTYPE)
        print(f"Generated group_reward_coeff={group_reward_coeff}")
    else:
        group_reward_coeff = config.group_reward_coeff_override

    states = [init_state]
    rewards = []
    assert config.NUM_STATE_DIMS == 2 and config.NUM_ACTIONS == 2

    actions = []
    trans_actions = []
    for t in range(config.num_total_steps()):
        actions_t, trans_actions_t, new_state = _apply_policy(
            config, states[-1], noise[t, :, :, :], pi_sample
        )
        actions.append(actions_t)
        trans_actions.append(trans_actions_t)
        states.append(new_state)

    match config.basis_expansion_method:
        case "NONE":
            features = states
        case "BSPLINE":
            features = bspline_expand(states, config)
        case "LEGENDRE":
            features = legendre_expand(states, config)
        case "SINE":
            features = sine_expand(states, config)
        case _:
            raise ValueError(
                'basis_expansion_method can only be ["NONE", "BSPLINE"], but got: '
                + config.basis_expansion_method
            )

    match config.transformation_method:
        case "NONE":
            pass
        case "NORMCDF":
            for i in range(len(features)):
                features[i] = normal_distribution.cdf(features[i])
        case _:
            raise ValueError(
                'transformation_method can only be ["NONE", "NORMCDF"], but got: '
                + config.transformation_method
            )

    for t in range(config.num_total_steps()):
        reward_t = []
        for k in range(config.NUM_GROUPS):
            feature_t_k = features[t][k, :, :]
            b_k = group_reward_coeff[k, :]
            if config.use_01_action_values:
                action_values = (trans_actions[t][k, :] + 1) / 2
            else:
                action_values = trans_actions[t][k, :]
            reward_t.append(
                (feature_t_k * b_k).sum(axis=1)
                + config.action_reward_coeff[k] * action_values
            )
        rewards.append(np.stack(reward_t, axis=0))

    current_features_no_intercept = np.stack(
        features[config.num_burnin_steps : -1], axis=0
    ).reshape([-1, config.num_feature_dims_without_intercept()])
    current_features = np.ones(
        [current_features_no_intercept.shape[0], config.num_feature_dims()], dtype=DTYPE
    )
    current_features[
        :, : config.num_feature_dims_without_intercept()
    ] = current_features_no_intercept

    next_features_no_intercept = np.stack(
        features[(config.num_burnin_steps + 1) :], axis=0
    ).reshape([-1, config.num_feature_dims_without_intercept()])
    next_features = np.ones(
        [next_features_no_intercept.shape[0], config.num_feature_dims()], dtype=DTYPE
    )
    next_features[
        :, : config.num_feature_dims_without_intercept()
    ] = next_features_no_intercept

    actions = np.stack(actions[config.num_burnin_steps :], axis=0).reshape(
        [-1, config.NUM_ACTIONS]
    )
    rewards = np.stack(rewards[config.num_burnin_steps :], axis=0).reshape([-1])
    labels = [
        divmod(
            i % (config.NUM_GROUPS * config.num_trajectories), config.num_trajectories
        )
        for i in range(current_features.shape[0])
    ]

    return SARSDataWithLabels(
        current_features,
        actions,
        rewards,
        next_features,
        labels,
        LabelPartitioner(labels),
        [
            i // (config.NUM_GROUPS * config.num_trajectories)
            for i in range(current_features.shape[0])
        ],
    )


def extract_first_n_order(data, order, config):
    current_features = data.current_features[:, : (config.NUM_STATE_DIMS * order)]
    if config.add_intercept_column:
        current_features = np.concatenate(
            [current_features, np.ones([current_features.shape[0], 1], dtype=DTYPE)],
            axis=1,
        )
    next_features = data.next_features[:, : (config.NUM_STATE_DIMS * order)]
    if config.add_intercept_column:
        next_features = np.concatenate(
            [next_features, np.ones([next_features.shape[0], 1], dtype=DTYPE)], axis=1
        )

    return SARSDataWithLabels(
        current_features,
        data.actions,
        data.rewards,
        next_features,
        data.labels,
        data.partitioner,
        data.time_step_indices,
    )
